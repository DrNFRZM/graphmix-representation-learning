"""Behavioural tests of the encoders: what each one is allowed to look at."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch.nn.functional as F  # noqa: E402
from torch_geometric.nn import SAGEConv  # noqa: E402

from graphmix.graphs import (  # noqa: E402
    Graph,
    bipartite_graph,
    distance_features,
    graph_from_neighbors,
    knn_neighbors,
    random_graph,
    selfloop_graph,
)
from graphmix.models import (  # noqa: E402
    BipartiteSAGEEncoder,
    Decoder,
    GraphTensors,
    MLPEncoder,
    SAGEEncoder,
    build_encoder,
    flat_input,
    mean_operator,
    reconstruction_loss,
)
from graphmix.train import to_tensors  # noqa: E402

N_NUM, CARDS, HIDDEN, DIM = 4, [3, 2, 5], 16, 6
KINDS = ["mlp_ae", "sage_selfloop", "sage_rand", "sage_knn", "sage_bip", "sage_bip_nopool"]
CPU = torch.device("cpu")


def toy(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x_num = torch.tensor(rng.normal(size=(n, N_NUM)), dtype=torch.float32)
    x_cat = torch.tensor(np.stack([rng.integers(0, c, n) for c in CARDS], axis=1))
    return x_num, x_cat


def masks(n: int, p: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_NUM, generator=gen) < p, torch.rand(n, len(CARDS), generator=gen) < p


def make_graph(kind: str, x_num: torch.Tensor, x_cat: torch.Tensor, n_train: int) -> Graph | None:
    n = len(x_num)
    if kind == "mlp_ae":
        return None
    if kind == "sage_selfloop":
        return selfloop_graph(n_train, n)
    if kind == "sage_rand":
        return random_graph(n_train, n, 4, seed=0)
    if kind == "sage_knn":
        dist = distance_features(x_num.numpy(), x_cat.numpy(), CARDS)
        return graph_from_neighbors(knn_neighbors(dist, n_train, 4), n_train)
    return bipartite_graph(x_cat.numpy(), CARDS, n_train)


def encoder(kind: str) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_encoder(kind, N_NUM, CARDS, HIDDEN, DIM).eval()


# --------------------------------------------------------------------------- shapes and inputs


@pytest.mark.parametrize("kind", KINDS)
def test_encoder_output_shape(kind: str) -> None:
    n, n_train = 50, 30
    x_num, x_cat = toy(n)
    g = to_tensors(make_graph(kind, x_num, x_cat, n_train), n, n_train, CPU)
    z = encoder(kind)(x_num, x_cat, *masks(n, 0.3), g)
    assert z.shape == (n, DIM) and torch.isfinite(z).all()


def test_flat_input_layout() -> None:
    x_num = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    x_cat = torch.tensor([[2, 1, 4]])
    m_num = torch.tensor([[False, True, False, False]])
    m_cat = torch.tensor([[False, True, False]])
    flat = flat_input(x_num, x_cat, m_num, m_cat, CARDS)[0]
    assert flat.shape == (2 * N_NUM + sum(CARDS),)
    assert flat[:N_NUM].tolist() == [1.0, 0.0, 3.0, 4.0]  # hidden value replaced by 0
    assert flat[N_NUM : 2 * N_NUM].tolist() == [0.0, 1.0, 0.0, 0.0]  # ... and flagged
    onehot = flat[2 * N_NUM :].tolist()
    assert onehot == [0, 0, 1] + [0, 0] + [0, 0, 0, 0, 1]  # hidden column is all zeros


@pytest.mark.parametrize("kind", KINDS)
def test_masked_cells_are_invisible_to_every_encoder(kind: str) -> None:
    n, n_train = 50, 30
    x_num, x_cat = toy(n)
    m_num, m_cat = masks(n, 0.4, seed=3)
    g = to_tensors(make_graph(kind, x_num, x_cat, n_train), n, n_train, CPU)
    enc = encoder(kind)
    base = enc(x_num, x_cat, m_num, m_cat, g)
    x_num2 = torch.where(m_num, x_num + 100.0, x_num)
    cards = torch.tensor(CARDS)
    x_cat2 = torch.where(m_cat, (x_cat + 1) % cards, x_cat)
    assert not torch.equal(x_num, x_num2) and not torch.equal(x_cat, x_cat2)
    assert torch.equal(base, enc(x_num2, x_cat2, m_num, m_cat, g))


def test_mlp_ignores_the_graph_argument() -> None:
    n = 20
    x_num, x_cat = toy(n)
    m = masks(n, 0.2)
    enc = encoder("mlp_ae")
    a = enc(x_num, x_cat, *m, GraphTensors(None, torch.ones(n, dtype=torch.bool)))
    b = enc(x_num, x_cat, *m, to_tensors(selfloop_graph(n, n), n, n, CPU))
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- message passing


def adjacency(edges: torch.Tensor, n: int) -> torch.Tensor:
    """Mean operator of a graph given as an edge list (row 0 = sender, row 1 = receiver)."""
    adj = to_tensors(Graph("knn", n, n, edge_index=edges.numpy()), n, n, CPU).adj
    assert adj is not None
    return adj


def chain_input(n: int = 5) -> tuple[torch.Tensor, torch.Tensor, GraphTensors]:
    x_num, x_cat = toy(n, seed=1)
    edges = torch.tensor([list(range(n - 1)), list(range(1, n))])  # 0 -> 1 -> 2 -> ...
    return x_num, x_cat, GraphTensors(adjacency(edges, n), torch.ones(n, dtype=torch.bool))


def test_information_flows_along_edges_for_two_hops_only() -> None:
    x_num, x_cat, g = chain_input()
    m = masks(5, 0.0)
    enc = encoder("sage_knn")
    base = enc(x_num, x_cat, *m, g)

    x_first = x_num.clone()
    x_first[0] += 5.0
    changed = ~torch.isclose(base, enc(x_first, x_cat, *m, g), atol=1e-6).all(dim=1)
    assert changed.tolist() == [True, True, True, False, False]  # node 0 reaches nodes 1 and 2

    x_last = x_num.clone()
    x_last[4] += 5.0
    changed = ~torch.isclose(base, enc(x_last, x_cat, *m, g), atol=1e-6).all(dim=1)
    assert changed.tolist() == [False, False, False, False, True]  # the last node sends nothing


def test_sage_layer_is_root_weight_plus_mean_of_senders() -> None:
    n = 6
    x_num, x_cat = toy(n, seed=2)
    edges = torch.tensor([[0, 1, 2, 0], [3, 3, 3, 4]])  # node 3 hears from 0, 1, 2; node 4 from 0
    m = masks(n, 0.0)
    enc = SAGEEncoder(N_NUM, CARDS, HIDDEN, DIM)
    x = flat_input(x_num, x_cat, *m, CARDS)
    agg = torch.zeros_like(x)
    agg[3] = x[[0, 1, 2]].mean(dim=0)
    agg[4] = x[0]
    expected = enc.conv1.lin_l(agg) + enc.conv1.lin_r(x)  # PyG's SAGEConv with mean aggregation
    assert torch.allclose(enc.conv1(x, adjacency(edges, n)), expected, atol=1e-5)


def test_csr_sum_aggregation_equals_pyg_mean_on_an_edge_list() -> None:
    """The speed-up (aggr="sum" on a row-normalised CSR matrix) must not change the model:
    compare with PyG's own mean aggregator, including repeated edges and nodes without senders."""
    n = 8
    x = torch.randn(n, 6)
    edges = torch.tensor([[0, 0, 1, 2, 5, 5, 5], [3, 3, 3, 4, 6, 6, 7]])  # 0 -> 3 and 5 -> 6 twice
    fast = SAGEConv(6, 4, aggr="sum")
    slow = SAGEConv(6, 4, aggr="mean")
    slow.load_state_dict(fast.state_dict())
    assert torch.allclose(fast(x, adjacency(edges, n)), slow(x, edges), atol=1e-6)


def test_selfloop_sage_is_an_mlp_with_two_summed_weights() -> None:
    """On a self-loop graph both layers compute (W_root + W_neigh) h + b: the function class of the
    MLP. The two matrices get identical gradients, so under Adam (which acts elementwise) their sum
    moves twice as far per step as the MLP's single matrix (docs/research_note.md, section 9)."""
    n = 12
    x_num, x_cat = toy(n, seed=3)
    m = masks(n, 0.3, seed=3)
    sage = SAGEEncoder(N_NUM, CARDS, HIDDEN, DIM)
    mlp = MLPEncoder(N_NUM, CARDS, HIDDEN, DIM)
    with torch.no_grad():
        for layer, conv in ((mlp.net[0], sage.conv1), (mlp.net[2], sage.conv2)):
            layer.weight.copy_(conv.lin_l.weight + conv.lin_r.weight)
            layer.bias.copy_(conv.lin_l.bias)
    g = to_tensors(selfloop_graph(n, n), n, n, CPU)
    out = sage(x_num, x_cat, *m, g)
    assert torch.allclose(out, mlp(x_num, x_cat, *m, GraphTensors(adj=None, is_train=g.is_train)), atol=1e-5)
    out.pow(2).sum().backward()
    for conv in (sage.conv1, sage.conv2):
        assert conv.lin_l.weight.grad is not None and conv.lin_r.weight.grad is not None
        assert torch.allclose(conv.lin_l.weight.grad, conv.lin_r.weight.grad, atol=1e-6)


@pytest.mark.parametrize("kind", ["sage_selfloop", "sage_rand", "sage_knn", "sage_bip", "sage_bip_nopool"])
def test_training_nodes_cannot_tell_whether_evaluation_nodes_exist(kind: str) -> None:
    n, n_train = 70, 45
    x_num, x_cat = toy(n)
    m_num, m_cat = masks(n, 0.3, seed=5)
    graph = make_graph(kind, x_num, x_cat, n_train)
    assert graph is not None
    enc = encoder(kind)
    full = enc(x_num, x_cat, m_num, m_cat, to_tensors(graph, n, n_train, CPU))
    train_only = enc(
        x_num[:n_train],
        x_cat[:n_train],
        m_num[:n_train],
        m_cat[:n_train],
        to_tensors(graph.restrict_to_train(), n_train, n_train, CPU),
    )
    assert torch.allclose(full[:n_train], train_only, atol=1e-5)


# --------------------------------------------------------------------------- bipartite encoder


def test_bipartite_evaluation_rows_do_not_change_other_rows() -> None:
    n, n_train = 60, 40
    x_num, x_cat = toy(n)
    m = masks(n, 0.0)
    g = to_tensors(make_graph("sage_bip", x_num, x_cat, n_train), n, n_train, CPU)
    enc = encoder("sage_bip")
    base = enc(x_num, x_cat, *m, g)

    eval_row, others = 50, [i for i in range(n) if i != 50]
    moved = x_num.clone()
    moved[eval_row] += 3.0
    after = enc(moved, x_cat, *m, g)
    assert torch.equal(base[others], after[others])  # an evaluation row sends nothing anywhere
    assert not torch.equal(base[eval_row], after[eval_row])

    train_row = 5
    moved = x_num.clone()
    moved[train_row] += 3.0
    after = enc(moved, x_cat, *m, g)
    other_changed = (~torch.isclose(base, after, atol=1e-7)).any(dim=1)
    other_changed[train_row] = False
    assert other_changed.any()  # training records do reach the records that share a value with them


def bipartite_with_edge_lists(
    enc: BipartiteSAGEEncoder,
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    m_num: torch.Tensor,
    m_cat: torch.Tensor,
    is_train: torch.Tensor,
) -> torch.Tensor:
    """The bipartite forward pass written with PyG's mean aggregator on edge lists (what the
    encoder computed before it switched to sparse matrices)."""

    def mean_conv(conv: SAGEConv) -> SAGEConv:
        ref = SAGEConv(conv.in_channels, conv.out_channels, aggr="mean")
        ref.load_state_dict(conv.state_dict())
        return ref

    x_rec = torch.cat([x_num.masked_fill(m_num, 0.0), m_num.to(x_num.dtype)], dim=1)
    rec, col = (~m_cat).nonzero(as_tuple=True)
    val = x_cat[rec, col] + enc.offsets[col]
    to_rec = torch.stack([val, rec])
    sender = is_train[rec]
    to_val = torch.stack([rec[sender], val[sender]])
    emb = enc.value_emb.weight
    h_rec = F.relu(mean_conv(enc.rec_from_val)((emb, x_rec), to_rec))
    h_val = F.relu(mean_conv(enc.val_from_rec)((x_rec, emb), to_val))
    return mean_conv(enc.out)((h_val, h_rec), to_rec)


def test_bipartite_encoder_matches_the_edge_list_formulation() -> None:
    n, n_train = 50, 30
    x_num, x_cat = toy(n)
    m_num, m_cat = masks(n, 0.4, seed=2)
    m_cat[7] = True  # a record with no visible categorical cell at all
    g = to_tensors(make_graph("sage_bip", x_num, x_cat, n_train), n, n_train, CPU)
    enc = encoder("sage_bip")
    assert isinstance(enc, BipartiteSAGEEncoder)
    expected = bipartite_with_edge_lists(enc, x_num, x_cat, m_num, m_cat, g.is_train)
    assert torch.allclose(enc(x_num, x_cat, m_num, m_cat, g), expected, atol=1e-5)


def test_bipartite_control_learns_nothing_from_other_records() -> None:
    """With pool_records=False the value nodes receive nothing, so a record's code depends on its own
    cells only; with the pooling it also depends on the numeric features of the records that share its
    values. Both have the same parameters (that is what makes the first a control for the second)."""
    n, n_train = 40, 30
    x_num, x_cat = toy(n)
    m = masks(n, 0.1, seed=4)
    g = to_tensors(make_graph("sage_bip", x_num, x_cat, n_train), n, n_train, CPU)
    pooled, control = encoder("sage_bip"), encoder("sage_bip_nopool")
    assert isinstance(control, BipartiteSAGEEncoder) and not control.pool_records and pooled.pool_records
    assert sum(p.numel() for p in control.parameters()) == sum(p.numel() for p in pooled.parameters())
    control.load_state_dict(pooled.state_dict())  # same weights, so only the message passing differs
    others = x_num.clone()
    others[1:] += 3.0  # every record except record 0 changes
    assert torch.equal(control(x_num, x_cat, *m, g)[0], control(others, x_cat, *m, g)[0])
    assert not torch.allclose(pooled(x_num, x_cat, *m, g)[0], pooled(others, x_cat, *m, g)[0], atol=1e-6)


def test_mean_operator_rows_are_averages() -> None:
    row, col = torch.tensor([0, 0, 2, 2, 2]), torch.tensor([1, 3, 0, 1, 2])
    dense = mean_operator(row, col, 4, 5).to_dense()
    expected = torch.zeros(4, 5)
    expected[0, [1, 3]] = 0.5
    expected[2, [0, 1, 2]] = 1 / 3
    assert torch.allclose(dense, expected)  # rows 1 and 3 have no entries and stay zero
    empty = torch.zeros(0, dtype=torch.long)
    assert mean_operator(empty, empty, 3, 4).to_dense().shape == (3, 4)


def test_bipartite_value_node_bookkeeping() -> None:
    enc = BipartiteSAGEEncoder(N_NUM, CARDS, HIDDEN, DIM)
    assert enc.offsets.tolist() == [0, 3, 5]
    assert enc.value_emb.num_embeddings == sum(CARDS)


def test_a_masked_categorical_cell_removes_its_edge() -> None:
    """Hiding the only cell that connects a record to a value must equal a record without that
    edge: here, two records that differ only in the hidden cell get identical embeddings."""
    n = 30
    x_num, x_cat = toy(n)
    x_num[1], x_cat[1] = x_num[0], x_cat[0]
    x_cat[1, 2] = (x_cat[0, 2] + 1) % CARDS[2]  # records 0 and 1 differ in column 2 only
    m_num = torch.zeros(n, N_NUM, dtype=torch.bool)
    m_cat = torch.zeros(n, len(CARDS), dtype=torch.bool)
    m_cat[:2, 2] = True  # ... and that column is hidden for both
    g = to_tensors(make_graph("sage_bip", x_num, x_cat, n), n, n, CPU)
    z = encoder("sage_bip")(x_num, x_cat, m_num, m_cat, g)
    assert torch.equal(z[0], z[1])


# --------------------------------------------------------------------------- decoder and loss


def test_decoder_output_shapes() -> None:
    dec = Decoder(N_NUM, CARDS, HIDDEN, DIM)
    num, cats = dec(torch.randn(9, DIM))
    assert num.shape == (9, N_NUM) and [c.shape for c in cats] == [(9, c) for c in CARDS]


def test_reconstruction_loss_is_mse_plus_mean_cross_entropy() -> None:
    torch.manual_seed(0)
    n = 12
    x_num, x_cat = toy(n)
    num_pred = torch.randn(n, N_NUM)
    logits = [torch.randn(n, c) for c in CARDS]
    total, mse, ce = reconstruction_loss(num_pred, logits, x_num, x_cat)
    manual_mse = ((num_pred - x_num) ** 2).mean()
    manual_ce = torch.stack([F.cross_entropy(lg, x_cat[:, j]) for j, lg in enumerate(logits)]).mean()
    assert torch.allclose(mse, manual_mse) and torch.allclose(ce, manual_ce)
    assert torch.allclose(total, manual_mse + manual_ce)


def test_mlp_parameter_count_follows_the_architecture() -> None:
    enc = MLPEncoder(N_NUM, CARDS, HIDDEN, DIM)
    d_in = 2 * N_NUM + sum(CARDS)
    assert sum(p.numel() for p in enc.parameters()) == d_in * HIDDEN + HIDDEN + HIDDEN * DIM + DIM


def test_build_encoder_rejects_non_neural_kinds() -> None:
    with pytest.raises(ValueError, match="no neural encoder"):
        build_encoder("pca", N_NUM, CARDS, HIDDEN, DIM)
