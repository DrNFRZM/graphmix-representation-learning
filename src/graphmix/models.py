"""Encoders and the shared decoder.

All encoders share one call signature and map (masked) mixed-type rows to a `dim`-vector.
They differ only in what they may look at besides the row itself:
  MLPEncoder            nothing (flat baseline)
  SAGEEncoder           incoming edges of a homogeneous graph (kNN / random / self-loop)
  BipartiteSAGEEncoder  the record-value graph: value nodes carry learned embeddings and
                        pool the numeric features of the training records that share them
                        (pool_records=False removes that pooling: the control for the graph part)

Every GraphSAGE layer uses the mean aggregator. It is written as a sum over a row-normalised
sparse matrix (PyG's aggr="sum" on a CSR tensor) rather than aggr="mean" on an edge list: the
result is the same (tests/test_models.py compares the two), but PyG then runs one fused sparse
product instead of gathering a (num_edges x features) tensor, which was about 4x faster per
training step on two CPU cores.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

with warnings.catch_warnings():
    # PyG calls torch.jit.script while it is imported, which recent torch releases report as
    # deprecated. That is PyG's business, so keep it out of our console output.
    warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=FutureWarning)
    from torch_geometric.nn import SAGEConv

# The CSR adjacency below is what makes the GraphSAGE layers fast; torch still calls it beta.
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*", category=UserWarning)


@dataclass
class GraphTensors:
    adj: Tensor | None  # sparse CSR (n, n) mean operator of a homogeneous graph, rows = receivers
    is_train: Tensor  # (n,) bool, True for training nodes (the only senders towards value nodes)


def mean_operator(row: Tensor, col: Tensor, n_rows: int, n_cols: int) -> Tensor:
    """Sparse CSR matrix with entry (row, col) = 1 / (number of entries in that row); empty rows
    stay zero. Entries must be distinct or they are summed, which matches a mean over an edge
    list with repeated edges."""
    count = torch.bincount(row, minlength=n_rows).clamp(min=1)
    weight = (1.0 / count[row]).to(torch.float32)
    coo = torch.sparse_coo_tensor(torch.stack([row, col]), weight, (n_rows, n_cols), check_invariants=False)
    return coo.coalesce().to_sparse_csr()


def flat_input(x_num: Tensor, x_cat: Tensor, m_num: Tensor, m_cat: Tensor, cards: Sequence[int]) -> Tensor:
    """[numeric (masked -> 0), numeric mask flags, one-hot blocks (masked -> all zeros)]."""
    parts = [x_num.masked_fill(m_num, 0.0), m_num.to(x_num.dtype)]
    for j, card in enumerate(cards):
        onehot = F.one_hot(x_cat[:, j], num_classes=card).to(x_num.dtype)
        parts.append(onehot.masked_fill(m_cat[:, j : j + 1], 0.0))
    return torch.cat(parts, dim=1)


class MLPEncoder(nn.Module):
    def __init__(self, n_num: int, cards: Sequence[int], hidden: int, dim: int) -> None:
        super().__init__()
        self.cards = list(cards)
        self.net = nn.Sequential(nn.Linear(2 * n_num + sum(cards), hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, x_num: Tensor, x_cat: Tensor, m_num: Tensor, m_cat: Tensor, g: GraphTensors) -> Tensor:
        return self.net(flat_input(x_num, x_cat, m_num, m_cat, self.cards))


class SAGEEncoder(nn.Module):
    """Two GraphSAGE layers (mean aggregator, root weight) on a homogeneous graph:
    out_i = W_neigh * mean_{j -> i} h_j + W_root * h_i."""

    def __init__(self, n_num: int, cards: Sequence[int], hidden: int, dim: int) -> None:
        super().__init__()
        self.cards = list(cards)
        self.conv1 = SAGEConv(2 * n_num + sum(cards), hidden, aggr="sum")
        self.conv2 = SAGEConv(hidden, dim, aggr="sum")

    def forward(self, x_num: Tensor, x_cat: Tensor, m_num: Tensor, m_cat: Tensor, g: GraphTensors) -> Tensor:
        assert g.adj is not None
        x = flat_input(x_num, x_cat, m_num, m_cat, self.cards)
        h = F.relu(self.conv1(x, g.adj))
        return self.conv2(h, g.adj)


class BipartiteSAGEEncoder(nn.Module):
    """Record nodes carry the numeric columns (+ mask flags); every categorical cell is an edge
    to the value node of that (column, level). A masked cell is simply a missing edge.

      layer 1  record <- mean of its value embeddings,   value <- mean of its records' numerics
      layer 2  record <- mean of its values' layer-1 states

    With pool_records=False the value nodes receive nothing from the records (that edge set is
    empty). Weights, value embeddings and layers are the same, but a record can no longer learn
    anything from other records; it is the control for the graph part of this encoder, as the
    self-loop graph is for the kNN encoder.
    """

    offsets: Tensor  # first value-node id of every categorical column (a buffer)

    def __init__(
        self, n_num: int, cards: Sequence[int], hidden: int, dim: int, pool_records: bool = True
    ) -> None:
        super().__init__()
        self.pool_records = pool_records
        offsets, total = [], 0
        for card in cards:
            offsets.append(total)
            total += card
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.value_emb = nn.Embedding(total, hidden)
        rec_dim = 2 * n_num
        self.rec_from_val = SAGEConv((hidden, rec_dim), hidden, aggr="sum")  # value -> record
        self.val_from_rec = SAGEConv((rec_dim, hidden), hidden, aggr="sum")  # record -> value
        self.out = SAGEConv((hidden, hidden), dim, aggr="sum")  # value -> record

    def forward(self, x_num: Tensor, x_cat: Tensor, m_num: Tensor, m_cat: Tensor, g: GraphTensors) -> Tensor:
        n, n_values = x_num.shape[0], self.value_emb.num_embeddings
        x_rec = torch.cat([x_num.masked_fill(m_num, 0.0), m_num.to(x_num.dtype)], dim=1)
        rec, col = (~m_cat).nonzero(as_tuple=True)  # visible categorical cells
        val = x_cat[rec, col] + self.offsets[col]  # their value nodes
        to_rec = mean_operator(rec, val, n, n_values)  # every record averages over its values ...
        if self.pool_records:
            sender = g.is_train[rec]  # ... but only training records send to value nodes,
            to_val = mean_operator(val[sender], rec[sender], n_values, n)  # so evaluation rows never leak in
        else:
            to_val = mean_operator(rec[:0], rec[:0], n_values, n)  # control: no record sends to a value node
        emb = self.value_emb.weight
        h_rec = F.relu(self.rec_from_val((emb, x_rec), to_rec))
        h_val = F.relu(self.val_from_rec((x_rec, emb), to_val))
        return self.out((h_val, h_rec), to_rec)


class Decoder(nn.Module):
    """Shared by every method: dim -> hidden -> hidden -> (numeric values, categorical logits)."""

    def __init__(self, n_num: int, cards: Sequence[int], hidden: int, dim: int) -> None:
        super().__init__()
        self.cards = list(cards)
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.num_head = nn.Linear(hidden, n_num)
        self.cat_head = nn.Linear(hidden, sum(cards))

    def forward(self, z: Tensor) -> tuple[Tensor, list[Tensor]]:
        h = self.body(z)
        return self.num_head(h), list(self.cat_head(h).split(self.cards, dim=1))


def build_encoder(kind: str, n_num: int, cards: Sequence[int], hidden: int, dim: int) -> nn.Module:
    if kind == "mlp_ae":
        return MLPEncoder(n_num, cards, hidden, dim)
    if kind in ("sage_bip", "sage_bip_nopool"):
        return BipartiteSAGEEncoder(n_num, cards, hidden, dim, pool_records=kind == "sage_bip")
    if kind in ("sage_knn", "sage_rand", "sage_selfloop"):
        return SAGEEncoder(n_num, cards, hidden, dim)
    raise ValueError(f"no neural encoder for kind '{kind}'")


def reconstruction_loss(
    num_pred: Tensor, cat_logits: Sequence[Tensor], x_num: Tensor, x_cat: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """MSE on the numeric columns + cross-entropy on the categorical columns (each averaged
    over its columns, then added). Computed against the clean row, on visible and masked cells."""
    mse = F.mse_loss(num_pred, x_num)
    ce = torch.stack([F.cross_entropy(logits, x_cat[:, j]) for j, logits in enumerate(cat_logits)]).mean()
    return mse + ce, mse, ce
