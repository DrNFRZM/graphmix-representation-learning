"""Training-loop tests: reproducibility, no label access, no leakage from test rows."""

import inspect

import numpy as np
import pytest
from conftest import make_features

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from graphmix.config import ModelConfig, TrainConfig  # noqa: E402
from graphmix.corruption import Corruptor, Pretext  # noqa: E402
from graphmix.data import LABEL  # noqa: E402
from graphmix.graphs import (  # noqa: E402
    Graph,
    bipartite_graph,
    distance_features,
    graph_from_neighbors,
    knn_neighbors,
    random_graph,
    selfloop_graph,
)
from graphmix.train import (  # noqa: E402
    TrainResult,
    resolve_device,
    seed_everything,
    to_tensors,
    train_autoencoder,
)

KINDS = ["mlp_ae", "sage_selfloop", "sage_rand", "sage_knn", "sage_bip", "sage_bip_nopool"]
N_TRAIN, N_VAL, N_TEST, K = 300, 100, 100, 5
MODEL = ModelConfig(hidden=16, dim=4)
TRAIN = TrainConfig(steps=20, lr=3e-3, eval_every=5, mask_ratio=0.3, device="cpu")


class Problem:
    """Features of a synthetic Adult-like table, ordered [train | val | test]."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, cards: list[int]) -> None:
        self.x_num, self.x_cat, self.cards = x_num, x_cat, cards
        self.dist = distance_features(x_num, x_cat, cards)

    def graph(self, kind: str) -> Graph | None:
        n_total = len(self.x_num)
        if kind == "mlp_ae":
            return None
        if kind == "sage_selfloop":
            return selfloop_graph(N_TRAIN, n_total)
        if kind == "sage_rand":
            return random_graph(N_TRAIN, n_total, K, seed=0)
        if kind == "sage_knn":
            return graph_from_neighbors(knn_neighbors(self.dist, N_TRAIN, K), N_TRAIN)
        return bipartite_graph(self.x_cat, self.cards, N_TRAIN)

    def pretext(
        self, kind: str, mask: float = 0.3, neighbors: str = "visible", eval_ratio: float = 0.3
    ) -> Pretext:
        corruptor = Corruptor(
            self.x_num, self.x_cat, self.cards, self.dist, N_TRAIN, N_VAL, seed=0,
            n_views=2, eval_ratio=eval_ratio, cat_cost=1.0, chunk=128, k_max=K,
        )  # fmt: skip
        table = knn_neighbors(self.dist, N_TRAIN, K) if kind == "sage_knn" else None
        return corruptor.pretext(kind, K, mask, neighbors, self.graph(kind), table)

    def train(
        self,
        kind: str,
        seed: int = 0,
        cfg: TrainConfig = TRAIN,
        mask: float = 0.3,
        neighbors: str = "visible",
    ) -> TrainResult:
        return train_autoencoder(
            kind, self.x_num, self.x_cat, self.cards, self.pretext(kind, mask, neighbors),
            N_TRAIN, N_VAL, MODEL, cfg, seed,
        )  # fmt: skip


@pytest.fixture(scope="module")
def problem() -> Problem:
    return Problem(*make_features(N_TRAIN + N_VAL + N_TEST, N_TRAIN, seed=11))


# --------------------------------------------------------------------------- basics


def test_seed_everything_fixes_the_torch_stream() -> None:
    seed_everything(3)
    a = torch.rand(4)
    seed_everything(3)
    assert torch.equal(a, torch.rand(4))


def test_resolve_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    if not torch.cuda.is_available():
        assert resolve_device("auto") == torch.device("cpu")
        with pytest.raises(RuntimeError, match="no CUDA"):
            resolve_device("cuda")


def test_training_function_cannot_see_labels() -> None:
    params = set(inspect.signature(train_autoencoder).parameters)
    assert not params & {"y", "labels", "label", "target", LABEL}


def test_graph_tensors_check_the_number_of_nodes(problem: Problem) -> None:
    graph = problem.graph("sage_knn")
    assert graph is not None
    with pytest.raises(ValueError, match="nodes"):
        to_tensors(graph, N_TRAIN, N_TRAIN, torch.device("cpu"))
    tensors = to_tensors(graph, graph.n_total, N_TRAIN, torch.device("cpu"))
    assert tensors.adj is not None and tensors.adj.layout == torch.sparse_csr
    assert torch.allclose(tensors.adj.to_dense().sum(dim=1), torch.ones(graph.n_total))  # rows are means
    assert tensors.is_train.sum().item() == N_TRAIN
    assert to_tensors(None, 10, 5, torch.device("cpu")).adj is None
    bip = problem.graph("sage_bip")
    assert bip is not None and to_tensors(bip, bip.n_total, N_TRAIN, torch.device("cpu")).adj is None


# --------------------------------------------------------------------------- each encoder trains


@pytest.mark.parametrize("kind", KINDS)
def test_training_runs_and_reduces_the_validation_loss(problem: Problem, kind: str) -> None:
    res = problem.train(kind)
    n_total = N_TRAIN + N_VAL + N_TEST
    assert res.z.shape == (n_total, MODEL.dim) and np.isfinite(res.z).all()
    assert list(res.log.columns) == ["step", "train_loss", "val_loss"]
    assert res.log["step"].tolist() == [5, 10, 15, 20]
    assert res.log["val_loss"].iloc[-1] < res.log["val_loss"].iloc[0]
    assert res.best_step == res.log.loc[res.log["val_loss"].idxmin(), "step"]
    assert res.n_params > 0 and res.seconds > 0
    assert 0.0 <= res.recon_cat_acc <= 1.0 and res.recon_num_mse > 0
    assert 0.0 <= res.masked_cat_acc <= 1.0 and res.masked_num_mse > 0


def test_encoder_parameter_counts_differ_as_designed(problem: Problem) -> None:
    counts = {k: problem.train(k, cfg=TrainConfig(steps=1, eval_every=1)).n_params for k in KINDS}
    assert counts["sage_selfloop"] == counts["sage_knn"] == counts["sage_rand"]  # same architecture
    assert counts["sage_selfloop"] > counts["mlp_ae"]  # a second (neighbour) weight matrix per layer


def test_masked_scores_are_nan_when_nothing_is_hidden(problem: Problem) -> None:
    pretext = problem.pretext("mlp_ae", mask=0.0, eval_ratio=0.0)
    res = train_autoencoder(
        "mlp_ae", problem.x_num, problem.x_cat, problem.cards, pretext, N_TRAIN, N_VAL, MODEL, TRAIN, 0
    )
    assert np.isnan(res.masked_num_mse) and np.isnan(res.masked_cat_acc)
    assert np.isfinite(res.recon_num_mse) and np.isfinite(res.recon_cat_acc)


# --------------------------------------------------------------------------- reproducibility


@pytest.mark.parametrize("kind", KINDS)
def test_same_seed_gives_identical_results(problem: Problem, kind: str) -> None:
    a, b = problem.train(kind, seed=3), problem.train(kind, seed=3)
    assert np.array_equal(a.z, b.z)
    assert a.log.equals(b.log) and a.best_step == b.best_step
    assert (a.masked_num_mse, a.masked_cat_acc) == (b.masked_num_mse, b.masked_cat_acc)


def test_different_seeds_give_different_embeddings(problem: Problem) -> None:
    assert not np.array_equal(problem.train("mlp_ae", seed=0).z, problem.train("mlp_ae", seed=1).z)


# --------------------------------------------------------------------------- leakage


@pytest.mark.parametrize("kind", KINDS)
def test_test_rows_cannot_influence_training_or_model_selection(problem: Problem, kind: str) -> None:
    """Replace the features of every test row by noise and rebuild the views and graphs. Everything
    that is computed for the training and validation rows (embeddings, loss curves, chosen
    checkpoint) must be unchanged; only the test rows' own embeddings may move."""
    rng = np.random.default_rng(0)
    keep = N_TRAIN + N_VAL
    x_num, x_cat = problem.x_num.copy(), problem.x_cat.copy()
    x_num[keep:] = rng.normal(size=x_num[keep:].shape) * 5
    for j, card in enumerate(problem.cards):
        x_cat[keep:, j] = rng.integers(0, card, size=N_TEST)
    other = Problem(x_num, x_cat, problem.cards)

    a, b = problem.train(kind), other.train(kind)
    assert np.array_equal(a.z[:keep], b.z[:keep])
    assert a.log.equals(b.log) and a.best_step == b.best_step
    assert not np.array_equal(a.z[keep:], b.z[keep:])


def test_validation_rows_do_steer_the_checkpoint_choice_only(problem: Problem) -> None:
    """Complement of the test above: perturbing validation rows leaves the *training-row* part of
    the loss curve untouched (train_loss) but changes val_loss."""
    rng = np.random.default_rng(1)
    x_num = problem.x_num.copy()
    x_num[N_TRAIN : N_TRAIN + N_VAL] += rng.normal(size=(N_VAL, x_num.shape[1])) * 3
    a, b = problem.train("mlp_ae"), Problem(x_num, problem.x_cat, problem.cards).train("mlp_ae")
    assert a.log["train_loss"].equals(b.log["train_loss"])
    assert not a.log["val_loss"].equals(b.log["val_loss"])


def test_no_corruption_mode_runs(problem: Problem) -> None:
    res = problem.train("sage_knn", mask=0.0)
    assert np.isfinite(res.z).all() and np.isfinite(res.masked_num_mse)  # test rows are still hidden


# --------------------------------------------------------------------------- the neighbour shortcut


def test_a_graph_built_from_clean_rows_gives_the_encoder_the_hidden_cells(problem: Problem) -> None:
    """The reason for corrupted-view graphs. With neighbours picked from the clean rows, the
    encoder recovers hidden numeric cells far better than when the graph is built from visible
    cells only, where it is no better than the flat encoder (on this random table, whose columns
    are nearly independent, there is nothing else to gain)."""
    cfg = TrainConfig(steps=150, lr=3e-3, eval_every=50, mask_ratio=0.3, device="cpu")
    leaky = problem.train("sage_knn", cfg=cfg, neighbors="clean")
    honest = problem.train("sage_knn", cfg=cfg, neighbors="visible")
    flat = problem.train("mlp_ae", cfg=cfg)
    assert leaky.masked_num_mse < 0.5 * honest.masked_num_mse
    assert honest.masked_num_mse > 0.8 * flat.masked_num_mse
