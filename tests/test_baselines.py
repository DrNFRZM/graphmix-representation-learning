import numpy as np
import pytest

from graphmix.baselines import PROP_HOPS, pca_embedding, propagated_embedding
from graphmix.config import GRAPH_OF_KIND, PROP_KINDS
from graphmix.graphs import bipartite_graph, graph_from_neighbors, knn_neighbors, random_graph


def test_hops_are_defined_for_every_propagation_graph() -> None:
    assert set(PROP_HOPS) == {GRAPH_OF_KIND[k] for k in PROP_KINDS}


def test_pca_is_fitted_on_training_rows_only() -> None:
    rng = np.random.default_rng(0)
    feat = rng.normal(size=(120, 10)).astype(np.float32)
    a = pca_embedding(feat, n_train=80, dim=4, seed=0)
    shifted = feat.copy()
    shifted[80:] = rng.normal(size=(40, 10)) * 50  # wild evaluation rows
    b = pca_embedding(shifted, n_train=80, dim=4, seed=0)
    assert a.shape == (120, 4) and a.dtype == np.float32
    assert np.allclose(a[:80], b[:80])  # so training rows are embedded exactly as before
    assert not np.allclose(a[80:], b[80:])


def test_pca_components_come_out_in_order_of_variance() -> None:
    rng = np.random.default_rng(1)
    feat = rng.normal(size=(200, 6)) * np.array([5, 4, 3, 2, 1, 0.5])
    z = pca_embedding(feat, n_train=150, dim=3, seed=0)
    variances = z[:150].var(axis=0)
    assert np.all(np.diff(variances) < 0)
    assert np.allclose(np.corrcoef(z[:150].T) - np.eye(3), 0, atol=1e-6)  # uncorrelated


def test_pca_dimension_is_clipped_to_what_exists() -> None:
    feat = np.random.default_rng(2).normal(size=(50, 3))
    assert pca_embedding(feat, n_train=40, dim=16, seed=0).shape == (50, 3)


@pytest.mark.parametrize("kind", ["knn", "random", "bipartite"])
def test_propagated_embedding_does_not_leak_evaluation_rows(kind: str) -> None:
    rng = np.random.default_rng(3)
    n_train, n_total = 60, 90
    x_cat = np.stack([rng.integers(0, 3, n_total), rng.integers(0, 4, n_total)], axis=1)
    flat = np.concatenate([rng.normal(size=(n_total, 3)), np.eye(3)[x_cat[:, 0]]], axis=1)
    if kind == "knn":
        graph = graph_from_neighbors(knn_neighbors(flat, n_train, 5), n_train)
    elif kind == "random":
        graph = random_graph(n_train, n_total, 5, seed=0)
    else:
        graph = bipartite_graph(x_cat, [3, 4], n_train)
    flat2 = flat.copy()
    flat2[n_train:] += 10.0
    a = propagated_embedding(graph, flat, dim=4, seed=0)
    b = propagated_embedding(graph, flat2, dim=4, seed=0)
    assert a.shape == (n_total, 4)
    assert np.allclose(a[:n_train], b[:n_train], atol=1e-5)
