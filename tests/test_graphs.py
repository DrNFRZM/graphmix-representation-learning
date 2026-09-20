import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from graphmix.graphs import (
    Graph,
    adjusted_homophily,
    bipartite_graph,
    column_label_agreement,
    distance_features,
    edge_homophily,
    graph_from_neighbors,
    knn_indices,
    knn_neighbors,
    label_agreement_by_chance,
    make_propagator,
    propagate,
    random_graph,
    selfloop_graph,
    sender_degree_stats,
)


def brute_force(query: np.ndarray, ref: np.ndarray, k: int, exclude_self: bool = False) -> np.ndarray:
    """Slow reference: sort by (squared distance, index)."""
    out = []
    for i, q in enumerate(query):
        d = ((ref - q) ** 2).sum(axis=1)
        if exclude_self:
            d[i] = np.inf
        out.append(np.lexsort((np.arange(len(ref)), d))[:k])
    return np.array(out)


# --------------------------------------------------------------------------- kNN search


def test_knn_matches_brute_force_including_ties() -> None:
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, size=(80, 3)).astype(np.float64)  # tiny integer grid -> many exact ties
    assert np.array_equal(knn_indices(x, x, 7, exclude_self=True), brute_force(x, x, 7, exclude_self=True))
    q = rng.integers(0, 3, size=(30, 3)).astype(np.float64)
    assert np.array_equal(knn_indices(q, x, 5), brute_force(q, x, 5))


def test_knn_result_does_not_depend_on_chunk_size() -> None:
    x = np.random.default_rng(1).integers(0, 4, size=(90, 4)).astype(np.float64)
    reference = knn_indices(x, x, 6, exclude_self=True, chunk=1024)
    for chunk in (1, 7, 89):
        assert np.array_equal(knn_indices(x, x, 6, exclude_self=True, chunk=chunk), reference)


def test_knn_agrees_with_sklearn_on_continuous_data() -> None:
    rng = np.random.default_rng(2)
    ref, query = rng.normal(size=(200, 6)), rng.normal(size=(40, 6))
    expected = NearestNeighbors(n_neighbors=9, algorithm="brute").fit(ref).kneighbors(query)[1]
    assert np.array_equal(knn_indices(query, ref, 9), expected)


def test_knn_never_returns_the_query_itself_when_excluded() -> None:
    x = np.zeros((20, 2))  # all rows identical: only the exclusion keeps a row from finding itself
    nb = knn_indices(x, x, 5, exclude_self=True)
    assert all(i not in row for i, row in enumerate(nb))


def test_knn_argument_checks() -> None:
    x = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError, match="not possible"):
        knn_indices(x, x, 10, exclude_self=True)
    with pytest.raises(ValueError, match="not possible"):
        knn_indices(x, x, 0)
    with pytest.raises(ValueError, match="same rows"):
        knn_indices(x[:5], x, 2, exclude_self=True)
    with pytest.raises(ValueError, match="too large"):
        knn_indices(x * 1e6, x * 1e6, 2)


def test_distance_features_give_a_gower_style_distance() -> None:
    x_num = np.array([[0.0, 1.0], [1.0, 1.0]])
    x_cat = np.array([[0, 1, 2], [1, 1, 0]])  # differ in columns 0 and 2 only
    for cost in (1.0, 2.5):
        f = distance_features(x_num, x_cat, cards=[2, 2, 3], cat_cost=cost)
        assert np.sum((f[0] - f[1]) ** 2) == pytest.approx(1.0 + 2 * cost)


def test_knn_neighbors_layout_and_train_pool_restriction() -> None:
    rng = np.random.default_rng(3)
    feat = rng.normal(size=(65, 4))
    n_train = 40
    nb = knn_neighbors(feat, n_train, 6)
    assert nb.shape == (65, 6)
    assert nb.max() < n_train  # everybody, evaluation rows included, points into the train pool
    assert all(i not in row for i, row in enumerate(nb[:n_train]))
    assert np.array_equal(nb[n_train:], knn_indices(feat[n_train:], feat[:n_train], 6))
    # GraphFactory relies on this: a smaller k is a prefix of a larger k
    assert np.array_equal(knn_neighbors(feat, n_train, 10)[:, :6], nb)


# --------------------------------------------------------------------------- graph objects


def test_edges_point_from_neighbour_to_the_node_that_asked() -> None:
    g = graph_from_neighbors(np.array([[1, 2], [0, 2], [0, 1]]), n_train=3)
    assert g.edge_index is not None and g.edge_index.shape == (2, 6)
    src, dst = g.edge_index
    assert sorted(src[dst == 0]) == [1, 2] and sorted(src[dst == 2]) == [0, 1]


def test_restricted_graph_only_contains_training_nodes() -> None:
    feat = np.random.default_rng(4).normal(size=(60, 3))
    g = graph_from_neighbors(knn_neighbors(feat, 40, 5), n_train=40)
    r = g.restrict_to_train()
    assert r.edge_index is not None and g.edge_index is not None
    assert r.n_total == 40 and r.edge_index.max() < 40 and r.edge_index.shape[1] == 40 * 5
    assert np.array_equal(r.edge_index, g.edge_index[:, g.edge_index[1] < 40])


def test_restrict_to_train_refuses_a_leaking_graph() -> None:
    leaky = Graph("knn", n_train=2, n_total=4, edge_index=np.array([[3], [0]]))  # eval node 3 -> train node 0
    with pytest.raises(AssertionError, match="evaluation node"):
        leaky.restrict_to_train()


def test_random_graph_is_degree_matched_and_uses_only_the_train_pool() -> None:
    g = random_graph(n_train=50, n_total=80, k=5, seed=3)
    assert g.edge_index is not None
    src, dst = g.edge_index
    assert src.shape == (80 * 5,) and src.max() < 50
    assert np.array_equal(np.bincount(dst, minlength=80), np.full(80, 5))  # in-degree k everywhere
    assert (src[dst < 50] != dst[dst < 50]).all()  # no self loops among training nodes
    again = random_graph(50, 80, 5, seed=3).edge_index
    other = random_graph(50, 80, 5, seed=4).edge_index
    assert again is not None and other is not None
    assert np.array_equal(g.edge_index, again) and not np.array_equal(g.edge_index, other)


def test_selfloop_graph() -> None:
    g = selfloop_graph(3, 5)
    assert g.edge_index is not None
    assert np.array_equal(g.edge_index, np.array([np.arange(5), np.arange(5)]))


def test_bipartite_ids_are_unique_per_column_and_level() -> None:
    x_cat = np.array([[0, 1, 3], [2, 0, 3], [1, 1, 0]])
    g = bipartite_graph(x_cat, cards=[3, 2, 4], n_train=2)
    assert g.val_ids is not None and g.n_values == 9
    assert np.array_equal(g.val_ids, x_cat + np.array([0, 3, 5]))
    assert len(np.unique(g.val_ids)) == len({(c, v) for row in x_cat for c, v in enumerate(row)})
    r = g.restrict_to_train()
    assert r.val_ids is not None and r.val_ids.shape == (2, 3) and r.n_total == 2


# --------------------------------------------------------------------------- propagation


def test_propagator_is_the_neighbour_mean() -> None:
    rng = np.random.default_rng(5)
    feat, x = rng.normal(size=(60, 3)), rng.normal(size=(60, 4))
    nb = knn_neighbors(feat, 40, 5)
    step = make_propagator(graph_from_neighbors(nb, n_train=40))
    expected = np.stack([x[nb[i]].mean(axis=0) for i in range(60)])
    assert np.allclose(step(x), expected)
    assert np.allclose(step(np.ones((60, 1))), 1.0)  # row-stochastic


def test_selfloop_propagation_repeats_the_features() -> None:
    x = np.random.default_rng(6).normal(size=(7, 3))
    out = propagate(selfloop_graph(4, 7), x, hops=2)
    assert out.shape == (7, 9) and np.allclose(out, np.tile(x, 3), atol=1e-6)


@pytest.mark.parametrize("kind", ["knn", "random"])
def test_evaluation_rows_never_influence_anyone_else(kind: str) -> None:
    rng = np.random.default_rng(7)
    n_train, n_total, width = 40, 65, 3
    feat, x = rng.normal(size=(n_total, 4)), rng.normal(size=(n_total, width))
    if kind == "knn":
        g = graph_from_neighbors(knn_neighbors(feat, n_train, 5), n_train)
    else:
        g = random_graph(n_train, n_total, 5, seed=1)
    x2 = x.copy()
    x2[n_train:] += rng.normal(size=(n_total - n_train, width)) * 10  # corrupt every evaluation row
    a, b = propagate(g, x, 2), propagate(g, x2, 2)
    assert np.array_equal(a[:n_train], b[:n_train])  # training rows: identical in every block
    assert np.array_equal(a[:, width:], b[:, width:])  # hops >= 1 of all rows: identical
    assert not np.array_equal(a[n_train:, :width], b[n_train:, :width])  # only the raw copy changed


def test_bipartite_propagation_matches_the_definition() -> None:
    rng = np.random.default_rng(8)
    n, n_train = 30, 20
    x_cat = np.stack([rng.integers(0, 3, n), rng.integers(0, 4, n)], axis=1)
    x = rng.normal(size=(n, 2))
    g = bipartite_graph(x_cat, [3, 4], n_train)
    out = make_propagator(g)(x)
    expected = np.zeros_like(x)
    for i in range(n):
        per_column = []
        for c in range(2):
            group = [j for j in range(n_train) if x_cat[j, c] == x_cat[i, c]]
            # mean over the *training* records sharing the value (an empty group contributes zeros)
            per_column.append(x[group].mean(axis=0) if group else np.zeros(2))
        expected[i] = np.mean(per_column, axis=0)
    assert np.allclose(out, expected)


def test_bipartite_evaluation_rows_do_not_send() -> None:
    rng = np.random.default_rng(9)
    n, n_train = 30, 20
    x_cat = np.stack([rng.integers(0, 3, n), rng.integers(0, 4, n)], axis=1)
    x = rng.normal(size=(n, 2))
    x2 = x.copy()
    x2[n_train:] += 100.0
    g = bipartite_graph(x_cat, [3, 4], n_train)
    a, b = propagate(g, x, 1), propagate(g, x2, 1)
    assert np.array_equal(a[:, 2:], b[:, 2:])


# --------------------------------------------------------------------------- diagnostics


def test_homophily_on_a_hand_example() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    edges = np.array([[0, 1, 2, 0, 0], [1, 0, 3, 2, 5]])  # last edge goes into an evaluation node
    assert edge_homophily(edges, y, n_train=4) == pytest.approx(3 / 4)
    chance = (5 / 8) ** 2 + (3 / 8) ** 2  # endpoint label shares of the four training edges
    assert adjusted_homophily(edges, y, n_train=4) == pytest.approx((0.75 - chance) / (1 - chance))


def test_label_agreement_by_chance() -> None:
    assert label_agreement_by_chance(np.array([1] * 25 + [0] * 75)) == pytest.approx(0.625)


def test_column_label_agreement_matches_a_loop() -> None:
    rng = np.random.default_rng(10)
    n, n_train = 120, 90
    x_cat = np.stack([rng.integers(0, 3, n), rng.integers(0, 5, n)], axis=1)
    y = (rng.random(n) < 0.3).astype(np.int64)
    g = bipartite_graph(x_cat, [3, 5], n_train)
    assert g.val_ids is not None
    got = column_label_agreement(g.val_ids, y, n_train)
    for c in range(2):
        fractions = []
        for i in range(n_train):
            others = [j for j in range(n_train) if j != i and x_cat[j, c] == x_cat[i, c]]
            if others:
                fractions.append(np.mean([y[j] == y[i] for j in others]))
        assert got[c] == pytest.approx(np.mean(fractions))


def test_sender_degree_stats() -> None:
    edges = np.array([[0, 0, 0, 1], [1, 2, 3, 0]])  # node 0 sends three times, node 1 once
    stats = sender_degree_stats(edges, n_train=4)
    assert stats["sender_deg_max"] == 3.0
    assert stats["frac_never_sender"] == pytest.approx(0.5)
    assert stats["sender_deg_skew"] > 0
