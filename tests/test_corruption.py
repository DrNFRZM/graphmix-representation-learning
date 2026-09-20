"""The corrupted views: masks, fill-in, and the property that motivates the whole module (the
graph of a view is a function of the visible cells only)."""

import numpy as np
import pytest
from conftest import make_features

from graphmix.corruption import Corruptor, fill_hidden, sample_masks
from graphmix.graphs import distance_features, graph_from_neighbors, knn_neighbors

N_TRAIN, N_VAL, N_TEST, K = 300, 100, 100, 5


@pytest.fixture(scope="module")
def features() -> tuple[np.ndarray, np.ndarray, list[int]]:
    return make_features(N_TRAIN + N_VAL + N_TEST, N_TRAIN, seed=5)


def corruptor(
    x_num: np.ndarray, x_cat: np.ndarray, cards: list[int], seed: int = 0, eval_ratio: float = 0.3
) -> Corruptor:
    dist = distance_features(x_num, x_cat, cards)
    return Corruptor(
        x_num, x_cat, cards, dist, N_TRAIN, N_VAL, seed,
        n_views=3, eval_ratio=eval_ratio, cat_cost=1.0, chunk=128, k_max=K,
    )  # fmt: skip


def scramble(x_num: np.ndarray, x_cat: np.ndarray, cards: list[int], m_num: np.ndarray, m_cat: np.ndarray):
    """Overwrite the cells flagged in the masks (which cover the leading rows) with other values."""
    n = len(m_num)
    a, b = x_num.copy(), x_cat.copy()
    a[:n] = np.where(m_num, a[:n] + 7.0, a[:n])
    b[:n] = np.where(m_cat, (b[:n] + 1) % np.array(cards), b[:n])
    return a, b


# --------------------------------------------------------------------------- masks and fill-in


def test_masks_have_the_requested_rate_and_are_reproducible() -> None:
    a = sample_masks(4000, 5, 7, 0.3, np.random.default_rng(0))
    b = sample_masks(4000, 5, 7, 0.3, np.random.default_rng(0))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].shape == (4000, 5) and a[1].shape == (4000, 7)
    assert a[0].mean() == pytest.approx(0.3, abs=0.02) and a[1].mean() == pytest.approx(0.3, abs=0.02)
    none = sample_masks(100, 5, 7, 0.0, np.random.default_rng(0))
    assert not none[0].any() and not none[1].any()


def test_fill_keeps_visible_cells_and_draws_hidden_ones_from_visible_values(features) -> None:
    x_num, x_cat, _ = features
    m_num, m_cat = sample_masks(len(x_num), x_num.shape[1], x_cat.shape[1], 0.3, np.random.default_rng(0))
    f_num, f_cat = fill_hidden(x_num, x_cat, m_num, m_cat, np.random.default_rng(1))
    assert np.array_equal(f_num[~m_num], x_num[~m_num])
    assert np.array_equal(f_cat[~m_cat], x_cat[~m_cat])
    assert (f_num[m_num] != x_num[m_num]).any()  # something was really replaced
    for j in range(x_num.shape[1]):
        assert np.isin(f_num[m_num[:, j], j], x_num[~m_num[:, j], j]).all()
    for j in range(x_cat.shape[1]):
        assert np.isin(f_cat[m_cat[:, j], j], x_cat[~m_cat[:, j], j]).all()


def test_filled_rows_do_not_depend_on_the_hidden_values(features) -> None:
    x_num, x_cat, cards = features
    m_num, m_cat = sample_masks(len(x_num), x_num.shape[1], x_cat.shape[1], 0.3, np.random.default_rng(0))
    x_num2, x_cat2 = scramble(x_num, x_cat, cards, m_num, m_cat)
    assert not np.array_equal(x_num, x_num2) and not np.array_equal(x_cat, x_cat2)
    a = fill_hidden(x_num, x_cat, m_num, m_cat, np.random.default_rng(1))
    b = fill_hidden(x_num2, x_cat2, m_num, m_cat, np.random.default_rng(1))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_fill_can_draw_from_a_separate_pool(features) -> None:
    x_num, x_cat, _ = features
    rows = slice(N_TRAIN, N_TRAIN + 50)
    m_num, m_cat = sample_masks(50, x_num.shape[1], x_cat.shape[1], 0.5, np.random.default_rng(0))
    pool = (x_num[:N_TRAIN], x_cat[:N_TRAIN])
    f_num, f_cat = fill_hidden(x_num[rows], x_cat[rows], m_num, m_cat, np.random.default_rng(1), pool)
    for j in range(x_num.shape[1]):
        assert np.isin(f_num[m_num[:, j], j], pool[0][:, j]).all()
    for j in range(x_cat.shape[1]):
        assert np.isin(f_cat[m_cat[:, j], j], pool[1][:, j]).all()


def test_fill_needs_at_least_one_visible_value_per_column() -> None:
    x_num, x_cat = np.zeros((4, 1)), np.zeros((4, 1), dtype=np.int64)
    everything = np.ones((4, 1), dtype=bool)
    with pytest.raises(ValueError, match="no visible value"):
        fill_hidden(x_num, x_cat, everything, ~everything, np.random.default_rng(0))


# --------------------------------------------------------------------------- views


def test_training_views_are_fixed_distinct_and_seeded(features) -> None:
    c = corruptor(*features)
    views = c.train_masks(0.3)
    assert len(views) == 3 and c.train_masks(0.3) is views  # cached
    assert views[0][0].shape == (N_TRAIN, 5) and views[0][1].shape == (N_TRAIN, 7)
    assert np.mean([m.mean() for m, _ in views]) == pytest.approx(0.3, abs=0.03)
    assert not np.array_equal(views[0][0], views[1][0])
    again = corruptor(*features).train_masks(0.3)
    assert all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(views, again, strict=True)
    )
    other = corruptor(*features, seed=1).train_masks(0.3)
    assert not np.array_equal(views[0][0], other[0][0])


def test_without_corruption_there_is_one_view_and_nothing_is_hidden(features) -> None:
    views = corruptor(*features).train_masks(0.0)
    assert len(views) == 1 and not views[0][0].any() and not views[0][1].any()


def test_eval_masks_hide_cells_of_their_own_split_only(features) -> None:
    c = corruptor(*features)
    val_num, val_cat = c.eval_masks("val", 0.3)
    test_num, test_cat = c.eval_masks("test", 0.3)
    assert val_num.shape == (N_TRAIN + N_VAL + N_TEST, 5)
    val_rows, test_rows = slice(N_TRAIN, N_TRAIN + N_VAL), slice(N_TRAIN + N_VAL, None)
    assert val_num[val_rows].any() and val_cat[val_rows].any()
    assert not val_num[:N_TRAIN].any() and not val_num[test_rows].any() and not val_cat[test_rows].any()
    assert test_num[test_rows].any() and not test_num[: N_TRAIN + N_VAL].any()
    assert not test_cat[: N_TRAIN + N_VAL].any()


# --------------------------------------------------------------------------- graphs of the views


def test_view_graphs_depend_on_the_visible_cells_only(features) -> None:
    """Changing exactly the cells hidden in view 0 must leave view 0's neighbour table untouched
    (it is what keeps the pretext task honest), while the other views, in which some of those
    cells are visible, do change. The clean kNN graph changes as well: that is the shortcut."""
    x_num, x_cat, cards = features
    c = corruptor(x_num, x_cat, cards)
    tables = c.train_tables(0.3)
    m_num, m_cat = c.train_masks(0.3)[0]
    x_num2, x_cat2 = scramble(x_num, x_cat, cards, m_num, m_cat)
    tables2 = corruptor(x_num2, x_cat2, cards).train_tables(0.3)
    assert tables[0].shape == (N_TRAIN, K)
    assert np.array_equal(tables[0], tables2[0])
    assert not np.array_equal(tables[1], tables2[1])

    clean = knn_neighbors(distance_features(x_num, x_cat, cards), N_TRAIN, K)
    clean2 = knn_neighbors(distance_features(x_num2, x_cat2, cards), N_TRAIN, K)
    assert not np.array_equal(clean, clean2)


def test_neighbours_of_corrupted_validation_rows_come_from_visible_cells(features) -> None:
    x_num, x_cat, cards = features
    c = corruptor(x_num, x_cat, cards)
    table = c.eval_table("val", 0.3)
    assert table.shape == (N_VAL, K) and table.max() < N_TRAIN  # neighbours are training rows
    m_num, m_cat = c.eval_masks("val", 0.3)
    hidden_num, hidden_cat = m_num[N_TRAIN : N_TRAIN + N_VAL], m_cat[N_TRAIN : N_TRAIN + N_VAL]
    # scramble the hidden cells of the validation rows (they start at row N_TRAIN)
    x_num2, x_cat2 = x_num.copy(), x_cat.copy()
    rows = slice(N_TRAIN, N_TRAIN + N_VAL)
    x_num2[rows] = np.where(hidden_num, x_num[rows] + 7.0, x_num[rows])
    x_cat2[rows] = np.where(hidden_cat, (x_cat[rows] + 1) % np.array(cards), x_cat[rows])
    assert np.array_equal(table, corruptor(x_num2, x_cat2, cards).eval_table("val", 0.3))
    # ... but scrambling visible cells does move them
    x_num3 = x_num.copy()
    x_num3[rows] = np.where(~hidden_num, x_num[rows] + 7.0, x_num[rows])
    assert not np.array_equal(table, corruptor(x_num3, x_cat, cards).eval_table("val", 0.3))


def clean_knn(features):
    x_num, x_cat, cards = features
    table = knn_neighbors(distance_features(x_num, x_cat, cards), N_TRAIN, K)
    return table, graph_from_neighbors(table, N_TRAIN)


def test_pretext_of_a_knn_encoder(features) -> None:
    c = corruptor(*features)
    table, full = clean_knn(features)
    pt = c.pretext("sage_knn", K, 0.3, "visible", full, table)
    assert len(pt.train) == 3 and pt.full is full
    for view, tab in zip(pt.train, c.train_tables(0.3), strict=True):
        assert view.graph is not None and view.graph.n_total == N_TRAIN
        assert np.array_equal(view.graph.edge_index, graph_from_neighbors(tab, N_TRAIN).edge_index)

    n_edges_train, n_edges_val = N_TRAIN * K, N_VAL * K
    assert pt.val.graph is not None and pt.val.graph.edge_index is not None and full.edge_index is not None
    edges = pt.val.graph.edge_index
    assert edges[0].max() < N_TRAIN  # evaluation rows are never senders
    assert np.array_equal(edges[:, :n_edges_train], full.edge_index[:, :n_edges_train])  # train rows: clean
    val_senders = edges[0, n_edges_train : n_edges_train + n_edges_val].reshape(N_VAL, K)
    assert np.array_equal(val_senders, c.eval_table("val", 0.3))  # validation rows: from visible cells

    assert pt.test.graph is not None and pt.test.graph.edge_index is not None
    test_edges = pt.test.graph.edge_index
    test_senders = test_edges[0, -N_TEST * K :].reshape(N_TEST, K)
    assert np.array_equal(test_senders, c.eval_table("test", 0.3))
    assert pt.val.m_num[:N_TRAIN].sum() == 0 and pt.test.m_num[: N_TRAIN + N_VAL].sum() == 0


def test_the_clean_control_keeps_the_leaky_graph(features) -> None:
    c = corruptor(*features)
    table, full = clean_knn(features)
    pt = c.pretext("sage_knn", K, 0.3, "clean", full, table)
    restricted = full.restrict_to_train()
    assert all(np.array_equal(v.graph.edge_index, restricted.edge_index) for v in pt.train)
    assert pt.val.graph is full and pt.test.graph is full
    assert c.seconds == 0.0  # no search on corrupted rows was needed


def test_without_corruption_the_training_graph_is_the_clean_one(features) -> None:
    c = corruptor(*features)
    table, full = clean_knn(features)
    pt = c.pretext("sage_knn", K, 0.0, "visible", full, table)
    assert len(pt.train) == 1 and not pt.train[0].m_num.any() and not pt.train[0].m_cat.any()
    assert np.array_equal(pt.train[0].graph.edge_index, full.restrict_to_train().edge_index)
    assert pt.val.graph is full
    assert pt.test.m_num.any()  # test rows are still corrupted, so that all methods are scored alike


@pytest.mark.parametrize("has_graph", [False, True])
def test_other_encoders_share_one_static_graph(features, has_graph: bool) -> None:
    c = corruptor(*features)
    _, full = clean_knn(features)  # any graph will do for this check
    pt = c.pretext("sage_rand", K, 0.3, "visible", full if has_graph else None, None)
    graphs = [v.graph for v in pt.train]
    assert (graphs[0] is None) == (not has_graph)
    assert len({id(g) for g in graphs}) == 1  # the same object for every view
    assert pt.val.graph is (full if has_graph else None) and pt.test.graph is pt.val.graph
    assert c.seconds == 0.0


def test_corrupted_searches_are_cached_and_timed(features) -> None:
    c = corruptor(*features)
    first = c.train_tables(0.3)
    spent = c.seconds
    assert spent > 0 and c.train_tables(0.3) is first and c.seconds == spent
