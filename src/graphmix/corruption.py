"""Corrupted inputs for the denoising pretext task, and the graphs that go with them.

The pretext task hides a share of the cells of every training row and asks the decoder to
recover them. A kNN graph built from the *clean* rows would defeat that: neighbours are picked
because they are similar in the very cells that are hidden, so an encoder can read the answer
off its neighbours instead of learning anything (kNN-imputation with the answer key). To avoid
this, every corruption ("view") gets its own kNN graph, built from a copy of the rows in which
each hidden cell was replaced by a random draw from the visible values of that column. The graph
of a view is then a function of the visible cells and a seed only; tests/test_corruption.py
checks exactly that. `neighbors="clean"` keeps the shortcut available as a control.

Training cycles through `n_views` fixed views (instead of drawing fresh masks every step)
because each kNN graph costs one brute-force search. All encoders use the same views, so they
see identical corrupted inputs. Nothing in this module needs torch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from graphmix.graphs import Graph, distance_features, graph_from_neighbors, knn_indices


@dataclass(frozen=True)
class View:
    """One corruption of a set of rows, plus the graph an encoder may use together with it."""

    m_num: np.ndarray  # (n, n_num) bool, True = hidden
    m_cat: np.ndarray  # (n, n_cat) bool, True = hidden
    graph: Graph | None


@dataclass(frozen=True)
class Pretext:
    """Everything an encoder needs besides the features."""

    train: list[View]  # over the training rows only; training cycles through these
    val: View  # over all rows; only validation rows are hidden (checkpoint selection)
    test: View  # over all rows; only test rows are hidden (masked-cell metrics, never selection)
    full: Graph | None  # clean graph over all rows, used for the final embeddings


def sample_masks(
    n: int, n_num: int, n_cat: int, p: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Bernoulli(p) cell masks (True = hidden)."""
    return rng.random((n, n_num)) < p, rng.random((n, n_cat)) < p


def _fill_column(
    col: np.ndarray, hidden: np.ndarray, pool: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    out = col.copy()
    at = np.flatnonzero(hidden)
    if at.size:
        source = pool if pool is not None else col[~hidden]
        if source.size == 0:
            raise ValueError("no visible value left to draw from; lower the mask ratio")
        out[at] = source[rng.integers(0, source.size, at.size)]
    return out


def fill_hidden(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    m_num: np.ndarray,
    m_cat: np.ndarray,
    rng: np.random.Generator,
    pool: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace every hidden cell by a random value of the same column.

    The value is drawn from the visible cells of the same column (or from `pool`, a pair of
    (numeric, categorical) arrays, e.g. the clean training rows). The result therefore does not
    depend on the hidden cells: unlike zero- or mean-filling, it also does not make rows with
    the same missingness pattern look alike.
    """
    filled_num, filled_cat = x_num.copy(), x_cat.copy()
    for j in range(x_num.shape[1]):
        filled_num[:, j] = _fill_column(
            x_num[:, j], m_num[:, j], None if pool is None else pool[0][:, j], rng
        )
    for j in range(x_cat.shape[1]):
        filled_cat[:, j] = _fill_column(
            x_cat[:, j], m_cat[:, j], None if pool is None else pool[1][:, j], rng
        )
    return filled_num, filled_cat


class Corruptor:
    """Builds views for one split. Masks and neighbour tables are cached, so methods that share a
    mask ratio share their corrupted inputs and their (expensive) searches."""

    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        cards: list[int],
        dist_clean: np.ndarray,
        n_train: int,
        n_val: int,
        seed: int,
        *,
        n_views: int,
        eval_ratio: float,
        cat_cost: float,
        chunk: int,
        k_max: int,
    ) -> None:
        self.x_num, self.x_cat, self.cards = x_num, x_cat, cards
        self.dist_clean = dist_clean
        self.n_train, self.n_val, self.n_total = n_train, n_val, len(x_num)
        self.seed, self.n_views, self.eval_ratio = seed, n_views, eval_ratio
        self.cat_cost, self.chunk, self.k_max = cat_cost, chunk, k_max
        self.seconds = 0.0  # time spent in neighbour searches on corrupted rows
        self._train_masks: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._train_tables: dict[float, list[np.ndarray]] = {}
        self._eval_tables: dict[tuple[str, float], np.ndarray] = {}

    # ---- masks

    def _rng(self, *key: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, *key])

    def train_masks(self, p: float) -> list[tuple[np.ndarray, np.ndarray]]:
        if p not in self._train_masks:
            n_views = self.n_views if p > 0 else 1
            rng = self._rng(1, round(p * 1000))
            self._train_masks[p] = [
                sample_masks(self.n_train, self.x_num.shape[1], self.x_cat.shape[1], p, rng)
                for _ in range(n_views)
            ]
        return self._train_masks[p]

    def _split_rows(self, split: str) -> tuple[int, int]:
        if split == "val":
            return self.n_train, self.n_train + self.n_val
        return self.n_train + self.n_val, self.n_total

    def eval_masks(self, split: str, p: float) -> tuple[np.ndarray, np.ndarray]:
        """Masks over all rows in which only rows of `split` can be hidden."""
        lo, hi = self._split_rows(split)
        m_num = np.zeros(self.x_num.shape, dtype=bool)
        m_cat = np.zeros(self.x_cat.shape, dtype=bool)
        rng = self._rng(2 if split == "val" else 3, round(p * 1000))
        m_num[lo:hi], m_cat[lo:hi] = sample_masks(hi - lo, self.x_num.shape[1], self.x_cat.shape[1], p, rng)
        return m_num, m_cat

    # ---- neighbour tables from corrupted rows

    def train_tables(self, p: float) -> list[np.ndarray]:
        """One (n_train, k_max) table per view: neighbours among the training rows, chosen from the
        filled-in copy of the corrupted rows."""
        if p not in self._train_tables:
            started = time.time()
            n = self.n_train
            tables = []
            for v, (m_num, m_cat) in enumerate(self.train_masks(p)):
                rng = self._rng(4, round(p * 1000), v)
                f_num, f_cat = fill_hidden(self.x_num[:n], self.x_cat[:n], m_num, m_cat, rng)
                feat = distance_features(f_num, f_cat, self.cards, self.cat_cost)
                tables.append(knn_indices(feat, feat, self.k_max, exclude_self=True, chunk=self.chunk))
            self._train_tables[p] = tables
            self.seconds += time.time() - started
        return self._train_tables[p]

    def eval_table(self, split: str, p: float) -> np.ndarray:
        """(rows of split, k_max) neighbours in the *clean* training pool for corrupted query rows."""
        key = (split, p)
        if key not in self._eval_tables:
            started = time.time()
            lo, hi = self._split_rows(split)
            m_num, m_cat = self.eval_masks(split, p)
            rng = self._rng(5, 2 if split == "val" else 3, round(p * 1000))
            pool = (self.x_num[: self.n_train], self.x_cat[: self.n_train])
            f_num, f_cat = fill_hidden(
                self.x_num[lo:hi], self.x_cat[lo:hi], m_num[lo:hi], m_cat[lo:hi], rng, pool
            )
            query = distance_features(f_num, f_cat, self.cards, self.cat_cost)
            self._eval_tables[key] = knn_indices(
                query, self.dist_clean[: self.n_train], self.k_max, chunk=self.chunk
            )
            self.seconds += time.time() - started
        return self._eval_tables[key]

    # ---- assembling a pretext

    def _eval_graph(self, split: str, p: float, k: int, clean_table: np.ndarray) -> Graph:
        """Clean graph, except that the rows of `split` pick their neighbours from what is visible."""
        lo, hi = self._split_rows(split)
        table = clean_table[:, :k].copy()
        table[lo:hi] = self.eval_table(split, p)[:, :k]
        return graph_from_neighbors(table, self.n_train)

    def pretext(
        self,
        kind: str,
        k: int,
        p: float,
        neighbors: str,
        full: Graph | None,
        clean_table: np.ndarray | None,
    ) -> Pretext:
        """`full` is the clean graph of the method (None for the flat MLP); `clean_table` the clean
        kNN table, needed only for kind sage_knn."""
        masks = self.train_masks(p)
        val_masks = self.eval_masks("val", p)
        test_masks = self.eval_masks("test", self.eval_ratio)
        if kind == "sage_knn" and neighbors == "visible":
            assert full is not None and clean_table is not None
            if p > 0:
                tables = self.train_tables(p)
                train_graphs: list[Graph | None] = [
                    graph_from_neighbors(t[:, :k], self.n_train) for t in tables
                ]
                val_graph: Graph | None = self._eval_graph("val", p, k, clean_table)
            else:
                train_graphs = [full.restrict_to_train()] * len(masks)
                val_graph = full
            test_graph: Graph | None = (
                self._eval_graph("test", self.eval_ratio, k, clean_table) if self.eval_ratio > 0 else full
            )
        else:
            train_graphs = [None if full is None else full.restrict_to_train()] * len(masks)
            val_graph = test_graph = full
        return Pretext(
            train=[View(mn, mc, g) for (mn, mc), g in zip(masks, train_graphs, strict=True)],
            val=View(*val_masks, val_graph),
            test=View(*test_masks, test_graph),
            full=full,
        )
