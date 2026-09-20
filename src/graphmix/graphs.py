"""Graph construction over a node set laid out as [train | eval].

Everything is numpy/scipy so it can be tested without torch. Convention throughout:
`edge_index[0]` is the message sender and `edge_index[1]` the receiver (PyG's default flow).
Evaluation nodes (validation/test) are attached to the training pool and are never senders
towards training nodes, so an evaluation row cannot influence a training embedding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

_INT_MAX = np.iinfo(np.int64).max


# --------------------------------------------------------------------------- distances / kNN


def distance_features(
    x_num: np.ndarray, x_cat: np.ndarray, cards: Sequence[int], cat_cost: float = 1.0
) -> np.ndarray:
    """Encoding whose squared Euclidean distance is a Gower-style mixed distance:
    sum of squared z-score gaps over numeric columns + cat_cost * (# categorical mismatches)."""
    scale = np.sqrt(cat_cost / 2.0)  # two one-hot coordinates differ per mismatch
    blocks = [x_num.astype(np.float64)]
    for j, card in enumerate(cards):
        blocks.append(np.eye(card, dtype=np.float64)[x_cat[:, j]] * scale)
    return np.concatenate(blocks, axis=1)


def knn_indices(
    query: np.ndarray,
    ref: np.ndarray,
    k: int,
    exclude_self: bool = False,
    chunk: int = 1024,
) -> np.ndarray:
    """Indices (into `ref`) of the k nearest rows for every query row, nearest first.

    Brute force, squared Euclidean, float64. Distances are rounded to 1e-6 and ties are broken
    by reference index, so the result does not depend on BLAS rounding noise or partition order.
    With `exclude_self`, `query` must be the same array as `ref` and a row never returns itself.
    """
    n_query, n_ref = query.shape[0], ref.shape[0]
    if exclude_self and query.shape != ref.shape:
        raise ValueError("exclude_self requires query and ref to be the same rows")
    if not 1 <= k <= n_ref - int(exclude_self):
        raise ValueError(f"k={k} not possible with {n_ref} reference rows")
    query = np.ascontiguousarray(query, dtype=np.float64)
    ref = np.ascontiguousarray(ref, dtype=np.float64)
    ref_sq = np.einsum("ij,ij->i", ref, ref)
    cols = np.arange(n_ref, dtype=np.int64)
    out = np.empty((n_query, k), dtype=np.int64)
    for start in range(0, n_query, chunk):
        q = query[start : start + chunk]
        d = np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * (q @ ref.T) + ref_sq[None, :]
        np.maximum(d, 0.0, out=d)
        if d.max() * 1e6 * n_ref >= 2**62:
            raise ValueError("distances too large for the integer tie-breaking key")
        key = np.rint(d * 1e6).astype(np.int64) * n_ref + cols[None, :]
        if exclude_self:
            rows = np.arange(q.shape[0])
            key[rows, start + rows] = _INT_MAX
        part = np.argpartition(key, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(key, part, axis=1), axis=1)
        out[start : start + chunk] = np.take_along_axis(part, order, axis=1)
    return out


# --------------------------------------------------------------------------- graph objects


@dataclass(frozen=True)
class Graph:
    """A graph over n_total nodes; the first n_train are training rows.

    kind "knn" / "random" / "selfloop": homogeneous, described by `edge_index` (2, E).
    kind "bipartite": record nodes plus one value node per (column, level); described by
    `val_ids` (n_total, n_cat), the global value-node id of every record's cell.
    """

    kind: str
    n_train: int
    n_total: int
    edge_index: np.ndarray | None = None
    val_ids: np.ndarray | None = None
    n_values: int = 0

    def restrict_to_train(self) -> Graph:
        """The graph the encoder sees during training: training nodes only."""
        n = self.n_train
        if self.kind == "bipartite":
            assert self.val_ids is not None
            return Graph(self.kind, n, n, val_ids=self.val_ids[:n], n_values=self.n_values)
        assert self.edge_index is not None
        kept = self.edge_index[:, self.edge_index[1] < n]
        if kept.size and kept[0].max() >= n:
            raise AssertionError("a training node receives from an evaluation node")
        return Graph(self.kind, n, n, edge_index=kept)


def knn_neighbors(feat: np.ndarray, n_train: int, k: int, chunk: int = 1024) -> np.ndarray:
    """(n_total, k) neighbour table: training rows use other training rows, evaluation rows
    use the training pool. Rows are sorted nearest first."""
    train, rest = feat[:n_train], feat[n_train:]
    table = [knn_indices(train, train, k, exclude_self=True, chunk=chunk)]
    if len(rest):
        table.append(knn_indices(rest, train, k, chunk=chunk))
    return np.vstack(table)


def graph_from_neighbors(neighbors: np.ndarray, n_train: int, kind: str = "knn") -> Graph:
    n_total, k = neighbors.shape
    src = neighbors.reshape(-1).astype(np.int64)
    dst = np.repeat(np.arange(n_total, dtype=np.int64), k)
    return Graph(kind, n_train, n_total, edge_index=np.stack([src, dst]))


def random_graph(n_train: int, n_total: int, k: int, seed: int) -> Graph:
    """Degree-matched control: every node receives from k senders drawn uniformly (with
    replacement) from the training pool, excluding itself."""
    rng = np.random.default_rng(seed)
    own = np.arange(n_train)[:, None]
    train_src = rng.integers(0, n_train - 1, size=(n_train, k))
    train_src = train_src + (train_src >= own)  # skip the node itself
    parts = [train_src]
    if n_total > n_train:
        parts.append(rng.integers(0, n_train, size=(n_total - n_train, k)))
    return graph_from_neighbors(np.vstack(parts), n_train, kind="random")


def selfloop_graph(n_train: int, n_total: int) -> Graph:
    """No relational information: every node only receives from itself."""
    idx = np.arange(n_total, dtype=np.int64)
    return Graph("selfloop", n_train, n_total, edge_index=np.stack([idx, idx]))


def bipartite_graph(x_cat: np.ndarray, cards: Sequence[int], n_train: int) -> Graph:
    offsets = np.concatenate([[0], np.cumsum(cards)[:-1]]).astype(np.int64)
    val_ids = x_cat.astype(np.int64) + offsets[None, :]
    return Graph("bipartite", n_train, len(x_cat), val_ids=val_ids, n_values=int(sum(cards)))


# --------------------------------------------------------------------------- propagation


def row_normalized(graph: Graph) -> sparse.csr_matrix:
    """Mean-aggregation operator of a homogeneous graph, rows = receivers: entry (i, j) is the
    number of edges j -> i divided by the in-degree of i (repeated edges count repeatedly; a node
    without incoming edges gets an all-zero row)."""
    assert graph.edge_index is not None
    n = graph.n_total
    src, dst = graph.edge_index
    adj = sparse.coo_matrix((np.ones(src.size), (dst, src)), shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(adj.sum(axis=1)).ravel(), 1.0)
    op = (sparse.diags(1.0 / deg) @ adj).tocsr()
    op.sort_indices()
    return op


def make_propagator(graph: Graph) -> Callable[[np.ndarray], np.ndarray]:
    """Return f with f(x)[i] = mean of x over the nodes that send to i (a row-stochastic operator).

    For the bipartite graph one application is a record -> value -> record round trip, i.e. two
    message-passing layers: the mean, over the record's values, of the mean over the *training*
    records that share that value. It is applied as two sparse products; the record-record
    matrix is never formed (hub values such as sex would make it almost dense).
    """
    n = graph.n_total
    if graph.kind == "bipartite":
        assert graph.val_ids is not None
        n_cat = graph.val_ids.shape[1]
        rows = np.repeat(np.arange(n), n_cat)
        ones = np.ones(rows.size)
        inc = sparse.csr_matrix((ones, (rows, graph.val_ids.reshape(-1))), shape=(n, graph.n_values))
        train_inc_t = inc[: graph.n_train].T.tocsr()
        size = np.maximum(np.asarray(train_inc_t.sum(axis=1)).ravel(), 1.0)

        def step(x: np.ndarray) -> np.ndarray:
            group_mean = (train_inc_t @ x[: graph.n_train]) / size[:, None]
            return (inc @ group_mean) / n_cat

        return step

    op = row_normalized(graph)
    return lambda x: op @ x


def propagate(graph: Graph, x: np.ndarray, hops: int) -> np.ndarray:
    """Parameter-free message passing: concatenate [x, A x, ..., A^hops x] along the features."""
    step = make_propagator(graph)
    outs = [x.astype(np.float64)]
    for _ in range(hops):
        outs.append(step(outs[-1]))
    return np.concatenate(outs, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- diagnostics
# These use labels and are only ever called after representations have been computed.


def label_agreement_by_chance(y: np.ndarray) -> float:
    p = float(np.mean(y))
    return p * p + (1 - p) * (1 - p)


def edge_homophily(edge_index: np.ndarray, y: np.ndarray, n_train: int) -> float:
    """Share of edges into training nodes whose two endpoints have the same label."""
    src, dst = edge_index
    keep = dst < n_train
    return float(np.mean(y[src[keep]] == y[dst[keep]]))


def adjusted_homophily(edge_index: np.ndarray, y: np.ndarray, n_train: int) -> float:
    """Edge homophily corrected for class imbalance (Platonov et al., 2023), treating edges as
    undirected: (h - sum_c p_c^2) / (1 - sum_c p_c^2) with p_c the share of edge endpoints in c."""
    src, dst = edge_index
    keep = dst < n_train
    ends = np.concatenate([y[src[keep]], y[dst[keep]]])
    p = np.bincount(ends, minlength=2) / ends.size
    chance = float(np.sum(p**2))
    return (edge_homophily(edge_index, y, n_train) - chance) / (1.0 - chance)


def column_label_agreement(val_ids: np.ndarray, y: np.ndarray, n_train: int) -> np.ndarray:
    """Per categorical column: mean over training rows of the fraction of the *other* training
    rows sharing the row's value that have the same label. This is the bipartite analogue of
    edge homophily (what a mean aggregator over a value node sees)."""
    yt = y[:n_train].astype(np.int64)
    out = np.empty(val_ids.shape[1])
    for c in range(val_ids.shape[1]):
        codes = val_ids[:n_train, c]
        _, codes = np.unique(codes, return_inverse=True)
        size = np.bincount(codes)
        pos = np.bincount(codes, weights=yt)
        same = np.where(yt == 1, pos[codes], size[codes] - pos[codes]) - 1.0
        others = size[codes] - 1.0
        ok = others > 0
        out[c] = np.mean(same[ok] / others[ok])
    return out


def sender_degree_stats(edge_index: np.ndarray, n_train: int) -> dict[str, float]:
    """How often each training node is chosen as a neighbour (hubness of a kNN graph)."""
    src, dst = edge_index
    counts = np.bincount(src[dst < n_train], minlength=n_train).astype(np.float64)
    centred = counts - counts.mean()
    std = counts.std()
    return {
        "sender_deg_max": float(counts.max()),
        "sender_deg_skew": float(np.mean(centred**3) / std**3) if std > 0 else 0.0,
        "frac_never_sender": float(np.mean(counts == 0)),
    }
