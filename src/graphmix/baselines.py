"""Representations that need no neural network."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from graphmix.graphs import Graph, propagate

# Number of applications of the mean operator that matches a 2-layer GNN receptive field:
# two hops on a kNN / random graph, one record -> value -> record round trip on the bipartite graph.
PROP_HOPS = {"knn": 2, "random": 2, "bipartite": 1}


def pca_embedding(feat: np.ndarray, n_train: int, dim: int, seed: int) -> np.ndarray:
    """PCA fitted on the training rows, applied to every row."""
    dim = min(dim, feat.shape[1], n_train)
    pca = PCA(n_components=dim, svd_solver="full", random_state=seed).fit(feat[:n_train])
    return pca.transform(feat).astype(np.float32)


def propagated_embedding(graph: Graph, feat: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """[x, Ax, A^2 x] (SIGN / SGC-style, no learned weights) compressed with PCA to `dim`.
    Same output size as the learned encoders, so the probes see equal-width inputs."""
    stacked = propagate(graph, feat, PROP_HOPS[graph.kind])
    return pca_embedding(stacked, graph.n_train, dim, seed)
