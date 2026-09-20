"""Frozen-embedding evaluation: linear probe, low-label probe, kNN retrieval, summaries.

Labels are used here and only here (plus homophily diagnostics); no encoder ever sees them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from graphmix.config import EvalConfig
from graphmix.graphs import knn_indices


@dataclass
class Embedding:
    z_train: np.ndarray
    z_val: np.ndarray
    z_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def save_embedding(path: str | Path, emb: Embedding, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z_train=emb.z_train.astype(np.float32),
        z_val=emb.z_val.astype(np.float32),
        z_test=emb.z_test.astype(np.float32),
        y_train=emb.y_train,
        y_val=emb.y_val,
        y_test=emb.y_test,
        meta=np.array(json.dumps(meta)),
    )


def load_embedding(path: str | Path) -> tuple[Embedding, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as npz:
        emb = Embedding(*(npz[k] for k in ("z_train", "z_val", "z_test", "y_train", "y_val", "y_test")))
        meta = json.loads(str(npz["meta"]))
    return emb, meta


# --------------------------------------------------------------------------- metrics


def _standardise(emb: Embedding) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-score every dimension with training-row statistics."""
    mu = emb.z_train.mean(axis=0)
    sd = emb.z_train.std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (emb.z_train - mu) / sd, (emb.z_val - mu) / sd, (emb.z_test - mu) / sd


def _logreg(c: float) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=5000)


def linear_probe(emb: Embedding, c_grid: list[float]) -> dict[str, float]:
    """Logistic regression on frozen embeddings; C picked on the validation AUC."""
    z_tr, z_va, z_te = _standardise(emb)
    best_auc, best_clf, best_c = -1.0, None, c_grid[0]
    for c in c_grid:
        clf = _logreg(c).fit(z_tr, emb.y_train)
        auc = roc_auc_score(emb.y_val, clf.predict_proba(z_va)[:, 1])
        if auc > best_auc:
            best_auc, best_clf, best_c = auc, clf, c
    assert best_clf is not None
    p = best_clf.predict_proba(z_te)[:, 1]
    return {
        "probe_auc": roc_auc_score(emb.y_test, p),
        "probe_ap": average_precision_score(emb.y_test, p),
        "probe_acc": accuracy_score(emb.y_test, p > 0.5),
        "probe_C": best_c,
    }


def label_efficiency(
    emb: Embedding, sizes: list[int], repeats: int, seed: int, c: float = 1.0
) -> dict[str, float]:
    """Test AUC of a probe fitted on n randomly chosen labelled training rows (C fixed).
    The same rows are drawn for every method within a seed, so comparisons are paired."""
    z_tr, _, z_te = _standardise(emb)
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for n in sizes:
        n = min(n, len(z_tr))
        aucs = []
        for _ in range(repeats):
            idx = rng.choice(len(z_tr), size=n, replace=False)
            if len(np.unique(emb.y_train[idx])) < 2:
                continue
            clf = _logreg(c).fit(z_tr[idx], emb.y_train[idx])
            aucs.append(roc_auc_score(emb.y_test, clf.predict_proba(z_te)[:, 1]))
        out[f"lowlabel_auc@{n}"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


def _unit(z: np.ndarray, mu: np.ndarray) -> np.ndarray:
    z = z - mu
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def retrieval(emb: Embedding, k: int = 25, k_pos: int = 10) -> dict[str, float]:
    """Cosine kNN from test rows into the training rows.

    knn_auc: AUC of the share of positive labels among the k neighbours.
    knn_pos_p10: mean share of positives among the 10 neighbours of positive test rows
    (compare with the base rate of positives)."""
    mu = emb.z_train.mean(axis=0)
    z_tr, z_te = _unit(emb.z_train, mu), _unit(emb.z_test, mu)
    nb = knn_indices(z_te, z_tr, k=max(k, k_pos))
    y_nb = emb.y_train[nb]
    pos = emb.y_test == 1
    return {
        "knn_auc": roc_auc_score(emb.y_test, y_nb[:, :k].mean(axis=1)),
        "knn_pos_p10": float(y_nb[pos, :k_pos].mean()),
    }


def hgb_reference(emb: Embedding, seed: int) -> dict[str, float]:
    """Supervised gradient-boosting reference on the flat features (not a representation)."""
    clf = HistGradientBoostingClassifier(random_state=seed).fit(emb.z_train, emb.y_train)
    p = clf.predict_proba(emb.z_test)[:, 1]
    return {
        "probe_auc": roc_auc_score(emb.y_test, p),
        "probe_ap": average_precision_score(emb.y_test, p),
        "probe_acc": accuracy_score(emb.y_test, p > 0.5),
    }


def evaluate_embedding(emb: Embedding, cfg: EvalConfig, seed: int) -> dict[str, float]:
    out = linear_probe(emb, cfg.c_grid)
    out.update(label_efficiency(emb, cfg.label_sizes, cfg.label_repeats, seed))
    out.update(retrieval(emb, k=cfg.knn_k))
    out["base_rate"] = float(np.mean(emb.y_test))
    return out


# --------------------------------------------------------------------------- aggregation


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """metrics: long table with columns seed, method, metric, value -> mean / std / n."""
    return metrics.groupby(["method", "metric"])["value"].agg(mean="mean", std="std", n="count").reset_index()


def paired_delta(metrics: pd.DataFrame, method: str, baseline: str, metric: str) -> dict[str, float]:
    """Per-seed difference method - baseline (same split and probe subsets within a seed)."""
    sub = metrics[metrics["metric"] == metric]
    wide = sub.pivot(index="seed", columns="method", values="value")
    if method not in wide or baseline not in wide:
        return {"mean": float("nan"), "std": float("nan"), "wins": float("nan"), "n": 0.0}
    diff = (wide[method] - wide[baseline]).dropna()
    return {
        "mean": float(diff.mean()),
        "std": float(diff.std()) if len(diff) > 1 else float("nan"),
        "wins": float((diff > 0).sum()),
        "n": float(len(diff)),
    }


DIGITS = {"dim": 0, "n_params": 0, "best_step": 0, "train_seconds": 1}  # default: 3


def _cell(mean: float, std: float, n: int, digits: int) -> str:
    # a spread of exactly zero (the width of an embedding, a parameter count) is not worth printing
    if n > 1 and not np.isnan(std) and std > 0:
        return f"{mean:.{digits}f} ± {std:.{digits}f}"
    return f"{mean:.{digits}f}"


def markdown_table(
    summary: pd.DataFrame,
    methods: list[str],
    metrics: list[str],
    labels: dict[str, str] | None = None,
) -> str:
    """Rows = methods (in the given order), columns = metrics; cells are mean ± std over seeds."""
    labels = labels or {}
    stats = {
        (str(m), str(k)): (float(mean), float(std), int(n))
        for m, k, mean, std, n in zip(
            summary["method"], summary["metric"], summary["mean"], summary["std"], summary["n"], strict=True
        )
    }
    header = "| method | " + " | ".join(labels.get(m, m) for m in metrics) + " |"
    rule = "|---|" + "---:|" * len(metrics)
    rows = []
    for method in methods:
        cells = []
        for metric in metrics:
            if (method, metric) in stats:
                mean, std, n = stats[(method, metric)]
                cells.append(_cell(mean, std, n, DIGITS.get(metric, 3)))
            else:
                cells.append("n/a")
        rows.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *rows])


def frame_to_markdown(df: pd.DataFrame, digits: int | dict[str, int] = 3) -> str:
    """Small DataFrame -> markdown table (index becomes the first column). `digits` may be given
    per column (by column name; columns that are not listed get 3)."""
    head = "| " + " | ".join([str(df.index.name or ""), *map(str, df.columns)]) + " |"
    rule = "|---|" + "---:|" * len(df.columns)
    rows = []
    for idx, row in df.iterrows():
        cells = []
        for col, v in row.items():
            d = digits if isinstance(digits, int) else digits.get(str(col), 3)
            cells.append(f"{v:.{d}f}" if pd.notna(v) else "n/a")
        rows.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join([head, rule, *rows])
