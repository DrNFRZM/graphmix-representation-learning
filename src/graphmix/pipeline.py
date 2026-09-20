"""Glue: split -> graphs -> representations (train stage), embeddings -> metrics (evaluate stage).

Stage 1 (`run_training`) writes one .npz of embeddings per (seed, method) under runs/<name>/.
Stage 2 (`run_evaluation`) reads them back and writes metrics, tables and figures under
results/<name>/. torch is imported lazily, so the non-neural methods run without it.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from graphmix.baselines import pca_embedding, propagated_embedding
from graphmix.config import GRAPH_OF_KIND, NEURAL_KINDS, PROP_KINDS, Config, MethodConfig
from graphmix.corruption import Corruptor, Pretext
from graphmix.data import (
    LABEL,
    Preprocessor,
    describe,
    flat_features,
    load_clean,
    make_splits,
    subsample,
)
from graphmix.evaluate import (
    Embedding,
    evaluate_embedding,
    frame_to_markdown,
    hgb_reference,
    load_embedding,
    markdown_table,
    paired_delta,
    save_embedding,
    summarize,
)
from graphmix.graphs import (
    Graph,
    adjusted_homophily,
    bipartite_graph,
    column_label_agreement,
    distance_features,
    edge_homophily,
    graph_from_neighbors,
    knn_neighbors,
    label_agreement_by_chance,
    random_graph,
    selfloop_graph,
    sender_degree_stats,
)

# --------------------------------------------------------------------------- split and graphs


@dataclass
class SplitData:
    """One seeded split. Rows are ordered [train | val | test]; the preprocessor was fitted on
    the training rows only."""

    seed: int
    n_train: int
    n_val: int
    n_test: int
    x_num: np.ndarray
    x_cat: np.ndarray
    y: np.ndarray
    cards: list[int]
    cat_names: list[str]
    flat: np.ndarray  # numeric + one-hot, for raw / PCA / propagation
    dist: np.ndarray  # mixed-type distance features for the kNN graph

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test

    def embedding(self, z: np.ndarray) -> Embedding:
        a, b = self.n_train, self.n_train + self.n_val
        return Embedding(z[:a], z[a:b], z[b:], self.y[:a], self.y[a:b], self.y[b:])


def prepare_split(clean_df: pd.DataFrame, cfg: Config, seed: int) -> SplitData:
    df = subsample(clean_df, cfg.data.n_rows, seed)
    tr, va, te = make_splits(df[LABEL].to_numpy(), cfg.data.val_frac, cfg.data.test_frac, seed)
    df = df.iloc[np.concatenate([tr, va, te])].reset_index(drop=True)
    pre = Preprocessor(min_freq=cfg.data.min_freq).fit(df.iloc[: len(tr)])
    td = pre.transform(df)
    return SplitData(
        seed=seed,
        n_train=len(tr),
        n_val=len(va),
        n_test=len(te),
        x_num=td.x_num,
        x_cat=td.x_cat,
        y=df[LABEL].to_numpy(),
        cards=pre.cards,
        cat_names=pre.cat_cols,
        flat=flat_features(td.x_num, td.x_cat, pre.cards),
        dist=distance_features(td.x_num, td.x_cat, pre.cards, cfg.graph.cat_cost),
    )


class GraphFactory:
    """Builds each graph once per split. The kNN table is computed for the largest k that any
    method asks for and sliced for the smaller ones. Also owns the Corruptor, so that methods
    with the same mask ratio share their corrupted views."""

    def __init__(self, sd: SplitData, cfg: Config) -> None:
        self.sd, self.cfg = sd, cfg
        ks = [cfg.k_of(m) for m in cfg.methods if GRAPH_OF_KIND.get(m.kind) == "knn"]
        self.k_max = max(ks, default=0)
        self._neighbors: np.ndarray | None = None
        self.knn_seconds: float | None = None  # brute-force search for the k_max table
        self.cache: dict[tuple[str, int], Graph] = {}
        self._corruptor: Corruptor | None = None
        self.visible: set[tuple[float, int]] = set()  # (mask ratio, k) of corrupted kNN graphs in use

    def get(self, kind: str, k: int) -> Graph:
        key = (kind, k if kind in ("knn", "random") else 0)
        if key not in self.cache:
            self.cache[key] = self._build(kind, k)
        return self.cache[key]

    def _build(self, kind: str, k: int) -> Graph:
        sd = self.sd
        if kind == "knn":
            if self._neighbors is None:
                started = time.time()
                self._neighbors = knn_neighbors(sd.dist, sd.n_train, self.k_max, self.cfg.graph.chunk)
                self.knn_seconds = time.time() - started
            return graph_from_neighbors(self._neighbors[:, :k], sd.n_train)
        if kind == "random":
            return random_graph(sd.n_train, sd.n_total, k, sd.seed)
        if kind == "selfloop":
            return selfloop_graph(sd.n_train, sd.n_total)
        if kind == "bipartite":
            return bipartite_graph(sd.x_cat, sd.cards, sd.n_train)
        raise ValueError(f"unknown graph kind '{kind}'")

    @property
    def corruptor(self) -> Corruptor:
        if self._corruptor is None:
            sd, cfg = self.sd, self.cfg
            self._corruptor = Corruptor(
                sd.x_num,
                sd.x_cat,
                sd.cards,
                sd.dist,
                sd.n_train,
                sd.n_val,
                sd.seed,
                n_views=cfg.train.n_views,
                eval_ratio=cfg.train.mask_ratio,
                cat_cost=cfg.graph.cat_cost,
                chunk=cfg.graph.chunk,
                k_max=self.k_max,
            )
        return self._corruptor

    def _note_visible(self, spec: MethodConfig) -> None:
        """Remember which (mask ratio, k) corrupted kNN graphs a method trains on, for the diagnostics."""
        p = self.cfg.mask_of(spec)
        if spec.kind == "sage_knn" and spec.neighbors == "visible" and p > 0:
            self.visible.add((p, self.cfg.k_of(spec)))

    def pretext(self, spec: MethodConfig) -> Pretext:
        """Corrupted views (and the graph that goes with each) for one neural method."""
        graph_kind = GRAPH_OF_KIND.get(spec.kind)
        k, p = self.cfg.k_of(spec), self.cfg.mask_of(spec)
        full = self.get(graph_kind, k) if graph_kind else None
        table = self._neighbors if spec.kind == "sage_knn" else None  # filled in by get("knn", k) above
        self._note_visible(spec)
        return self.corruptor.pretext(spec.kind, k, p, spec.neighbors, full, table)

    def register_all(self) -> None:
        """Build the graphs of every configured method, including those whose embeddings already
        exist. Normally nothing is left to do by the time the diagnostics are written; after an
        interrupted run the skipped methods would otherwise be missing from them."""
        for spec in self.cfg.methods:
            graph_kind = GRAPH_OF_KIND.get(spec.kind)
            if graph_kind is not None:
                self.get(graph_kind, self.cfg.k_of(spec))
                self._note_visible(spec)


def _edge_stats(edge_index: np.ndarray, y: np.ndarray, n_train: int) -> dict[str, float]:
    return {
        "edge_homophily": edge_homophily(edge_index, y, n_train),
        "adjusted_homophily": adjusted_homophily(edge_index, y, n_train),
        **sender_degree_stats(edge_index, n_train),
    }


def graph_diagnostics(factory: GraphFactory) -> list[dict[str, Any]]:
    """Label-based descriptions of the graphs that were built (training rows only)."""
    sd = factory.sd
    rows: list[dict[str, Any]] = []

    def add(graph: str, metric: str, value: float) -> None:
        rows.append({"graph": graph, "metric": metric, "value": float(value)})

    for (kind, k), graph in factory.cache.items():
        name = f"{kind}_k{k}" if kind in ("knn", "random") else kind
        if kind == "bipartite":
            assert graph.val_ids is not None
            agree = column_label_agreement(graph.val_ids, sd.y, sd.n_train)
            for col, value in zip(sd.cat_names, agree, strict=True):
                add(name, f"agreement[{col}]", value)
            add(name, "agreement_mean", agree.mean())
        elif kind != "selfloop":
            assert graph.edge_index is not None
            for metric, value in _edge_stats(graph.edge_index, sd.y, sd.n_train).items():
                add(name, metric, value)
            if kind == "knn" and factory.knn_seconds is not None:
                add(name, "build_seconds", factory.knn_seconds)
    # the kNN graphs of the corrupted views: what the encoder sees during training
    for p, k in sorted(factory.visible):
        per_view = []
        for table in factory.corruptor.train_tables(p):
            edges = graph_from_neighbors(table[:, :k], sd.n_train).edge_index
            assert edges is not None
            per_view.append(_edge_stats(edges, sd.y, sd.n_train))
        name = f"knn_visible_k{k}_p{p:g}"
        for metric in per_view[0]:
            add(name, metric, float(np.mean([v[metric] for v in per_view])))
        add(name, "build_seconds", factory.corruptor.seconds)
    add("chance", "label_agreement_by_chance", label_agreement_by_chance(sd.y[: sd.n_train]))
    return rows


def reference_imputation(sd: SplitData, factory: GraphFactory) -> dict[str, float]:
    """Scale for the masked-cell scores: predict every hidden test cell by the training column
    mean (numeric) or mode (categorical)."""
    m_num, m_cat = factory.corruptor.eval_masks("test", factory.cfg.train.mask_ratio)
    te = slice(sd.n_train + sd.n_val, sd.n_total)
    m_num, m_cat = m_num[te], m_cat[te]
    if not m_num.any():
        return {}
    mean = sd.x_num[: sd.n_train].mean(axis=0)
    mse = float(np.mean(((sd.x_num[te] - mean) ** 2)[m_num]))
    accs = []
    for j, card in enumerate(sd.cards):
        mode = np.bincount(sd.x_cat[: sd.n_train, j], minlength=card).argmax()
        accs.append(np.mean((sd.x_cat[te, j] == mode)[m_cat[:, j]]))
    return {"masked_num_mse_column_mean": mse, "masked_cat_acc_column_mode": float(np.mean(accs))}


# --------------------------------------------------------------------------- stage 1: training


def compute_representation(
    spec: MethodConfig, sd: SplitData, cfg: Config, factory: GraphFactory, verbose: bool = False
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame | None]:
    """Embedding of every row (order [train | val | test]), run metadata, training log."""
    meta: dict[str, Any] = {"method": spec.name, "kind": spec.kind, "seed": sd.seed}
    if spec.kind == "raw":
        return sd.flat, meta, None
    if spec.kind == "pca":
        return pca_embedding(sd.flat, sd.n_train, cfg.model.dim, sd.seed), meta, None

    graph_kind = GRAPH_OF_KIND.get(spec.kind)
    if graph_kind in ("knn", "random"):
        meta["k"] = cfg.k_of(spec)
    if spec.kind in PROP_KINDS:
        assert graph_kind is not None
        graph = factory.get(graph_kind, cfg.k_of(spec))
        return propagated_embedding(graph, sd.flat, cfg.model.dim, sd.seed), meta, None

    assert spec.kind in NEURAL_KINDS
    from graphmix.train import train_autoencoder  # needs torch and torch_geometric

    train_cfg = replace(cfg.train, lr=spec.lr) if spec.lr is not None else cfg.train
    res = train_autoencoder(
        spec.kind,
        sd.x_num,
        sd.x_cat,
        sd.cards,
        factory.pretext(spec),
        sd.n_train,
        sd.n_val,
        cfg.model,
        train_cfg,
        sd.seed,
        verbose=verbose,
    )
    meta.update(
        mask_ratio=cfg.mask_of(spec),
        lr=train_cfg.lr,
        n_params=res.n_params,
        best_step=res.best_step,
        train_seconds=res.seconds,
        recon_num_mse=res.recon_num_mse,
        recon_cat_acc=res.recon_cat_acc,
        masked_num_mse=res.masked_num_mse,
        masked_cat_acc=res.masked_cat_acc,
    )
    if spec.kind == "sage_knn":
        meta["neighbors"] = spec.neighbors
    return res.z, meta, res.log


def run_training(
    cfg: Config,
    seeds: list[int] | None = None,
    only: list[str] | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    clean_df, report = load_clean(cfg.data.raw_dir, cfg.data.files)
    if verbose:
        print(f"[{cfg.name}] data after cleaning\n{describe(report)}\n")
    unknown = set(only or []) - {m.name for m in cfg.methods}
    if unknown:
        raise ValueError(f"unknown method(s) in --only: {sorted(unknown)}")

    for seed in cfg.seeds if seeds is None else seeds:
        sd = prepare_split(clean_df, cfg, seed)
        factory = GraphFactory(sd, cfg)
        out_dir = cfg.run_path / f"seed{seed}"
        if verbose:
            print(f"[{cfg.name}] seed {seed}: train/val/test = {sd.n_train}/{sd.n_val}/{sd.n_test}")
        for spec in cfg.methods:
            if only and spec.name not in only:
                continue
            path = out_dir / f"{spec.name}.npz"
            if path.exists() and not overwrite:
                if verbose:
                    print(f"  {spec.name:18s} exists, skipping")
                continue
            t0 = time.time()
            z, meta, log = compute_representation(spec, sd, cfg, factory)
            meta["wall_seconds"] = time.time() - t0
            save_embedding(path, sd.embedding(z), meta)
            if log is not None:
                log.to_csv(path.with_suffix(".log.csv"), index=False)
            if verbose:
                print(f"  {spec.name:18s} dim={z.shape[1]:3d}  {meta['wall_seconds']:6.1f}s")
        out_dir.mkdir(parents=True, exist_ok=True)
        if only and (out_dir / "graph_stats.json").exists():
            continue  # a partial re-run leaves the diagnostics of the full run alone
        (out_dir / "reference.json").write_text(json.dumps(reference_imputation(sd, factory), indent=1))
        # labels are touched only here, after every representation has been computed
        factory.register_all()
        (out_dir / "graph_stats.json").write_text(json.dumps(graph_diagnostics(factory), indent=1))


# --------------------------------------------------------------------------- stage 2: evaluation

METRIC_LABELS = {
    "dim": "dim",
    "probe_auc": "probe AUC",
    "probe_ap": "probe AP",
    "probe_acc": "probe acc",
    "knn_auc": "kNN AUC",
    "knn_pos_p10": "kNN pos. P@10",
}


TRAINING_KEYS = (
    "recon_num_mse",
    "recon_cat_acc",
    "masked_num_mse",
    "masked_cat_acc",
    "n_params",
    "best_step",
    "train_seconds",
)


def collect_metrics(cfg: Config, seeds: list[int] | None = None) -> pd.DataFrame:
    """Evaluate every saved embedding; returns a long table (seed, method, metric, value)."""
    rows: list[tuple[int, str, str, float]] = []
    for seed in cfg.seeds if seeds is None else seeds:
        for spec in cfg.methods:
            path = cfg.run_path / f"seed{seed}" / f"{spec.name}.npz"
            if not path.exists():
                print(f"  missing {path}, skipped")
                continue
            emb, meta = load_embedding(path)
            metrics = evaluate_embedding(emb, cfg.evaluation, seed)
            for key in TRAINING_KEYS:
                if key in meta:
                    metrics[key] = float(meta[key])
            metrics["dim"] = float(emb.z_train.shape[1])
            rows += [(seed, spec.name, key, float(val)) for key, val in metrics.items()]
            if spec.kind == "raw" and cfg.evaluation.hgb_reference:
                ref = hgb_reference(emb, seed)
                rows += [(seed, "hgb_ref", key, float(val)) for key, val in ref.items()]
    return pd.DataFrame(rows, columns=["seed", "method", "metric", "value"])


def _collect_files(cfg: Config, seeds: list[int] | None, pattern: str, columns: list[str]) -> pd.DataFrame:
    frames = []
    for seed in cfg.seeds if seeds is None else seeds:
        for spec in cfg.methods:
            path = cfg.run_path / f"seed{seed}" / pattern.format(method=spec.name)
            if path.exists():
                df = pd.read_csv(path)
                df.insert(0, "method", spec.name)
                df.insert(0, "seed", seed)
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def collect_logs(cfg: Config, seeds: list[int] | None = None) -> pd.DataFrame:
    cols = ["seed", "method", "step", "train_loss", "val_loss"]
    return _collect_files(cfg, seeds, "{method}.log.csv", cols)


def collect_graph_stats(cfg: Config, seeds: list[int] | None = None) -> pd.DataFrame:
    frames = []
    for seed in cfg.seeds if seeds is None else seeds:
        path = cfg.run_path / f"seed{seed}" / "graph_stats.json"
        if path.exists():
            df = pd.DataFrame(json.loads(path.read_text()))
            df.insert(0, "seed", seed)
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["seed", "graph", "metric", "value"])
    return pd.concat(frames, ignore_index=True)


def collect_reference(cfg: Config, seeds: list[int] | None = None) -> dict[str, float]:
    """Mean over seeds of the column-mean / column-mode imputation scores (see reference_imputation)."""
    values: dict[str, list[float]] = {}
    for seed in cfg.seeds if seeds is None else seeds:
        path = cfg.run_path / f"seed{seed}" / "reference.json"
        if path.exists():
            for key, value in json.loads(path.read_text()).items():
                values.setdefault(key, []).append(float(value))
    return {key: float(np.mean(v)) for key, v in values.items()}


def paired_table(metrics: pd.DataFrame, methods: list[str], baseline: str, tracked: list[str]) -> str:
    """Rows = methods, columns = per-seed difference to `baseline` (mean ± std) and the number of
    seeds in which the method is ahead, for each tracked metric."""
    head = ["method"]
    for metric in tracked:
        head += [f"Δ {METRIC_LABELS.get(metric, metric)}", "wins"]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for method in methods:
        if method == baseline:
            continue
        cells = [method]
        for metric in tracked:
            d = paired_delta(metrics, method, baseline, metric)
            if d["n"] == 0:
                cells += ["n/a", "n/a"]
            else:
                spread = f" ± {d['std']:.3f}" if d["n"] > 1 else ""
                cells += [f"{d['mean']:+.3f}{spread}", f"{int(d['wins'])}/{int(d['n'])}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def timing_table(metrics: pd.DataFrame, methods: list[str]) -> str:
    """Median and range over seeds of the wall-clock training time. Median, because wall-clock
    time is heavy-tailed when the machine is busy with something else."""
    sub = metrics[metrics["metric"] == "train_seconds"]
    lines = ["| method | median s | range s |", "|---|---:|---:|"]
    for method in methods:
        v = sub.loc[sub["method"] == method, "value"]
        if len(v):
            lines.append(f"| {method} | {v.median():.0f} | {v.min():.0f} to {v.max():.0f} |")
    return "\n".join(lines)


GRAPH_TABLE = {  # column of graph_stats -> (header, digits)
    "edge_homophily": ("edge homophily", 3),
    "adjusted_homophily": ("adjusted homophily", 3),
    "frac_never_sender": ("never chosen", 3),
    "sender_deg_max": ("max times chosen", 1),
    "sender_deg_skew": ("skew of times chosen", 2),
    "build_seconds": ("build s", 1),
}


def graph_table(graph_stats: pd.DataFrame) -> list[str]:
    """Markdown lines for the graph diagnostics: one row per kNN/random graph, and a sentence for the
    record-value graph, whose diagnostic is a per-column label agreement instead of an edge statistic."""
    wide = graph_stats.groupby(["graph", "metric"])["value"].mean().unstack("metric")
    edge_graphs = [g for g in wide.index if g not in ("bipartite", "chance")]
    cols = [c for c in GRAPH_TABLE if c in wide.columns]
    table = wide.loc[edge_graphs, cols].rename(columns={c: GRAPH_TABLE[c][0] for c in cols})
    table.index.name = "graph"
    digits = {GRAPH_TABLE[c][0]: GRAPH_TABLE[c][1] for c in cols}
    lines = [frame_to_markdown(table, digits), ""]
    lines += [
        "Adjusted homophily is 0 for a graph that ignores the label and 1 for one that only links equal "
        "labels (Platonov et al., 2023). `knn_k*` graphs are built from complete rows; `knn_visible_*` "
        "graphs are those of the corrupted views on which the kNN encoders train (mean over the views). "
        "One neighbour search serves every k, so its build time is shared. *never chosen* is the share of "
        "training rows that no row picks as a neighbour.",
        "",
    ]
    if {"bipartite", "chance"} <= set(wide.index) and "agreement_mean" in wide.columns:
        agree = wide.loc["bipartite", "agreement_mean"]
        chance = wide.loc["chance", "label_agreement_by_chance"]
        lines += [
            "Record-value graph: averaged over columns and training rows, "
            f"{agree:.3f} of the other records that share a categorical value with a record have the "
            f"same label, against {chance:.3f} expected by chance (per column: `graph_diagnostics.csv`).",
            "",
        ]
    return lines


def summary_markdown(
    cfg: Config, metrics: pd.DataFrame, graph_stats: pd.DataFrame, reference: dict[str, float] | None = None
) -> str:
    summary = summarize(metrics)
    present = set(metrics["method"])
    order = [m.name for m in cfg.methods if m.name in present]
    if "hgb_ref" in present:
        order.append("hgb_ref")
    learned = [m for m in order if m in set(metrics.loc[metrics["metric"] == "n_params", "method"])]
    sizes = sorted(
        {m for m in metrics["metric"] if m.startswith("lowlabel_auc@")},
        key=lambda s: int(s.split("@")[1]),
    )
    n_seeds = metrics["seed"].nunique()

    parts = [
        f"Run `{cfg.name}`; random seeds: {n_seeds}. Cells are mean ± standard deviation across "
        "seeds (the standard deviation is omitted for a single seed). The splits overlap, so these "
        "spreads are not confidence intervals.",
        "",
        "### Frozen-embedding evaluation (test rows)",
        "",
        markdown_table(summary, order, list(METRIC_LABELS), METRIC_LABELS),
        "",
        "`hgb_ref` is a supervised gradient-boosting reference on the flat features, not a representation.",
        "",
        "### Label efficiency (test AUC of a probe fitted on n labelled training rows)",
        "",
        markdown_table(summary, order, sizes, {s: f"n={s.split('@')[1]}" for s in sizes}),
        "",
    ]
    if learned:
        cols = ["recon_num_mse", "recon_cat_acc", "masked_num_mse", "masked_cat_acc", "n_params", "best_step"]
        labels = {
            "recon_num_mse": "clean: numeric MSE",
            "recon_cat_acc": "clean: categorical acc",
            "masked_num_mse": "hidden: numeric MSE",
            "masked_cat_acc": "hidden: categorical acc",
            "n_params": "encoder params",
            "best_step": "best step",
        }
        parts += [
            "### Learned encoders: reconstruction of test rows",
            "",
            "`clean` = all cells visible. `hidden` = only the 30 % hidden cells are scored, with the rest "
            "of the row (and, for graph encoders, a graph built from the visible cells only) as input.",
            "",
            markdown_table(summary, learned, cols, labels),
            "",
        ]
        if reference:
            parts += [
                "For scale, predicting every hidden test cell by the training column mean (numeric) or "
                f"mode (categorical) gives numeric MSE {reference['masked_num_mse_column_mean']:.3f} and "
                f"categorical accuracy {reference['masked_cat_acc_column_mode']:.3f}.",
                "",
            ]
        if (metrics["metric"] == "train_seconds").any():
            parts += [
                "Training time: wall-clock seconds of the full-batch steps on the CPU, without graph "
                "construction. Median and range over seeds. The times depend on what else the machine was "
                "doing, so only large ratios mean anything.",
                "",
                timing_table(metrics, learned),
                "",
            ]

    tracked = ["probe_auc", "knn_auc", *sizes[:1]]
    explain = (
        "Mean ± standard deviation across seeds of the per-seed difference; wins = seeds in which "
        "the method is ahead."
    )
    if "mlp_ae" in present:
        parts += [
            "### Paired differences vs `mlp_ae` (per seed, same split)",
            "",
            explain,
            "",
            paired_table(metrics, order, "mlp_ae", tracked),
            "",
        ]
    if "sage_selfloop" in present and any(m != "sage_selfloop" for m in learned):
        parts += [
            "### Paired differences vs `sage_selfloop` (same layers and parameters, no neighbours)",
            "",
            explain,
            "",
            paired_table(metrics, learned, "sage_selfloop", tracked),
            "",
        ]

    if len(graph_stats):
        parts += ["### Graph diagnostics (training graph, mean over seeds)", "", *graph_table(graph_stats)]
    return "\n".join(parts)


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "torch_geometric", "matplotlib"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run_evaluation(cfg: Config, seeds: list[int] | None = None, figures: bool = True) -> pd.DataFrame:
    out = cfg.result_path
    (out / "figures").mkdir(parents=True, exist_ok=True)
    metrics = collect_metrics(cfg, seeds)
    if metrics.empty:
        raise RuntimeError(f"no embeddings under {cfg.run_path}; run the training stage first")
    graph_stats = collect_graph_stats(cfg, seeds)
    logs = collect_logs(cfg, seeds)
    metrics.to_csv(out / "metrics.csv", index=False)
    graph_stats.to_csv(out / "graph_diagnostics.csv", index=False)
    if not logs.empty:
        logs.to_csv(out / "training_logs.csv", index=False)
    (out / "summary.md").write_text(
        summary_markdown(cfg, metrics, graph_stats, collect_reference(cfg, seeds)) + "\n"
    )

    _, report = load_clean(cfg.data.raw_dir, cfg.data.files)
    info = {
        "config": cfg.name,
        "seeds": sorted(int(s) for s in metrics["seed"].unique()),
        "rows_after_cleaning": report.n_final,
        "rows_used": min(cfg.data.n_rows or report.n_final, report.n_final),
        "duplicates_removed": report.n_duplicates,
        "positive_rate_after_cleaning": round(report.positive_rate, 4),
        "versions": _versions(),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=1) + "\n")
    if figures:
        from graphmix.viz import make_figures

        make_figures(cfg, metrics, graph_stats, logs, out / "figures", seeds)
    return metrics
