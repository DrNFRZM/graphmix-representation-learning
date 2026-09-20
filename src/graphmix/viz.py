"""Figures written by the evaluation stage. Everything is drawn from the metrics tables (and, for
the embedding plot, from the saved embeddings), never from hard-coded numbers.

Colour rule: the three headline encoders keep one colour in every figure (flat MLP autoencoder =
blue, GraphSAGE on the kNN graph = orange, GraphSAGE on the record-value graph = aqua); every
other method is grey and is identified by its label. Filled markers = trained encoders, open
markers = no training. The same numbers are in summary.md, so no figure is the only place a
value can be read.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from sklearn.manifold import TSNE

from graphmix.config import NEURAL_KINDS, PROP_KINDS, Config, MethodConfig
from graphmix.evaluate import load_embedding

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
ORANGE_LIGHT = "#f6b99b"
GRAY, LIGHT_GRAY = "#7d7c77", "#b8b7b1"

HEADLINE = {"mlp_ae": BLUE, "sage_knn": ORANGE, "sage_bip": AQUA}
PROBE_PANELS = [("probe_auc", "Linear probe AUC (test)"), ("knn_auc", "kNN AUC (test)")]

GROUP_ORDER = [
    "reference (flat features)",
    "flat, no training",
    "flat, learned",
    "graph, no training",
    "graph, learned",
]


@contextmanager
def _style() -> Iterator[None]:
    with rc_context(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.titlecolor": INK2,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
        }
    ):
        yield


def _colour(name: str) -> str:
    return HEADLINE.get(name, GRAY)


class Stat(NamedTuple):
    mean: float
    std: float  # 0 when there is a single seed


def _agg(metrics: pd.DataFrame, metric: str) -> dict[str, Stat]:
    """method -> mean and std over seeds of one metric."""
    grouped = metrics[metrics["metric"] == metric].groupby("method")["value"]
    mean, std = grouped.mean(), grouped.std()
    return {str(m): Stat(float(mean[m]), 0.0 if pd.isna(std[m]) else float(std[m])) for m in mean.index}


def _err(table: dict[str, Stat], name: str) -> float:
    return table[name].std if name in table else 0.0


def _spread(ys: Sequence[float], gap: float, bounds: tuple[float, float] | None = None) -> list[float]:
    """Move label positions apart until neighbours are at least `gap` apart (data units), keeping
    the group centred on where it started and, if `bounds` is given, inside them."""
    y = np.asarray(ys, dtype=float)
    if y.size == 0:
        return []
    order = np.argsort(y)
    pos = y[order].copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= (pos - y[order]).mean()
    if bounds is not None:
        pos += max(0.0, bounds[0] - pos[0])
        pos -= max(0.0, pos[-1] - bounds[1])
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()


def _end_labels(
    ax: Axes,
    xs: Sequence[float],
    ys: Sequence[float],
    names: Sequence[str],
    x_text: float,
    gap: float,
    axes_x: bool = False,
) -> None:
    """Label the ends of lines, spread apart vertically, each joined to its line end by a thin leader.
    With `axes_x` the x positions are axes fractions (used for horizontal reference lines)."""
    transform = ax.get_yaxis_transform() if axes_x else ax.transData
    lo, hi = ax.get_ylim()
    bounds = (lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo))
    for x, y, y_text, name in zip(xs, ys, _spread(ys, gap, bounds), names, strict=True):
        ax.annotate(
            name,
            xy=(x, y),
            xycoords=transform,
            xytext=(x_text, y_text),
            textcoords=transform,
            fontsize=8.5,
            color=INK2,
            va="center",
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "color": LIGHT_GRAY,
                "linewidth": 0.8,
                "shrinkA": 1,
                "shrinkB": 1,
                "relpos": (0, 0.5),
            },
        )


def _common_limits(extents: Sequence[tuple[float, float]], min_span: float) -> list[tuple[float, float]]:
    """One shared width for several panels (each centred on its own data), at least `min_span`, so that
    differences of the size of the noise are not stretched across a whole panel."""
    span = max(min_span, 1.12 * max(hi - lo for lo, hi in extents))
    return [((lo + hi - span) / 2, (lo + hi + span) / 2) for lo, hi in extents]


def _learned(cfg: Config) -> set[str]:
    """Names of the methods that train a neural encoder."""
    return {m.name for m in cfg.methods if m.kind in NEURAL_KINDS}


def _is_main(m: MethodConfig) -> bool:
    """A method that uses the shared defaults (no k / mask-ratio / lr override, corrupted-view graph),
    i.e. neither an ablation nor a control."""
    default = m.k is None and m.mask_ratio is None and m.lr is None and m.neighbors == "visible"
    return default and m.kind != "sage_bip_nopool"


def _group(kind: str) -> str:
    if kind in ("raw", "hgb"):
        return GROUP_ORDER[0]
    if kind == "pca":
        return GROUP_ORDER[1]
    if kind == "mlp_ae":
        return GROUP_ORDER[2]
    if kind in PROP_KINDS:
        return GROUP_ORDER[3]
    return GROUP_ORDER[4]


# --------------------------------------------------------------------------- figures


def plot_main_comparison(cfg: Config, metrics: pd.DataFrame) -> Figure:
    """Dot plot of the two headline metrics for every method, grouped by what the method uses."""
    kinds = {m.name: m.kind for m in cfg.methods} | {"hgb_ref": "hgb"}
    present = set(metrics["method"])
    rows: list[tuple[str, str | None]] = []  # (label, method); method None marks a group header
    for group in GROUP_ORDER:
        members = [m for m in kinds if m in present and _group(kinds[m]) == group]
        if members:
            rows.append((group, None))
            rows.extend((m, m) for m in members)

    leaky = {m.name for m in cfg.methods if m.neighbors == "clean"}
    rows = [(f"{label} (leaky control)" if label in leaky else label, method) for label, method in rows]
    fig = Figure(figsize=(9.2, 0.3 * len(rows) + 1.4), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    ys = {label: len(rows) - i for i, (label, _) in enumerate(rows)}
    extents = []
    for ax, (metric, title) in zip(axes, PROBE_PANELS, strict=True):
        table = _agg(metrics, metric)
        low, high = np.inf, -np.inf
        for label, method in rows:
            if method is None or method not in table:
                continue
            colour = ORANGE_LIGHT if method in leaky else _colour(method)
            learned = kinds[method] in NEURAL_KINDS
            ax.errorbar(
                table[method].mean,
                ys[label],
                xerr=_err(table, method),
                fmt="D" if method in leaky else "o",
                color=colour,
                ecolor=colour,
                elinewidth=1.2,
                capsize=2.5,
                markerfacecolor=colour if learned else SURFACE,
                markeredgewidth=1.5,
            )
            low = min(low, table[method].mean - _err(table, method))
            high = max(high, table[method].mean + _err(table, method))
        extents.append((float(low), float(high)))
        ax.set_title(title, loc="left")
        ax.set_ylim(0.3, len(rows) + 0.7)
        ax.grid(axis="y", visible=False)
    for ax, (lo, hi) in zip(axes, _common_limits(extents, min_span=0.03), strict=True):
        ax.set_xlim(lo, hi)
    axes[0].set_yticks([ys[label] for label, _ in rows])
    axes[0].set_yticklabels([label for label, _ in rows])
    for tick, (_, method) in zip(axes[0].get_yticklabels(), rows, strict=True):
        if method is None:
            tick.set_fontstyle("italic")
            tick.set_color(INK)
    axes[0].tick_params(axis="y", length=0)
    fig.text(
        0.0,
        -0.02,
        "filled = trained encoder, open = no training; bars = ±1 std over seeds; axes do not start at zero, "
        "both panels have the same width",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    return fig


def plot_graph_ablation(cfg: Config, metrics: pd.DataFrame) -> Figure | None:
    """AUC of the kNN-graph encoder as k grows, against the reference encoders (horizontal lines)."""
    ks = {
        m.name: cfg.k_of(m)
        for m in cfg.methods
        if m.kind == "sage_knn" and m.mask_ratio is None and m.neighbors == "visible"
    }
    if len(ks) < 2:
        return None
    refs = [
        m.name
        for m in cfg.methods
        if m.kind in ("mlp_ae", "sage_selfloop", "sage_rand", "sage_bip") and _is_main(m)
    ]
    fig = Figure(figsize=(9.6, 3.7), layout=ConstrainedLayoutEngine(wspace=0.3))
    axes = fig.subplots(1, 2)
    panels = []
    for ax, (metric, title) in zip(axes, PROBE_PANELS, strict=True):
        table = _agg(metrics, metric)
        names = [n for n in sorted(ks, key=lambda n: ks[n]) if n in table]
        xs = [ks[n] for n in names]
        means = [table[n].mean for n in names]
        errs = [_err(table, n) for n in names]
        ax.errorbar(xs, means, yerr=errs, color=ORANGE, marker="o", capsize=2.5, elinewidth=1.2)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.minorticks_off()
        shown = [n for n in refs if n in table]
        for name in shown:
            ax.axhline(table[name].mean, color=_colour(name), linewidth=1.3)
        values = [
            *(m - e for m, e in zip(means, errs, strict=True)),
            *(m + e for m, e in zip(means, errs, strict=True)),
        ]
        values += [table[n].mean for n in shown]
        panels.append((ax, shown, table, (min(values), max(values))))
        ax.set_xlabel("neighbours per node, k")
        ax.set_title(title, loc="left")
    limits = _common_limits([extent for *_, extent in panels], min_span=0.02)
    for (ax, shown, table, _), (lo, hi) in zip(panels, limits, strict=True):
        ax.set_ylim(lo, hi)
        _end_labels(
            ax,
            [1.0] * len(shown),
            [table[n].mean for n in shown],
            shown,
            x_text=1.05,
            gap=0.07 * (hi - lo),
            axes_x=True,
        )
    fig.text(
        0.0,
        -0.03,
        "orange: sage_knn with k neighbours (bars = ±1 std over seeds); horizontal lines: the other encoders "
        "with the default settings, mean over seeds",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    return fig


def plot_label_efficiency(cfg: Config, metrics: pd.DataFrame) -> Figure | None:
    """Test AUC of the probe as the number of labelled training rows grows."""
    sizes = sorted(
        {m for m in metrics["metric"] if m.startswith("lowlabel_auc@")},
        key=lambda s: int(s.split("@")[1]),
    )
    wanted = ("raw", "pca", "mlp_ae", "sage_selfloop", "sage_rand", "sage_knn", "sage_bip")
    shown = [m.name for m in cfg.methods if m.name in wanted and m.name in set(metrics["method"])]
    if not sizes or len(shown) < 2:
        return None
    labels = [s.split("@")[1] for s in sizes] + ["all"]
    style = {  # grey methods are told apart by line style, headline encoders by colour
        "raw": (INK2, "--"),
        "pca": (GRAY, ":"),
        "sage_selfloop": (GRAY, "-"),
        "sage_rand": (LIGHT_GRAY, "-"),
    }
    fig = Figure(figsize=(6.8, 4.0), layout="constrained")
    ax = fig.subplots()
    learned = _learned(cfg)
    tables = [_agg(metrics, s) for s in [*sizes, "probe_auc"]]
    x = np.arange(len(labels))
    ends = []
    for name in shown:
        colour, dashes = style.get(name, (_colour(name), "-"))
        mean = [t[name].mean if name in t else np.nan for t in tables]
        std = [_err(t, name) for t in tables]
        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=colour,
            linestyle=dashes,
            marker="o",
            capsize=2,
            elinewidth=1,
            linewidth=1.9 if name in HEADLINE else 1.4,
            markerfacecolor=colour if name in learned else SURFACE,
            label=name,
        )
        ends.append(mean[-1])
    lo, hi = ax.get_ylim()
    _end_labels(ax, [x[-1] + 0.05] * len(shown), ends, shown, x_text=x[-1] + 0.3, gap=0.05 * (hi - lo))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.2, len(labels) - 1 + 1.3)
    ax.set_xlabel("labelled training rows used by the probe")
    ax.set_ylabel("test AUC")
    return fig


def plot_training_curves(cfg: Config, logs: pd.DataFrame) -> Figure | None:
    """Validation loss of the shared masked-reconstruction objective (mean over seeds). The
    clean-neighbour control is drawn dashed: its loss is far lower because the graph gives the
    hidden cells away, not because its representation is better."""
    present = set(logs["method"]) if not logs.empty else set()
    main = [m.name for m in cfg.methods if m.kind in NEURAL_KINDS and _is_main(m) and m.name in present]
    leaky = [m.name for m in cfg.methods if m.neighbors == "clean" and m.name in present]
    if not main:
        return None
    fig = Figure(figsize=(6.8, 3.9), layout="constrained")
    ax = fig.subplots()
    style = {"sage_selfloop": (GRAY, "-"), "sage_rand": (LIGHT_GRAY, "-")}
    ends = []
    for name in [*main, *leaky]:
        curve = logs[logs["method"] == name].groupby("step")["val_loss"].mean()
        colour, dashes = style.get(name, (_colour(name), "-"))
        if name in leaky:
            colour, dashes = ORANGE_LIGHT, "--"
        ax.plot(curve.index, curve.to_numpy(), color=colour, linestyle=dashes, linewidth=1.7, label=name)
        ends.append((float(curve.index[-1]), float(curve.to_numpy()[-1])))
    lo, hi = ax.get_ylim()
    last_x = max(x for x, _ in ends)
    _end_labels(
        ax,
        [x for x, _ in ends],
        [y for _, y in ends],
        [*main, *leaky],
        x_text=last_x * 1.04,
        gap=0.055 * (hi - lo),
    )
    ax.set_xlim(0, last_x * 1.32)
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= last_x])
    ax.set_xlabel("full-batch step")
    ax.set_ylabel("validation loss (corrupted val. rows)")
    return fig


def _bars(ax: Axes, labels: list[str], values: list[float], colours: list[str], chance: float) -> None:
    ax.bar(range(len(labels)), values, color=colours, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(chance, color=INK2, linewidth=1)
    ax.text(1.01, chance, "chance", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="x", visible=False)


GRAPH_NAME = re.compile(r"^(knn_visible|knn|random)_k(\d+)(?:_p([0-9.]+))?$")
# (graph kind, legend text, colour), in drawing order
GRAPH_SERIES = [
    ("knn", "kNN, neighbours from complete rows", ORANGE),
    ("knn_visible", "kNN, neighbours from visible cells (what training uses)", ORANGE_LIGHT),
    ("random", "random, same degree as kNN", LIGHT_GRAY),
]


def _parse_graph(name: str) -> tuple[str, int]:
    match = GRAPH_NAME.match(name)
    assert match is not None
    return match.group(1), int(match.group(2))


def plot_homophily(graph_stats: pd.DataFrame) -> Figure | None:
    """How well neighbours agree on the label: kNN vs random graphs, and per bipartite column."""
    if graph_stats.empty:
        return None
    mean = graph_stats.groupby(["graph", "metric"])["value"].mean()
    graphs = {g for g, _ in mean.index}
    parsed = {g: _parse_graph(g) for g in graphs if GRAPH_NAME.match(g)}
    cols = sorted({m for _, m in mean.index if m.startswith("agreement[")})
    if not parsed and not cols:
        return None

    chance = float(graph_stats.loc[graph_stats["metric"] == "label_agreement_by_chance", "value"].mean())
    fig = Figure(figsize=(9.6, 3.8), layout=ConstrainedLayoutEngine(wspace=0.12))
    axes = fig.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1, 1.4]})
    if parsed:
        ks = sorted({k for _, k in parsed.values()})
        by_kind = {(kind, k): g for g, (kind, k) in parsed.items()}
        width = 0.27
        for j, (kind, text, colour) in enumerate(GRAPH_SERIES):
            present = [(i, by_kind[(kind, k)]) for i, k in enumerate(ks) if (kind, k) in by_kind]
            if present:
                axes[0].bar(
                    [i + (j - 1) * (width + 0.02) for i, _ in present],
                    [float(mean[(g, "edge_homophily")]) for _, g in present],
                    width=width,
                    color=colour,
                    label=text,
                )
        axes[0].set_xticks(range(len(ks)))
        axes[0].set_xticklabels([f"k={k}" for k in ks])
        axes[0].axhline(chance, color=INK2, linewidth=1)
        axes[0].text(
            1.01,
            chance,
            "chance",
            transform=axes[0].get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=INK2,
        )
        axes[0].set_ylim(0.0, 1.0)
        axes[0].grid(axis="x", visible=False)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, ncol=1)
    axes[0].set_title("Edge homophily w.r.t. income (training graph)", loc="left")
    if cols:
        _bars(
            axes[1],
            [c[len("agreement[") : -1].replace("-", "\n") for c in cols],
            [float(mean[("bipartite", c)]) for c in cols],
            [AQUA] * len(cols),
            chance,
        )
    axes[1].set_title("Label agreement inside value groups, per column", loc="left")
    return fig


def plot_embeddings(cfg: Config, seed: int, methods: list[str], n_points: int) -> Figure | None:
    """t-SNE of test-row embeddings (same rows in every panel), grey = <=50K, violet = >50K."""
    loaded = {}
    for name in methods:
        path = cfg.run_path / f"seed{seed}" / f"{name}.npz"
        if path.exists():
            loaded[name] = load_embedding(path)[0]
    if not loaded:
        return None
    first = next(iter(loaded.values()))
    rng = np.random.default_rng(seed)
    n_test = len(first.y_test)
    idx = np.sort(rng.choice(n_test, size=min(n_points, n_test), replace=False))
    y = first.y_test[idx]

    cols = min(len(loaded), 4)
    rows = int(np.ceil(len(loaded) / cols))
    fig = Figure(figsize=(3.1 * cols, 3.1 * rows), layout="constrained")
    axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False)).ravel()
    for ax, (name, emb) in zip(axes, loaded.items(), strict=False):
        mu = emb.z_train.mean(axis=0)
        sd = emb.z_train.std(axis=0)
        z = ((emb.z_test[idx] - mu) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float64)
        perplexity = min(30.0, max(5.0, (len(idx) - 1) / 4))
        xy = TSNE(
            n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed
        ).fit_transform(z)
        ax.scatter(
            xy[y == 0, 0], xy[y == 0, 1], s=6, color=LIGHT_GRAY, alpha=0.55, linewidths=0, label="<=50K"
        )
        ax.scatter(xy[y == 1, 0], xy[y == 1, 1], s=7, color=VIOLET, alpha=0.7, linewidths=0, label=">50K")
        ax.set_title(name, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    for ax in axes[len(loaded) :]:
        ax.set_visible(False)
    axes[0].legend(loc="lower left", fontsize=8, markerscale=2)
    return fig


# --------------------------------------------------------------------------- driver


def make_figures(
    cfg: Config,
    metrics: pd.DataFrame,
    graph_stats: pd.DataFrame,
    logs: pd.DataFrame,
    out_dir: Path,
    seeds: list[int] | None = None,
) -> list[Path]:
    """Write every figure that the available results support; returns the files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {m.name for m in cfg.methods}
    tsne_methods: list[str] = [m for m in ("pca", "mlp_ae", "sage_knn", "sage_bip") if m in names]
    seed = (seeds or cfg.seeds)[0]
    jobs: list[tuple[str, Callable[[], Figure | None]]] = [
        ("fig_main_comparison.png", lambda: plot_main_comparison(cfg, metrics)),
        ("fig_graph_ablation.png", lambda: plot_graph_ablation(cfg, metrics)),
        ("fig_label_efficiency.png", lambda: plot_label_efficiency(cfg, metrics)),
        ("fig_training_curves.png", lambda: plot_training_curves(cfg, logs)),
        ("fig_homophily.png", lambda: plot_homophily(graph_stats)),
        ("fig_embeddings.png", lambda: plot_embeddings(cfg, seed, tsne_methods, cfg.evaluation.tsne_points)),
    ]
    written: list[Path] = []
    with _style():
        for filename, build in jobs:
            fig = build()
            if fig is not None:
                fig.savefig(out_dir / filename, dpi=160, bbox_inches="tight")
                written.append(out_dir / filename)
    return written
