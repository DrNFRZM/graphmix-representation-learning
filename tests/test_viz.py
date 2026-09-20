from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphmix.config import Config, MethodConfig
from graphmix.viz import (
    _common_limits,
    _spread,
    make_figures,
    plot_graph_ablation,
    plot_homophily,
    plot_label_efficiency,
    plot_main_comparison,
    plot_training_curves,
)


def test_spread_separates_labels_without_reordering_them() -> None:
    ys = [0.50, 0.501, 0.502, 0.90, 0.10]
    out = _spread(ys, gap=0.05)
    order = np.argsort(ys)
    moved = np.array(out)[order]
    assert np.all(np.diff(moved) >= 0.05 - 1e-12)  # neighbours at least `gap` apart
    assert out[3] > out[0] and out[4] < out[0]  # the order of the labels is unchanged
    assert np.mean(out) == pytest.approx(np.mean(ys))  # the group stays centred where it was
    assert _spread([0.3], gap=0.1) == [0.3]
    assert _spread([], gap=0.1) == []


def test_spread_can_be_kept_inside_the_axes() -> None:
    low = _spread([0.01, 0.02, 0.03], gap=0.1, bounds=(0.0, 1.0))
    assert min(low) >= 0.0 and np.all(np.diff(sorted(low)) >= 0.1 - 1e-12)
    high = _spread([0.98, 0.99, 1.0], gap=0.1, bounds=(0.0, 1.0))
    assert max(high) <= 1.0 and np.all(np.diff(sorted(high)) >= 0.1 - 1e-12)


def test_common_limits_share_one_width_and_do_not_zoom_into_noise() -> None:
    (a_lo, a_hi), (b_lo, b_hi) = _common_limits([(0.880, 0.882), (0.850, 0.890)], min_span=0.03)
    assert a_hi - a_lo == pytest.approx(b_hi - b_lo)
    assert b_hi - b_lo > 0.04  # the wider panel decides, with some padding
    assert (a_lo + a_hi) / 2 == pytest.approx(0.881) and (b_lo + b_hi) / 2 == pytest.approx(0.87)
    ((lo, hi),) = _common_limits([(0.880, 0.882)], min_span=0.03)
    assert hi - lo == pytest.approx(0.03)  # a tiny range is not stretched over the whole panel


def fake_results() -> tuple[Config, pd.DataFrame]:
    methods = [
        MethodConfig("raw", "raw"),
        MethodConfig("mlp_ae", "mlp_ae"),
        MethodConfig("sage_knn", "sage_knn"),
        MethodConfig("sage_knn_k5", "sage_knn", k=5),
    ]
    cfg = Config(name="fake", seeds=[0, 1], methods=methods)
    rng = np.random.default_rng(0)
    rows = []
    for seed in (0, 1):
        for i, m in enumerate(["raw", "mlp_ae", "sage_knn", "sage_knn_k5"]):
            for metric in ("probe_auc", "knn_auc", "lowlabel_auc@50"):
                rows.append((seed, m, metric, 0.8 + 0.01 * i + rng.normal(0, 0.002)))
    return cfg, pd.DataFrame(rows, columns=["seed", "method", "metric", "value"])


def test_figures_are_only_made_when_the_results_support_them() -> None:
    cfg, metrics = fake_results()
    assert plot_main_comparison(cfg, metrics) is not None
    assert plot_graph_ablation(cfg, metrics) is not None  # two sage_knn variants with different k
    assert plot_label_efficiency(cfg, metrics) is not None
    no_logs = pd.DataFrame(columns=["seed", "method", "step", "train_loss", "val_loss"])
    assert plot_training_curves(cfg, no_logs) is None
    only_one_knn = Config(name="x", seeds=[0], methods=cfg.methods[:3])
    assert plot_graph_ablation(only_one_knn, metrics) is None
    assert plot_label_efficiency(cfg, metrics[metrics["metric"] != "lowlabel_auc@50"]) is None


def test_the_leaky_control_is_marked_in_the_main_comparison() -> None:
    cfg, metrics = fake_results()
    control = MethodConfig("sage_knn_cleannbr", "sage_knn", neighbors="clean")
    cfg.methods.append(control)
    extra = metrics[metrics["method"] == "sage_knn"].assign(method="sage_knn_cleannbr")
    fig = plot_main_comparison(cfg, pd.concat([metrics, extra], ignore_index=True))
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "sage_knn_cleannbr (leaky control)" in labels and "sage_knn" in labels


def test_homophily_figure_needs_only_the_series_that_exist() -> None:
    rows = [
        (0, "knn_k10", "edge_homophily", 0.78),
        (0, "knn_visible_k10_p0.3", "edge_homophily", 0.71),
        (0, "chance", "label_agreement_by_chance", 0.63),
    ]
    stats = pd.DataFrame(rows, columns=["seed", "graph", "metric", "value"])
    fig = plot_homophily(stats)
    assert fig is not None
    assert len(fig.axes[0].patches) == 2  # two bars, no random graph in this table
    assert plot_homophily(stats.iloc[0:0]) is None


def test_make_figures_writes_png_files(tmp_path: Path) -> None:
    cfg, metrics = fake_results()
    logs = pd.DataFrame(
        [(s, "mlp_ae", step, 1.0 / step, 1.0 / step) for s in (0, 1) for step in (10, 20, 30)],
        columns=["seed", "method", "step", "train_loss", "val_loss"],
    )
    stats = pd.DataFrame(
        [
            (0, "knn_k10", "edge_homophily", 0.78),
            (0, "random_k10", "edge_homophily", 0.63),
            (0, "bipartite", "agreement[sex]", 0.66),
            (0, "chance", "label_agreement_by_chance", 0.63),
        ],
        columns=["seed", "graph", "metric", "value"],
    )
    written = make_figures(cfg, metrics, stats, logs, tmp_path / "figs", seeds=[0])
    names = {p.name for p in written}
    assert {"fig_main_comparison.png", "fig_graph_ablation.png", "fig_label_efficiency.png"} <= names
    assert {"fig_training_curves.png", "fig_homophily.png"} <= names
    assert "fig_embeddings.png" not in names  # needs saved embeddings, which this test does not have
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in written)
