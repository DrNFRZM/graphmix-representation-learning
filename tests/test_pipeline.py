"""End-to-end runs of the two stages on small synthetic files in the UCI layout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphmix.config import (
    Config,
    DataConfig,
    EvalConfig,
    GraphConfig,
    MethodConfig,
    ModelConfig,
    TrainConfig,
)
from graphmix.data import LABEL, load_clean, make_splits
from graphmix.evaluate import load_embedding
from graphmix.graphs import graph_from_neighbors, knn_neighbors
from graphmix.pipeline import (
    GraphFactory,
    collect_metrics,
    prepare_split,
    run_evaluation,
    run_training,
)

NO_TORCH = [
    MethodConfig("raw", "raw"),
    MethodConfig("pca", "pca"),
    MethodConfig("prop_knn", "prop_knn"),
    MethodConfig("prop_bip", "prop_bip"),
    MethodConfig("prop_rand", "prop_rand"),
]
NEURAL = [
    MethodConfig("mlp_ae", "mlp_ae"),
    MethodConfig("sage_selfloop", "sage_selfloop"),
    MethodConfig("sage_rand", "sage_rand"),
    MethodConfig("sage_knn", "sage_knn"),
    MethodConfig("sage_bip", "sage_bip"),
    MethodConfig("sage_bip_nopool", "sage_bip_nopool"),
    MethodConfig("sage_knn_k3", "sage_knn", k=3),
    MethodConfig("sage_knn_clean", "sage_knn", neighbors="clean"),
    MethodConfig("mlp_ae_nomask", "mlp_ae", mask_ratio=0.0),
    MethodConfig("mlp_ae_lr2x", "mlp_ae", lr=0.006),
    MethodConfig("sage_selfloop_nomask", "sage_selfloop", mask_ratio=0.0),
    MethodConfig("sage_rand_nomask", "sage_rand", mask_ratio=0.0),
]


def make_cfg(raw_dir: Path, out: Path, methods: list[MethodConfig], seeds: list[int] | None = None) -> Config:
    return Config(
        name="t",
        seeds=seeds or [0, 1],
        methods=methods,
        runs_dir=str(out / "runs"),
        results_dir=str(out / "results"),
        data=DataConfig(raw_dir=str(raw_dir)),
        graph=GraphConfig(k=5),
        model=ModelConfig(hidden=16, dim=4),
        train=TrainConfig(steps=6, eval_every=3),
        evaluation=EvalConfig(
            c_grid=[0.1, 1.0], label_sizes=[20, 50], label_repeats=2, knn_k=5, tsne_points=80
        ),
    )


# --------------------------------------------------------------------------- split and graphs


def test_split_is_ordered_seeded_and_fitted_on_training_rows(raw_dir: Path, tmp_path: Path) -> None:
    cfg = make_cfg(raw_dir, tmp_path, NO_TORCH)
    df, _ = load_clean(raw_dir)
    sd = prepare_split(df, cfg, seed=0)
    tr, va, te = make_splits(df[LABEL].to_numpy(), cfg.data.val_frac, cfg.data.test_frac, seed=0)
    assert (sd.n_train, sd.n_val, sd.n_test) == (len(tr), len(va), len(te))
    assert np.array_equal(sd.y, df[LABEL].to_numpy()[np.concatenate([tr, va, te])])
    # numeric columns are standardised with training rows only: train ~ N(0,1), the others are not
    assert np.allclose(sd.x_num[: sd.n_train].mean(axis=0), 0, atol=1e-4)
    assert not np.allclose(sd.x_num[sd.n_train :].mean(axis=0), 0, atol=1e-4)
    assert sd.flat.shape == (sd.n_total, sd.x_num.shape[1] + sum(sd.cards))
    assert np.array_equal(prepare_split(df, cfg, seed=0).x_num, sd.x_num)
    assert not np.array_equal(prepare_split(df, cfg, seed=1).x_num, sd.x_num)
    emb = sd.embedding(np.arange(sd.n_total)[:, None].astype(float))
    assert len(emb.z_train) == sd.n_train and len(emb.y_test) == sd.n_test


def test_graph_factory_reuses_the_neighbour_table(raw_dir: Path, tmp_path: Path) -> None:
    methods = [
        MethodConfig("a", "sage_knn", k=3),
        MethodConfig("b", "prop_knn", k=8),
        MethodConfig("c", "raw"),
    ]
    cfg = make_cfg(raw_dir, tmp_path, methods)
    sd = prepare_split(load_clean(raw_dir)[0], cfg, seed=0)
    factory = GraphFactory(sd, cfg)
    assert factory.k_max == 8
    g3 = factory.get("knn", 3)
    direct = graph_from_neighbors(knn_neighbors(sd.dist, sd.n_train, 3), sd.n_train)
    assert g3.edge_index is not None and direct.edge_index is not None
    assert np.array_equal(g3.edge_index, direct.edge_index)
    assert factory.get("knn", 3) is g3 and factory.get("knn", 8) is not g3
    assert factory.get("random", 4) is not factory.get("random", 5)
    assert factory.get("bipartite", 3) is factory.get("bipartite", 9)  # k is irrelevant there
    with pytest.raises(ValueError, match="unknown graph kind"):
        factory.get("hypergraph", 3)


# --------------------------------------------------------------------------- stage 1 and 2


def test_training_stage_writes_skips_and_overwrites(raw_dir: Path, tmp_path: Path) -> None:
    cfg = make_cfg(raw_dir, tmp_path, NO_TORCH)
    run_training(cfg, verbose=False)
    for seed in (0, 1):
        folder = cfg.run_path / f"seed{seed}"
        assert {p.name for p in folder.iterdir()} == {f"{m.name}.npz" for m in NO_TORCH} | {
            "graph_stats.json",
            "reference.json",
        }
    target = cfg.run_path / "seed0" / "pca.npz"
    stamp = target.stat().st_mtime_ns
    run_training(cfg, seeds=[0], verbose=False)
    assert target.stat().st_mtime_ns == stamp  # existing files are left alone
    run_training(cfg, seeds=[0], only=["pca"], overwrite=True, verbose=False)
    assert target.stat().st_mtime_ns > stamp
    stats = cfg.run_path / "seed0" / "graph_stats.json"
    assert (
        stats.stat().st_mtime_ns < target.stat().st_mtime_ns
    )  # a partial re-run leaves the diagnostics alone
    stats.unlink()
    run_training(cfg, seeds=[0], only=["pca"], verbose=False)
    assert stats.exists()  # ... unless they are missing
    emb, meta = load_embedding(target)
    assert meta["method"] == "pca" and meta["seed"] == 0 and emb.z_train.shape[1] == 4
    with pytest.raises(ValueError, match="--only"):
        run_training(cfg, only=["nonsense"], verbose=False)


def test_graph_diagnostics_are_written(raw_dir: Path, tmp_path: Path) -> None:
    cfg = make_cfg(raw_dir, tmp_path, NO_TORCH, seeds=[0])
    run_training(cfg, verbose=False)
    stats = pd.DataFrame(json.loads((cfg.run_path / "seed0" / "graph_stats.json").read_text()))
    assert {"knn_k5", "random_k5", "bipartite", "chance"} == set(stats["graph"])
    value = stats.set_index(["graph", "metric"])["value"]
    assert value[("knn_k5", "edge_homophily")] > value[("random_k5", "edge_homophily")]
    assert value[("knn_k5", "build_seconds")] >= 0
    assert 0.5 < value[("chance", "label_agreement_by_chance")] < 1.0
    assert any(m.startswith("agreement[") for _, m in value.index)


def test_evaluation_stage_produces_tables_and_figures(raw_dir: Path, tmp_path: Path) -> None:
    cfg = make_cfg(raw_dir, tmp_path, NO_TORCH)
    run_training(cfg, verbose=False)
    metrics = run_evaluation(cfg)
    out = cfg.result_path
    assert {"metrics.csv", "summary.md", "graph_diagnostics.csv", "run_info.json", "figures"} == {
        p.name for p in out.iterdir()
    }
    assert set(metrics["method"]) == {m.name for m in NO_TORCH} | {"hgb_ref"}
    auc = metrics[(metrics["metric"] == "probe_auc") & (metrics["method"] == "raw")]["value"]
    assert len(auc) == 2 and (auc > 0.7).all()  # the synthetic label is learnable from the features
    summary = (out / "summary.md").read_text()
    assert "Frozen-embedding evaluation" in summary and "hgb_ref" in summary
    assert "Paired differences" not in summary  # no mlp_ae in this config
    assert "adjusted homophily" in summary and "expected by chance" in summary  # kNN rows + record-value line
    assert "Training time" not in summary  # nothing was trained
    info = json.loads((out / "run_info.json").read_text())
    assert info["seeds"] == [0, 1] and "torch" in info["versions"]
    figures = {p.name for p in (out / "figures").iterdir()}
    assert {"fig_main_comparison.png", "fig_label_efficiency.png", "fig_homophily.png"} <= figures
    assert "fig_graph_ablation.png" not in figures and "fig_training_curves.png" not in figures
    assert all((out / "figures" / f).stat().st_size > 5_000 for f in figures)


def test_evaluation_without_embeddings_is_an_error(raw_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="training stage"):
        run_evaluation(make_cfg(raw_dir, tmp_path, NO_TORCH))


def test_collect_metrics_skips_missing_files(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = make_cfg(raw_dir, tmp_path, NO_TORCH, seeds=[0])
    run_training(cfg, only=["raw", "pca"], verbose=False)
    metrics = collect_metrics(cfg)
    assert set(metrics["method"]) == {"raw", "pca", "hgb_ref"}
    assert "missing" in capsys.readouterr().out


# --------------------------------------------------------------------------- with neural encoders


def test_graph_factory_builds_the_pretext_of_each_neural_method(raw_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    cfg = make_cfg(raw_dir, tmp_path, NEURAL)
    cfg.train.n_views = 3
    sd = prepare_split(load_clean(raw_dir)[0], cfg, seed=0)
    factory = GraphFactory(sd, cfg)
    by_name = {m.name: m for m in NEURAL}

    mlp = factory.pretext(by_name["mlp_ae"])
    assert len(mlp.train) == 3 and mlp.full is None and all(v.graph is None for v in mlp.train)
    assert factory.visible == set()

    knn = factory.pretext(by_name["sage_knn_k3"])
    assert len(knn.train) == 3 and all(
        v.graph is not None and v.graph.n_total == sd.n_train for v in knn.train
    )
    assert knn.full is factory.get("knn", 3)
    assert factory.visible == {(0.3, 3)}

    clean = factory.pretext(by_name["sage_knn_clean"])  # the leaky control: no corrupted searches
    assert clean.val.graph is clean.full and (0.3, 5) not in factory.visible

    nomask = factory.pretext(by_name["mlp_ae_nomask"])
    assert len(nomask.train) == 1 and not nomask.train[0].m_num.any()
    assert nomask.test.m_num.any()  # ... but every method is scored on the same hidden test cells
    for name in ("sage_selfloop_nomask", "sage_rand_nomask"):  # graph controls without corruption
        plain = factory.pretext(by_name[name])
        assert len(plain.train) == 1 and plain.train[0].graph is not None and not plain.train[0].m_num.any()
        assert plain.test.graph is plain.full and plain.test.m_num.any()


def test_resumed_run_still_reports_every_graph(raw_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    methods = [MethodConfig("sage_knn_k3", "sage_knn", k=3), MethodConfig("prop_bip", "prop_bip")]
    cfg = make_cfg(raw_dir, tmp_path, methods, seeds=[0])
    cfg.train.n_views = 2
    run_training(cfg, verbose=False)
    path = cfg.run_path / "seed0" / "graph_stats.json"
    first = {(r["graph"], r["metric"]) for r in json.loads(path.read_text())}
    assert ("knn_visible_k3_p0.3", "edge_homophily") in first and ("bipartite", "agreement_mean") in first
    path.unlink()
    run_training(cfg, verbose=False)  # every embedding exists, so nothing is retrained
    assert {(r["graph"], r["metric"]) for r in json.loads(path.read_text())} == first


def test_full_pipeline_with_neural_encoders(raw_dir: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch_geometric")
    cfg = make_cfg(raw_dir, tmp_path, [*NO_TORCH, *NEURAL], seeds=[0, 1])
    run_training(cfg, verbose=False)
    folder = cfg.run_path / "seed0"
    for m in NEURAL:
        assert (folder / f"{m.name}.npz").exists() and (folder / f"{m.name}.log.csv").exists()
    _, meta = load_embedding(folder / "sage_knn_k3.npz")
    assert meta["k"] == 3 and meta["mask_ratio"] == 0.3 and meta["n_params"] > 0
    assert meta["neighbors"] == "visible" and 0 <= meta["masked_cat_acc"] <= 1
    assert load_embedding(folder / "sage_knn_clean.npz")[1]["neighbors"] == "clean"
    assert load_embedding(folder / "mlp_ae_nomask.npz")[1]["mask_ratio"] == 0.0
    assert load_embedding(folder / "mlp_ae_lr2x.npz")[1]["lr"] == 0.006  # per-method override ...
    assert load_embedding(folder / "mlp_ae.npz")[1]["lr"] == cfg.train.lr  # ... and the shared default

    reference = json.loads((folder / "reference.json").read_text())
    assert reference["masked_num_mse_column_mean"] > 0 and 0 < reference["masked_cat_acc_column_mode"] < 1
    stats = pd.DataFrame(json.loads((folder / "graph_stats.json").read_text()))
    assert {"knn_visible_k3_p0.3", "knn_visible_k5_p0.3"} <= set(stats["graph"])
    value = stats.set_index(["graph", "metric"])["value"]
    assert value[("knn_visible_k5_p0.3", "build_seconds")] > 0

    metrics = run_evaluation(cfg)
    assert {
        "recon_num_mse",
        "recon_cat_acc",
        "masked_num_mse",
        "masked_cat_acc",
        "n_params",
        "best_step",
        "train_seconds",
    } <= set(metrics["metric"])
    summary = (cfg.result_path / "summary.md").read_text()
    assert "Paired differences vs `mlp_ae`" in summary and "reconstruction of test rows" in summary
    assert "Paired differences vs `sage_selfloop`" in summary
    assert "For scale, predicting every hidden test cell" in summary
    assert "Training time" in summary and "| method | median s | range s |" in summary
    assert "| 4 ± 0 |" not in summary  # the width of the code is constant and is printed without a spread
    logs = pd.read_csv(cfg.result_path / "training_logs.csv")
    assert list(logs.columns) == ["seed", "method", "step", "train_loss", "val_loss"]
    assert set(logs["method"]) == {m.name for m in NEURAL} and set(logs["seed"]) == {0, 1}
    figures = {p.name for p in (cfg.result_path / "figures").iterdir()}
    assert figures == {
        "fig_main_comparison.png",
        "fig_graph_ablation.png",
        "fig_label_efficiency.png",
        "fig_training_curves.png",
        "fig_homophily.png",
        "fig_embeddings.png",
    }
