from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphmix.config import EvalConfig
from graphmix.evaluate import (
    Embedding,
    evaluate_embedding,
    frame_to_markdown,
    hgb_reference,
    label_efficiency,
    linear_probe,
    load_embedding,
    markdown_table,
    paired_delta,
    retrieval,
    save_embedding,
    summarize,
)


def make_embedding(
    seed: int = 0, signal: float = 2.0, sizes: tuple[int, int, int] = (300, 100, 100)
) -> Embedding:
    """Four-dimensional embedding whose first coordinate carries the label."""
    rng = np.random.default_rng(seed)

    def part(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = (rng.random(n) < 0.3).astype(np.int64)
        z = rng.normal(size=(n, 4))
        z[:, 0] += signal * y
        return z, y

    (ztr, ytr), (zva, yva), (zte, yte) = (part(n) for n in sizes)
    return Embedding(ztr, zva, zte, ytr, yva, yte)


def test_embedding_roundtrip(tmp_path: Path) -> None:
    emb = make_embedding()
    save_embedding(tmp_path / "deep" / "e.npz", emb, {"method": "x", "k": 5})
    back, meta = load_embedding(tmp_path / "deep" / "e.npz")
    assert meta == {"method": "x", "k": 5}
    assert np.array_equal(back.z_test, emb.z_test.astype(np.float32))
    assert np.array_equal(back.y_train, emb.y_train)


def test_linear_probe_finds_the_signal_and_picks_c_from_the_grid() -> None:
    out = linear_probe(make_embedding(), [0.01, 1.0, 100.0])
    assert out["probe_auc"] > 0.85 and out["probe_C"] in (0.01, 1.0, 100.0)
    assert 0.0 <= out["probe_acc"] <= 1.0 and out["probe_ap"] > 0.3


def test_probe_uses_validation_labels_for_c_and_never_test_labels() -> None:
    emb = make_embedding()
    scrambled = replace(emb, y_test=np.random.default_rng(1).permutation(emb.y_test))
    grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    assert linear_probe(emb, grid)["probe_C"] == linear_probe(scrambled, grid)["probe_C"]


def test_probe_is_invariant_to_rescaling_the_embedding() -> None:
    emb = make_embedding()
    big = Embedding(
        emb.z_train * 1000 + 7,
        emb.z_val * 1000 + 7,
        emb.z_test * 1000 + 7,
        emb.y_train,
        emb.y_val,
        emb.y_test,
    )
    a, b = linear_probe(emb, [1.0]), linear_probe(big, [1.0])
    assert a["probe_auc"] == pytest.approx(b["probe_auc"], abs=1e-6)


def test_label_efficiency_is_seeded_and_clips_the_sample_size() -> None:
    emb = make_embedding()
    a = label_efficiency(emb, [20, 10_000], repeats=3, seed=1)
    assert set(a) == {"lowlabel_auc@20", "lowlabel_auc@300"}  # 10_000 clipped to the 300 training rows
    assert a == label_efficiency(emb, [20, 10_000], repeats=3, seed=1)
    assert a != label_efficiency(emb, [20, 10_000], repeats=3, seed=2)
    assert a["lowlabel_auc@300"] > a["lowlabel_auc@20"] - 0.05  # more labels do not hurt


def test_label_efficiency_returns_nan_when_a_probe_cannot_be_fitted() -> None:
    emb = make_embedding()
    one_class = replace(emb, y_train=np.zeros_like(emb.y_train))
    assert np.isnan(label_efficiency(one_class, [30], repeats=2, seed=0)["lowlabel_auc@30"])


def test_retrieval_is_perfect_on_separated_clusters() -> None:
    rng = np.random.default_rng(0)

    def part(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = (np.arange(n) % 2).astype(np.int64)
        z = rng.normal(scale=0.1, size=(n, 3))
        z[:, 0] += np.where(y == 1, 5.0, -5.0)
        return z, y

    (ztr, ytr), (zva, yva), (zte, yte) = part(100), part(20), part(40)
    out = retrieval(Embedding(ztr, zva, zte, ytr, yva, yte), k=15)
    assert out["knn_auc"] == pytest.approx(1.0) and out["knn_pos_p10"] == pytest.approx(1.0)


def test_retrieval_of_noise_is_near_chance() -> None:
    emb = make_embedding(signal=0.0, sizes=(600, 100, 600))
    assert abs(retrieval(emb, k=25)["knn_auc"] - 0.5) < 0.06


def test_hgb_reference_beats_chance() -> None:
    assert hgb_reference(make_embedding(), seed=0)["probe_auc"] > 0.85


def test_evaluate_embedding_reports_every_metric() -> None:
    cfg = EvalConfig(c_grid=[0.1, 1.0], label_sizes=[20, 50], label_repeats=2, knn_k=10)
    out = evaluate_embedding(make_embedding(), cfg, seed=0)
    expected = {"probe_auc", "probe_ap", "probe_acc", "probe_C", "lowlabel_auc@20", "lowlabel_auc@50"}
    assert expected | {"knn_auc", "knn_pos_p10", "base_rate"} == set(out)


# --------------------------------------------------------------------------- aggregation


def long_metrics() -> pd.DataFrame:
    rows = []
    for seed, (a, b) in enumerate([(0.80, 0.78), (0.82, 0.83), (0.84, 0.80)]):
        rows += [(seed, "a", "probe_auc", a), (seed, "b", "probe_auc", b), (seed, "a", "n_params", 1234.0)]
    return pd.DataFrame(rows, columns=["seed", "method", "metric", "value"])


def test_summarize_and_paired_delta() -> None:
    summary = summarize(long_metrics()).set_index(["method", "metric"])
    assert summary.loc[("a", "probe_auc"), "mean"] == pytest.approx(0.82)
    assert summary.loc[("a", "probe_auc"), "n"] == 3
    d = paired_delta(long_metrics(), "a", "b", "probe_auc")
    assert (
        d["n"] == 3 and d["wins"] == 2 and d["mean"] == pytest.approx(0.05 / 3)
    )  # differences 0.02, -0.01, 0.04
    missing = paired_delta(long_metrics(), "a", "zzz", "probe_auc")
    assert missing["n"] == 0 and np.isnan(missing["mean"])


def test_markdown_table_formatting() -> None:
    summary = summarize(long_metrics())
    table = markdown_table(summary, ["a", "b"], ["probe_auc", "n_params"], {"probe_auc": "AUC"}).splitlines()
    assert table[0] == "| method | AUC | n_params |"
    assert table[1] == "|---|---:|---:|"
    assert table[2].startswith("| a | 0.820 ± 0.020 | 1234 |")  # a spread of exactly zero is left out
    assert table[3].endswith("| n/a |")  # method b has no n_params
    single = markdown_table(summarize(long_metrics().query("seed == 0")), ["a"], ["probe_auc"])
    assert "±" not in single


def test_frame_to_markdown() -> None:
    df = pd.DataFrame({"x": [0.5, np.nan]}, index=pd.Index(["r1", "r2"], name="graph"))
    assert frame_to_markdown(df).splitlines() == [
        "| graph | x |",
        "|---|---:|",
        "| r1 | 0.500 |",
        "| r2 | n/a |",
    ]
    wide = pd.DataFrame({"x": [0.12345], "seconds": [12.3456]}, index=pd.Index(["r"], name="graph"))
    assert frame_to_markdown(wide, {"seconds": 1}).splitlines()[-1] == "| r | 0.123 | 12.3 |"
