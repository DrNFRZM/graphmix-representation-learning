import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import LEVELS, make_frame

from graphmix import data
from graphmix.data import (
    CAT_COLS,
    LABEL,
    NUM_COLS,
    RARE,
    UNKNOWN,
    Preprocessor,
    clean,
    download_adult,
    flat_features,
    load_clean,
    make_splits,
    read_adult_file,
    subsample,
    verify_raw,
)

REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "adult"


# --------------------------------------------------------------------------- parsing / cleaning


def test_read_adult_data_and_test_layouts(raw_dir: Path) -> None:
    train = read_adult_file(raw_dir / "adult.data")
    test = read_adult_file(raw_dir / "adult.test")
    assert len(train) == 500 and len(test) == 250  # header line and trailing blank line skipped
    assert set(train[LABEL]) == {"<=50K", ">50K"}
    assert set(test[LABEL]) == {"<=50K", ">50K"}  # trailing periods stripped
    assert train["occupation"].isna().any()  # "?" became NaN
    assert set(train["workclass"].dropna()) <= set(LEVELS["workclass"][0])


def test_clean_turns_question_marks_into_a_level_and_binarises_label(raw_dir: Path) -> None:
    raw = pd.concat([read_adult_file(raw_dir / "adult.data"), read_adult_file(raw_dir / "adult.test")])
    n_missing = int(raw["occupation"].isna().sum())
    df, report = clean(raw.reset_index(drop=True))
    assert list(df.columns) == [*NUM_COLS, *CAT_COLS, LABEL]
    assert (df["occupation"] == UNKNOWN).sum() >= 1
    assert not df.isna().any().any()
    assert set(df[LABEL].unique()) <= {0, 1}
    assert report.missing["occupation"] == n_missing
    assert report.n_raw == len(raw)
    assert report.n_final == len(df) == report.n_raw - report.n_duplicates
    assert report.positive_rate == pytest.approx(df[LABEL].mean())


def test_clean_removes_exact_duplicates_only() -> None:
    base = make_frame(50, seed=3)
    raw = base.copy()
    raw["fnlwgt"] = np.arange(50)  # differs per row, but is dropped by clean()
    raw["education"] = "HS-grad"
    raw[LABEL] = np.where(base[LABEL] == 1, ">50K", "<=50K")
    doubled = pd.concat([raw, raw.iloc[:10]], ignore_index=True)
    df, report = clean(doubled)
    n_unique = len(base.drop_duplicates())  # chance duplicates inside `base` would be removed too
    assert len(df) == n_unique and report.n_duplicates == len(doubled) - n_unique


def test_clean_rejects_unknown_labels() -> None:
    raw = make_frame(5)
    raw[LABEL] = "maybe"
    with pytest.raises(ValueError, match="unexpected income labels"):
        clean(raw)


def test_verify_raw_checks_row_counts(raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="expected 32561"):
        verify_raw(raw_dir)
    monkeypatch.setattr(data, "EXPECTED_ROWS", {"adult.data": 500, "adult.test": 250})
    digests = verify_raw(raw_dir)
    assert set(digests) == {"adult.data", "adult.test"} and all(len(d) == 64 for d in digests.values())
    with pytest.raises(FileNotFoundError):
        verify_raw(raw_dir / "nowhere")


def test_download_unpacks_archive_and_skips_when_present(
    raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "EXPECTED_ROWS", {"adult.data": 500, "adult.test": 250})
    archive = tmp_path / "adult.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("adult.data", "adult.test"):
            zf.write(raw_dir / name, arcname=f"adult/{name}")  # nested path, like a real archive
    dest = tmp_path / "out"
    assert download_adult(dest, url=archive.as_uri()) == dest
    assert (dest / "adult.data").read_bytes() == (raw_dir / "adult.data").read_bytes()
    (dest / "adult.data").write_text("changed")
    download_adult(dest, url="file:///does/not/exist")  # already present -> no network access needed
    assert (dest / "adult.data").read_text() == "changed"


def test_download_failure_explains_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="manually"):
        download_adult(tmp_path / "x", url=(tmp_path / "missing.zip").as_uri())


# --------------------------------------------------------------------------- splitting


def test_make_splits_partition_and_stratify() -> None:
    y = (np.random.default_rng(0).random(1000) < 0.25).astype(np.int64)
    tr, va, te = make_splits(y, val_frac=0.2, test_frac=0.2, seed=1)
    everything = np.concatenate([tr, va, te])
    assert sorted(everything) == list(range(1000))  # disjoint and complete
    assert (len(tr), len(va), len(te)) == (600, 200, 200)
    for part in (tr, va, te):
        assert list(part) == sorted(part)
        assert abs(y[part].mean() - y.mean()) < 0.02
    again = make_splits(y, 0.2, 0.2, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip((tr, va, te), again, strict=True))
    other = make_splits(y, 0.2, 0.2, seed=2)
    assert not np.array_equal(tr, other[0])


def test_subsample_is_stratified_and_seeded(toy_df: pd.DataFrame) -> None:
    sub = subsample(toy_df, 200, seed=0)
    assert len(sub) == 200
    assert abs(sub[LABEL].mean() - toy_df[LABEL].mean()) < 0.02
    assert sub.equals(subsample(toy_df, 200, seed=0))
    assert len(subsample(toy_df, None, seed=0)) == len(toy_df)
    assert len(subsample(toy_df, 10_000, seed=0)) == len(toy_df)


# --------------------------------------------------------------------------- preprocessing


def test_preprocessor_uses_training_statistics_only(toy_df: pd.DataFrame) -> None:
    train, other = toy_df.iloc[:400], toy_df.iloc[400:].copy()
    other["age"] = other["age"] + 30  # a shifted "test" distribution must stay shifted
    pre = Preprocessor().fit(train)
    a, b = pre.transform(train), pre.transform(other)
    age = NUM_COLS.index("age")
    assert a.x_num[:, age].mean() == pytest.approx(0.0, abs=1e-5)
    assert a.x_num[:, age].std() == pytest.approx(1.0, abs=1e-5)
    assert b.x_num[:, age].mean() > 1.0
    assert a.x_num.dtype == np.float32 and a.x_cat.dtype == np.int64 and len(a) == 400


def test_preprocessor_log_transforms_capital_columns(toy_df: pd.DataFrame) -> None:
    pre = Preprocessor().fit(toy_df)
    gain = NUM_COLS.index("capital-gain")
    raw = np.log1p(toy_df["capital-gain"].to_numpy(dtype=float))
    expected = (raw - raw.mean()) / raw.std()
    assert np.allclose(pre.transform(toy_df).x_num[:, gain], expected, atol=1e-4)


def test_rare_levels_are_merged_and_unseen_levels_map_to_rare() -> None:
    df = make_frame(1000, seed=1)
    df.loc[:2, "workclass"] = "Never-worked"  # 3 of 1000 rows = 0.3% < min_freq
    pre = Preprocessor(min_freq=0.005).fit(df)
    j = CAT_COLS.index("workclass")
    levels = pre.levels_[j]
    assert "Never-worked" not in levels and levels[-1] == RARE
    unseen = df.iloc[:5].copy()
    unseen["workclass"] = "Something-new"
    codes = pre.transform(unseen).x_cat[:, j]
    assert (codes == levels.index(RARE)).all()
    assert pre.cards[j] == len(levels)


def test_unseen_level_without_rare_bucket_is_an_error(toy_df: pd.DataFrame) -> None:
    pre = Preprocessor(min_freq=0.0).fit(toy_df)  # nothing merged -> no rare bucket
    assert all(RARE not in lv for lv in pre.levels_)
    bad = toy_df.iloc[:3].copy()
    bad["race"] = "Martian"
    with pytest.raises(ValueError, match="unseen in training"):
        pre.transform(bad)


def test_preprocessor_requires_fit(toy_df: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="fit"):
        Preprocessor().transform(toy_df)


def test_flat_features_layout(toy_df: pd.DataFrame) -> None:
    pre = Preprocessor().fit(toy_df)
    td = pre.transform(toy_df)
    flat = flat_features(td.x_num, td.x_cat, pre.cards)
    assert flat.shape == (len(toy_df), len(NUM_COLS) + sum(pre.cards))
    blocks = np.split(flat[:, len(NUM_COLS) :], np.cumsum(pre.cards)[:-1], axis=1)
    for j, block in enumerate(blocks):
        assert (block.sum(axis=1) == 1).all()  # exactly one level per column
        assert (block.argmax(axis=1) == td.x_cat[:, j]).all()


# --------------------------------------------------------------------------- real data (optional)


@pytest.mark.skipif(not (REAL_DIR / "adult.data").exists(), reason="run `make data` first")
def test_real_files_have_published_size_and_clean_as_expected() -> None:
    digests = verify_raw(REAL_DIR)  # 32,561 + 16,281 rows
    assert len(digests) == 2
    df, report = load_clean(REAL_DIR)
    assert report.n_raw == 48842
    assert report.n_final == report.n_raw - report.n_duplicates
    assert 0.23 < report.positive_rate < 0.26
    assert set(report.missing) == {"workclass", "occupation", "native-country"}
    assert df["age"].between(17, 90).all()
