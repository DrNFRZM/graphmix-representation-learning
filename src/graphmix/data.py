"""UCI Adult (Census Income): download, parsing, cleaning, splitting and encoding.

Source: Becker & Kohavi (1996), https://doi.org/10.24432/C5XW20, CC BY 4.0.
Only numpy / pandas / scikit-learn are used here.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ADULT_FILES = ("adult.data", "adult.test")
EXPECTED_ROWS = {"adult.data": 32561, "adult.test": 16281}

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]  # fmt: skip

# fnlwgt is a survey sampling weight and `education` is a string copy of `education-num`,
# so both are dropped (see docs/design_decisions.md).
NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = [
    "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country",
]  # fmt: skip
LOG_COLS = ("capital-gain", "capital-loss")
LABEL = "income"
UNKNOWN = "Unknown"  # what "?" becomes
RARE = "__rare__"  # bucket for levels below the frequency threshold or unseen at fit time


# --------------------------------------------------------------------------- download / parsing


def _data_lines(path: Path) -> int:
    with path.open() as fh:
        return sum(1 for line in fh if line.strip() and not line.startswith("|"))


def verify_raw(raw_dir: str | Path, files: Sequence[str] = ADULT_FILES) -> dict[str, str]:
    """Check row counts against the published sizes and return SHA-256 digests."""
    digests = {}
    for name in files:
        path = Path(raw_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `make data` first")
        n = _data_lines(path)
        if n != EXPECTED_ROWS[name]:
            raise RuntimeError(f"{path}: found {n} rows, expected {EXPECTED_ROWS[name]}")
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def download_adult(raw_dir: str | Path, force: bool = False, url: str = ADULT_URL) -> Path:
    """Fetch the official UCI archive and unpack adult.data / adult.test into raw_dir."""
    raw_dir = Path(raw_dir)
    if not force and all((raw_dir / f).exists() for f in ADULT_FILES):
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "graphmix-representation-learning"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as err:
        raise RuntimeError(
            f"could not download {url}: {err}\n"
            f"Get adult.data and adult.test manually from https://archive.ics.uci.edu/dataset/2/adult "
            f"and put them in {raw_dir}/"
        ) from err
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = {Path(n).name: n for n in zf.namelist()}
        missing = [f for f in ADULT_FILES if f not in members]
        if missing:
            raise RuntimeError(f"archive has no {missing}; it contains {sorted(members)}")
        for name in (*ADULT_FILES, "adult.names"):
            if name in members:
                (raw_dir / name).write_bytes(zf.read(members[name]))
    verify_raw(raw_dir)
    return raw_dir


def read_adult_file(path: str | Path) -> pd.DataFrame:
    """Parse one raw UCI file. adult.test starts with a '|1x3 Cross validator' line and its
    labels end in a period (">50K."), both of which are handled here."""
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|",
        skip_blank_lines=True,
    )
    for col in [*CAT_COLS, LABEL, "education"]:
        df[col] = df[col].astype(object)
    df[LABEL] = df[LABEL].str.strip().str.rstrip(".")
    return df


def load_raw(raw_dir: str | Path, files: Sequence[str] = ADULT_FILES) -> pd.DataFrame:
    frames = []
    for name in files:
        path = Path(raw_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `make data` first")
        frames.append(read_adult_file(path))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- cleaning


@dataclass
class CleaningReport:
    n_raw: int
    n_duplicates: int
    n_final: int
    missing: dict[str, int]  # "?" counts per column, before they become UNKNOWN
    positive_rate: float


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Drop fnlwgt/education, turn '?' into an explicit level, binarise the label and remove
    exact duplicate rows (done before any split, so no row can sit in two splits)."""
    labels = set(raw[LABEL].unique())
    if not labels <= {"<=50K", ">50K"}:
        raise ValueError(f"unexpected income labels: {sorted(labels)}")
    missing = {c: int(raw[c].isna().sum()) for c in CAT_COLS if raw[c].isna().any()}
    df = raw[[*NUM_COLS, *CAT_COLS, LABEL]].copy()
    for col in CAT_COLS:
        df[col] = df[col].fillna(UNKNOWN).astype(str)
    df[LABEL] = (df[LABEL] == ">50K").astype(np.int64)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report = CleaningReport(
        n_raw=n_before,
        n_duplicates=n_before - len(df),
        n_final=len(df),
        missing=missing,
        positive_rate=float(df[LABEL].mean()),
    )
    return df, report


def load_clean(
    raw_dir: str | Path, files: Sequence[str] = ADULT_FILES
) -> tuple[pd.DataFrame, CleaningReport]:
    return clean(load_raw(raw_dir, files))


# --------------------------------------------------------------------------- splitting


def subsample(df: pd.DataFrame, n_rows: int | None, seed: int) -> pd.DataFrame:
    """Stratified subsample used by the quick config; a no-op if n_rows covers the table."""
    if n_rows is None or n_rows >= len(df):
        return df.reset_index(drop=True)
    keep, _ = train_test_split(
        np.arange(len(df)), train_size=n_rows, stratify=df[LABEL].to_numpy(), random_state=seed
    )
    return df.iloc[np.sort(keep)].reset_index(drop=True)


def make_splits(
    y: np.ndarray, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train / val / test indices (sorted, disjoint, covering all rows)."""
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    train, val = train_test_split(
        rest, test_size=val_frac / (1.0 - test_frac), stratify=y[rest], random_state=seed
    )
    return np.sort(train), np.sort(val), np.sort(test)


# --------------------------------------------------------------------------- encoding


@dataclass(frozen=True)
class TabularData:
    x_num: np.ndarray  # (n, n_num) float32, standardised with training statistics
    x_cat: np.ndarray  # (n, n_cat) int64 level indices

    def __len__(self) -> int:
        return len(self.x_num)


class Preprocessor:
    """Fitted on the training rows only; applied unchanged to validation and test rows.

    Numeric: log1p on the zero-inflated capital columns, then z-score.
    Categorical: integer codes; levels rarer than `min_freq` in the training rows share one
    `__rare__` level (also used for levels never seen in training).
    """

    def __init__(
        self,
        min_freq: float = 0.005,
        num_cols: Sequence[str] = NUM_COLS,
        cat_cols: Sequence[str] = CAT_COLS,
        log_cols: Sequence[str] = LOG_COLS,
    ) -> None:
        self.min_freq = min_freq
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.log_cols = set(log_cols)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.levels_: list[list[str]] = []

    def _numeric(self, df: pd.DataFrame) -> np.ndarray:
        x = np.array(df[self.num_cols], dtype=np.float64)
        for j, col in enumerate(self.num_cols):
            if col in self.log_cols:
                x[:, j] = np.log1p(np.clip(x[:, j], 0.0, None))
        return x

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        x = self._numeric(df)
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        self.levels_ = []
        for col in self.cat_cols:
            freq = df[col].astype(str).value_counts(normalize=True)
            kept = sorted(freq.index[freq >= self.min_freq])
            if not kept:
                raise ValueError(f"min_freq={self.min_freq} removes every level of '{col}'")
            merged = len(kept) < len(freq)
            self.levels_.append([*kept, RARE] if merged else kept)
        return self

    @property
    def cards(self) -> list[int]:
        return [len(levels) for levels in self.levels_]

    def transform(self, df: pd.DataFrame) -> TabularData:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Preprocessor.fit must be called first")
        x_num = ((self._numeric(df) - self.mean_) / self.std_).astype(np.float32)
        x_cat = np.empty((len(df), len(self.cat_cols)), dtype=np.int64)
        for j, col in enumerate(self.cat_cols):
            lookup = {lvl: i for i, lvl in enumerate(self.levels_[j])}
            codes = df[col].astype(str).map(lookup)
            if codes.isna().any():
                if RARE not in lookup:
                    bad = sorted(set(df[col].astype(str)[codes.isna()]))
                    raise ValueError(f"'{col}' has levels unseen in training: {bad}")
                codes = codes.fillna(lookup[RARE])
            x_cat[:, j] = codes.to_numpy(dtype=np.int64)
        return TabularData(x_num=x_num, x_cat=x_cat)


def flat_features(x_num: np.ndarray, x_cat: np.ndarray, cards: Sequence[int]) -> np.ndarray:
    """Numeric columns followed by one-hot blocks: the 'flat' view of a row."""
    blocks = [x_num.astype(np.float32)]
    for j, card in enumerate(cards):
        blocks.append(np.eye(card, dtype=np.float32)[x_cat[:, j]])
    return np.concatenate(blocks, axis=1)


def describe(report: CleaningReport, pre: Preprocessor | None = None) -> str:
    lines = [
        f"rows read            : {report.n_raw}",
        f"exact duplicates     : {report.n_duplicates}",
        f"rows after cleaning  : {report.n_final}",
        f"positive rate (>50K) : {report.positive_rate:.4f}",
        f"'?' per column       : {report.missing}",
    ]
    if pre is not None:
        lines.append("levels after rare-merging (fit on all rows, for illustration):")
        for col, levels in zip(pre.cat_cols, pre.levels_, strict=True):
            lines.append(f"  {col:15s} {len(levels):3d}  {levels if len(levels) <= 8 else '...'}")
    return "\n".join(lines)
