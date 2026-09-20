"""Synthetic stand-ins for the Adult table, so no test needs network access or the real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphmix.data import CAT_COLS, COLUMNS, LABEL, NUM_COLS, Preprocessor

LEVELS = {
    "workclass": (
        ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"],
        [0.6, 0.1, 0.05, 0.1, 0.08, 0.07],
    ),
    "marital-status": (
        ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"],
        [0.45, 0.3, 0.14, 0.06, 0.05],
    ),
    "occupation": (
        ["Sales", "Exec-managerial", "Craft-repair", "Adm-clerical", "Prof-specialty", "Other-service"],
        [0.2, 0.2, 0.15, 0.15, 0.2, 0.1],
    ),
    "relationship": (
        ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"],
        [0.4, 0.25, 0.15, 0.1, 0.1],
    ),
    "race": (["White", "Black", "Asian-Pac-Islander"], [0.8, 0.15, 0.05]),
    "sex": (["Male", "Female"], [0.65, 0.35]),
    "native-country": (["United-States", "Mexico", "India"], [0.9, 0.06, 0.04]),
}
EDUCATION = {"HS-grad": 9, "Some-college": 10, "Bachelors": 13, "Masters": 14, "11th": 7}


def make_frame(n: int, seed: int = 0) -> pd.DataFrame:
    """Random table with the *cleaned* schema; the label depends on age, education and marital
    status plus noise (about a quarter positives)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(17, 80, n),
            "education-num": rng.choice(list(EDUCATION.values()), n),
            "capital-gain": np.where(rng.random(n) < 0.1, rng.integers(500, 20000, n), 0),
            "capital-loss": np.where(rng.random(n) < 0.05, rng.integers(200, 2000, n), 0),
            "hours-per-week": rng.integers(10, 70, n),
        }
    )
    for col in CAT_COLS:
        levels, probs = LEVELS[col]
        df[col] = rng.choice(levels, n, p=probs)
    score = (
        0.05 * (df["age"] - 38)
        + 0.35 * (df["education-num"] - 10)
        + 1.2 * (df["marital-status"] == "Married-civ-spouse")
        + rng.normal(0, 1.0, n)
    )
    df[LABEL] = (score > np.quantile(score, 0.75)).astype(np.int64)
    return df[[*NUM_COLS, *CAT_COLS, LABEL]]


def make_features(n: int, n_train: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Model-ready features (numeric z-scores, categorical codes, cardinalities) of a random table;
    the preprocessor is fitted on the first n_train rows only, as in the real pipeline."""
    df = make_frame(n, seed)
    pre = Preprocessor().fit(df.iloc[:n_train])
    td = pre.transform(df)
    return td.x_num, td.x_cat, pre.cards


def uci_lines(n: int, seed: int, is_test: bool) -> list[str]:
    """Rows in the exact layout of the UCI files: ', ' separators, '?' for unknowns, no header;
    adult.test has a '|1x3 Cross validator' first line and labels ending in a period."""
    frame = make_frame(n, seed)
    rng = np.random.default_rng(seed + 1)
    education = {v: k for k, v in EDUCATION.items()}
    lines = ["|1x3 Cross validator"] if is_test else []
    for row in frame.to_dict("records"):
        values = {
            "age": row["age"], "workclass": row["workclass"], "fnlwgt": int(rng.integers(20000, 400000)),
            "education": education[row["education-num"]], "education-num": row["education-num"],
            "marital-status": row["marital-status"], "occupation": row["occupation"],
            "relationship": row["relationship"], "race": row["race"], "sex": row["sex"],
            "capital-gain": row["capital-gain"], "capital-loss": row["capital-loss"],
            "hours-per-week": row["hours-per-week"], "native-country": row["native-country"],
            "income": ">50K" if row[LABEL] else "<=50K",
        }  # fmt: skip
        if rng.random() < 0.05:
            values["occupation"] = "?"
        if is_test:
            values["income"] += "."
        lines.append(", ".join(str(values[c]) for c in COLUMNS))
    if not is_test:
        lines.append("")  # the real adult.data ends with an empty line
    return lines


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return make_frame(600, seed=0)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A directory laid out like data/raw/adult with small synthetic files."""
    (tmp_path / "adult.data").write_text("\n".join(uci_lines(500, seed=1, is_test=False)) + "\n")
    (tmp_path / "adult.test").write_text("\n".join(uci_lines(250, seed=2, is_test=True)) + "\n")
    return tmp_path
