"""The command-line entry points, run the way the Makefile runs them."""

import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_command_line_stages_run_end_to_end(raw_dir: Path, tmp_path: Path) -> None:
    config = {
        "name": "cli",
        "seeds": [0],
        "runs_dir": str(tmp_path / "runs"),
        "results_dir": str(tmp_path / "results"),
        "data": {"raw_dir": str(raw_dir)},
        "graph": {"k": 5},
        "model": {"hidden": 16, "dim": 4},
        "evaluation": {
            "c_grid": [1.0],
            "label_sizes": [20],
            "label_repeats": 2,
            "knn_k": 5,
            "tsne_points": 60,
        },
        "methods": [{"name": "raw", "kind": "raw"}, {"name": "pca", "kind": "pca"}],
    }
    cfg_path = tmp_path / "cli.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

    run("train_encoders.py", "--config", str(cfg_path), "--quiet")
    assert (tmp_path / "runs" / "cli" / "seed0" / "raw.npz").exists()
    out = run("evaluate.py", "--config", str(cfg_path))
    assert "Frozen-embedding evaluation" in out.stdout
    assert (tmp_path / "results" / "cli" / "config.yaml").exists()  # the config is archived with the results
