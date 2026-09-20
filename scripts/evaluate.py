"""Stage 2: evaluate the saved embeddings and write tables and figures.

    python scripts/evaluate.py --config configs/quick.yaml
    python scripts/evaluate.py --config configs/full.yaml

Outputs go to results/<name>/: metrics.csv (long format), summary.md, graph_diagnostics.csv,
training_logs.csv, run_info.json, config.yaml (a copy of the config used) and figures/*.png.
"""

from __future__ import annotations

import argparse
import shutil

from graphmix.config import load_config
from graphmix.pipeline import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="evaluate only these seeds")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_evaluation(cfg, seeds=args.seeds, figures=not args.no_figures)
    shutil.copy(args.config, cfg.result_path / "config.yaml")
    summary = (cfg.result_path / "summary.md").read_text()
    print(summary)


if __name__ == "__main__":
    main()
