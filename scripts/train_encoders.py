"""Stage 1: compute the representation of every row for every (seed, method) in a config.

    python scripts/train_encoders.py --config configs/quick.yaml
    python scripts/train_encoders.py --config configs/full.yaml --seeds 0 1 --only mlp_ae sage_knn

Embeddings are written to runs/<name>/seed<k>/<method>.npz (not tracked by git). Existing files
are skipped unless --overwrite is given, so an interrupted run can simply be restarted. The graph
diagnostics of a seed are written when its run is complete; --only leaves existing ones alone.
"""

from __future__ import annotations

import argparse

from graphmix.config import load_config
from graphmix.pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="override the seeds in the config")
    parser.add_argument("--only", nargs="+", help="restrict to these method names")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_training(cfg, seeds=args.seeds, only=args.only, overwrite=args.overwrite, verbose=not args.quiet)


if __name__ == "__main__":
    main()
