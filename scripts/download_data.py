"""Download the UCI Adult files (adult.data, adult.test) and check them.

python scripts/download_data.py            # download if missing, verify row counts
python scripts/download_data.py --summary  # also print what the cleaning step does to the table
"""

from __future__ import annotations

import argparse

from graphmix.data import (
    ADULT_URL,
    Preprocessor,
    describe,
    download_adult,
    load_clean,
    verify_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dest", default="data/raw/adult", help="target directory")
    parser.add_argument("--force", action="store_true", help="download again even if files exist")
    parser.add_argument("--summary", action="store_true", help="print a summary of the cleaned table")
    args = parser.parse_args()

    print(f"source: {ADULT_URL}")
    path = download_adult(args.dest, force=args.force)
    for name, digest in verify_raw(path).items():
        print(f"ok  {path / name}  sha256={digest}")
    if args.summary:
        df, report = load_clean(path)
        print()
        print(describe(report, Preprocessor().fit(df)))


if __name__ == "__main__":
    main()
