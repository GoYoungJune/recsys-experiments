"""Download MovieLens into data/: ml-latest-small -> data/raw/, ml-1m -> data/ml-1m/.

The datasets are not redistributed in this repository (see LICENSE); run this script instead.
    python scripts/download_data.py            # ml-latest-small
    python scripts/download_data.py ml-1m
"""
from __future__ import annotations

import io
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DATASETS = {
    "ml-latest-small": ("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
                        ("ratings.csv", "movies.csv"), DATA / "raw"),
    "ml-1m": ("https://files.grouplens.org/datasets/movielens/ml-1m.zip", ("ratings.dat", "movies.dat"), DATA / "ml-1m"),
}


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "ml-latest-small"
    url, files, out_dir = DATASETS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    if all((out_dir / f).exists() for f in files):
        print(f"Already downloaded: {out_dir}")
        return
    print(f"Downloading {url} ...")
    try:  # some Python installs (e.g. Windows) ship without a CA bundle; certifi fixes that if present
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=60, context=ctx) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for f in files:
            (out_dir / f).write_bytes(zf.read(f"{name}/{f}"))
            print(f"  saved {out_dir / f}")


if __name__ == "__main__":
    main()
