"""Download MovieLens ml-latest-small into data/raw/ (ratings.csv, movies.csv).

The dataset is not redistributed in this repository (see LICENSE); run this script instead.
"""
from __future__ import annotations

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
FILES = ("ratings.csv", "movies.csv")
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DIR / f).exists() for f in FILES):
        print(f"Already downloaded: {RAW_DIR}")
        return
    print(f"Downloading {URL} ...")
    try:  # some Python installs (e.g. Windows) ship without a CA bundle; certifi fixes that if present
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(URL, timeout=60, context=ctx) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in FILES:
            (RAW_DIR / name).write_bytes(zf.read(f"ml-latest-small/{name}"))
            print(f"  saved {RAW_DIR / name}")


if __name__ == "__main__":
    main()
