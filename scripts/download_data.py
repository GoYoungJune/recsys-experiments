"""Download the datasets into data/: ml-latest-small -> data/raw/, ml-1m -> data/ml-1m/, ESCI -> data/esci/.

The datasets are not redistributed in this repository (see LICENSE); run this script instead.
    python scripts/download_data.py            # ml-latest-small (part 1)
    python scripts/download_data.py ml-1m      # part 2
    python scripts/download_data.py esci       # search reranking, about 1.2 GB
    python scripts/download_data.py hillstrom  # part 4, real email A/B test, 4 MB
"""
from __future__ import annotations

import hashlib
import io
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DATASETS = {
    "ml-latest-small": ("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
                        ("ratings.csv", "movies.csv"), DATA / "raw"),
    "ml-1m": ("https://files.grouplens.org/datasets/movielens/ml-1m.zip", ("ratings.dat", "movies.dat", "users.dat"),
              DATA / "ml-1m"),
}
# Amazon Shopping Queries Dataset (Apache-2.0), Git LFS files; checksums from the LFS pointers in the repo
ESCI_URL = "https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/"
ESCI_FILES = {
    "examples.parquet": ("shopping_queries_dataset_examples.parquet",
                         "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"),
    "products.parquet": ("shopping_queries_dataset_products.parquet",
                         "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"),
}
# Kevin Hillstrom's MineThatData E-Mail Analytics challenge (2008). The server is HTTP only, so the file is
# checked against a fixed SHA-256 instead of relying on TLS.
HILLSTROM_URL = ("http://www.minethatdata.com/"
                 "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
HILLSTROM_SHA = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"


def ssl_context() -> ssl.SSLContext:
    try:  # some Python installs (e.g. Windows) ship without a CA bundle; certifi fixes that if present
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_esci() -> None:
    """Stream each file to disk (products is 1.1 GB) and check its SHA-256."""
    out_dir = DATA / "esci"
    out_dir.mkdir(parents=True, exist_ok=True)
    for local, (remote, sha) in ESCI_FILES.items():
        path = out_dir / local
        if path.exists():
            print(f"Already downloaded: {path}")
            continue
        print(f"Downloading {remote} ...")
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(ESCI_URL + remote, timeout=60, context=ssl_context()) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)
        digest = hashlib.sha256()
        with open(tmp, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != sha:
            tmp.unlink()
            raise RuntimeError(f"Checksum mismatch for {remote}; the download was removed, try again.")
        tmp.rename(path)
        print(f"  saved {path} (checksum ok)")


def download_hillstrom() -> None:
    path = DATA / "hillstrom" / "hillstrom.csv"
    if path.exists():
        print(f"Already downloaded: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {HILLSTROM_URL} ...")
    with urllib.request.urlopen(HILLSTROM_URL, timeout=60) as resp:
        payload = resp.read()
    if hashlib.sha256(payload).hexdigest() != HILLSTROM_SHA:
        raise RuntimeError("Checksum mismatch for the Hillstrom file; nothing was saved.")
    path.write_bytes(payload)
    print(f"  saved {path} (checksum ok)")


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "ml-latest-small"
    if name == "esci":
        download_esci()
        return
    if name == "hillstrom":
        download_hillstrom()
        return
    url, files, out_dir = DATASETS[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    if all((out_dir / f).exists() for f in files):
        print(f"Already downloaded: {out_dir}")
        return
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60, context=ssl_context()) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for f in files:
            (out_dir / f).write_bytes(zf.read(f"{name}/{f}"))
            print(f"  saved {out_dir / f}")


if __name__ == "__main__":
    main()
