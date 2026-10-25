"""Re-acquire the Kelvins dataset.

The dataset is ~221 MB zipped and is NOT part of the saved workspace snapshot
(it exceeds the snapshot size cap), so it has to be re-fetched after any reset.
This script reproduces the exact steps used to restore it.

Layout note: the Zenodo release unpacks to
    <root>/Collision Avoidance Challenge - Dataset/
but the loaders in ``src/kelvins_data.py`` expect
    <root>/kelvins/Collision Avoidance Challenge - Dataset/
so an extra ``kelvins`` level is inserted.  ``train_data.csv`` ships as
``train_data.zip`` and must be extracted.

Usage:
    python3 scripts/fetch_data.py            # download + unpack + place
    python3 scripts/fetch_data.py --check    # only verify what is present
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ZENODO_RECORD = "https://zenodo.org/api/records/4463683"
EXPECTED_ZIP_BYTES = 221_128_642
STAGING = Path("/home/user/data")
FINAL_ROOT = STAGING / "kelvins"
DATASET_DIR = "Collision Avoidance Challenge - Dataset"


def paths():
    root = FINAL_ROOT / DATASET_DIR
    return {
        "root": root,
        "train": root / "kelvins_competition_data" / "train_data.csv",
        "test": root / "kelvins_competition_data" / "test_data.csv",
        "truth": root / "kelvins_competition_data" / "test_data_private.csv",
        "raw": root / "raw_data" / "raw_data_2015-2019.gz",
        "zip": STAGING / "dataset.zip",
    }


def check() -> bool:
    p = paths()
    ok = True
    for k in ("train", "test", "truth", "raw"):
        exists = p[k].exists()
        print(("  OK   " if exists else "  MISS ") + f"{k:6s} {p[k]}")
        ok &= exists
    return ok


def fetch() -> bool:
    p = paths()
    STAGING.mkdir(parents=True, exist_ok=True)

    print(f"downloading {ZENODO_RECORD} ...")
    url = (f"{ZENODO_RECORD}/files/"
           f"Collision%20Avoidance%20Challenge%20-%20Dataset.zip/content")
    subprocess.run(["curl", "-sS", "-L", "--retry", "3", "--retry-delay", "5",
                    "-o", str(p["zip"]), url], check=True)
    size = p["zip"].stat().st_size
    print(f"  {size:,} bytes (expected {EXPECTED_ZIP_BYTES:,})")
    if size != EXPECTED_ZIP_BYTES:
        print("  WARNING: size mismatch -- the download may be truncated")

    print("unpacking ...")
    with zipfile.ZipFile(p["zip"]) as zf:
        zf.extractall(STAGING)

    # train_data.csv ships zipped
    comp = STAGING / DATASET_DIR / "kelvins_competition_data"
    inner = comp / "train_data.zip"
    if inner.exists():
        print("extracting train_data.zip ...")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(comp)

    # insert the extra `kelvins` level the loaders expect
    src = STAGING / DATASET_DIR
    dst = FINAL_ROOT / DATASET_DIR
    if src.exists() and not dst.exists():
        FINAL_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"placed dataset at {dst}")
    elif dst.exists():
        print(f"already present at {dst}")
    return True


def main() -> int:
    if "--check" in sys.argv:
        print("dataset check:")
        return 0 if check() else 1
    if check():
        print("dataset already complete; nothing to do")
        return 0
    fetch()
    print("\nfinal check:")
    ok = check()
    print("\nDONE" if ok else "\nINCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
