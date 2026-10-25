"""Extract the Alfano (2009) test-case states and covariances from CARA's .xls files.

CARA's `GetAlfanoTestCases.m` reads these with MATLAB `xlsread`, which the
Octave `io` package cannot do for legacy .xls. So we read them here and hand the
numbers straight to CARA's `Pc2D_Foster`, which stays the reference
implementation.

Field names follow GetAlfanoTestCases.m: R1o V1o P1o / R2o V2o P2o.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path("/home/user/CARA_Analysis_Tools/DataFiles/AlfanoInputData/"
            "Alfano_2009_Test_Cases")
OUT = Path("/tmp/alfano_cases.csv")


def read_sheet(path: Path) -> pd.DataFrame:
    """Read one of the legacy .xls sheets, trying the engines that are present."""
    last = None
    for engine in ("xlrd", "calamine", None):
        try:
            return pd.read_excel(path, header=None, engine=engine)
        except Exception as e:                      # noqa: BLE001
            last = e
    raise RuntimeError(f"cannot read {path.name}: {last}")


def find_block(df: pd.DataFrame, label: str):
    """Locate a labelled block and return the numeric rows that follow it."""
    lab = str(label).strip().lower()
    for i in range(len(df)):
        for j in range(df.shape[1]):
            v = df.iat[i, j]
            if isinstance(v, str) and v.strip().lower() == lab:
                # collect the next rows of numbers starting at column j+1
                rows = []
                k = i + 1
                while k < len(df):
                    vals = pd.to_numeric(df.iloc[k, j + 1:j + 4], errors="coerce")
                    if vals.isna().any():
                        break
                    rows.append(vals.to_numpy(dtype=float))
                    k += 1
                return rows
    return None


def main() -> int:
    # is any engine available at all?
    try:
        import xlrd  # noqa: F401
        print("xlrd available")
    except ImportError:
        print("xlrd NOT available -- will try calamine/openpyxl")

    rows = []
    for nc in range(1, 13):
        for tag in ("Epoch Time", "TCA"):
            p = BASE / f"Case{nc} data at {tag}.xls"
            if not p.exists():
                print(f"  MISSING {p.name}")
                continue
            df = read_sheet(p)
            print(f"  Case{nc} {tag}: {df.shape}")
            # dump the raw grid so we can see the layout
            if nc <= 1:
                print(df.head(30).to_string())

    # dump case 1 TCA fully to understand the layout
    p = BASE / "Case1 data at TCA.xls"
    df = read_sheet(p)
    print("\n=== Case1 data at TCA.xls (full) ===")
    print(df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
