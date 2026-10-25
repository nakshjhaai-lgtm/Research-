"""Extract the Alfano (2009) conjunction test cases from CARA's released .xls files.

CARA's `GetAlfanoTestCases.m` reads these with MATLAB `xlsread`, which Octave's
`io` package cannot do for legacy .xls. So we read them here and hand the numbers
straight to CARA's own `Pc2D_Foster.m`, which remains the reference
implementation.

Row indices follow `get_alfano_test_case.m` (1-based): 5=position, 9=velocity,
13..18=6x6 covariance for the primary; 23, 27, 31..36 for the secondary.
Units are metres and seconds, exactly as CARA consumes them (no conversion).

Reference: S. Alfano, "Satellite Conjunction Monte Carlo Analysis", AAS 09-233 (2009).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/home/user/CARA_Analysis_Tools/DataFiles/AlfanoInputData/"
            "Alfano_2009_Test_Cases")
OUT = Path("/tmp/alfano_cases.csv")

# published CARA 2-D Pc values, from CARA's own unit test and the A09 output CSV
PUBLISHED_PC2D = {
    1: 0.146749500506248, 2: 0.00622226705532899, 3: 0.100351017066071,
    4: 0.0493220745447709, 5: 0.044492344523551, 6: 0.00433545396136356,
    7: 0.000158146485930144, 8: 0.036947965085717, 9: 0.290161522245745,
    10: 0.290161522245745, 11: 0.00267203364642589, 12: float("nan"),
}
HBR = {1: 15, 2: 4, 3: 15, 4: 15, 5: 10, 6: 10, 7: 10, 8: 4, 9: 6, 10: 6,
       11: 4, 12: 4}


def read_case(nc: int) -> dict | None:
    p = BASE / f"Case{nc} data at TCA.xls"
    if not p.exists():
        return None
    df = pd.read_excel(p, header=None, engine="xlrd")

    def vec(row0: int):
        return [float(df.iat[row0, c]) for c in range(3)]

    def mat(row0: int):
        return [[float(df.iat[row0 + i, j]) for j in range(6)] for i in range(6)]

    return {
        "case": nc,
        "hbr": HBR[nc],
        "r1": vec(4), "v1": vec(8), "C1": mat(12),
        "r2": vec(22), "v2": vec(26), "C2": mat(30),
        "published_pc2d": PUBLISHED_PC2D[nc],
    }


def main() -> int:
    cases = [c for c in (read_case(n) for n in range(1, 13)) if c]
    print(f"extracted {len(cases)} Alfano cases")

    # sanity: covariance blocks must be symmetric and the 3x3 position block PD
    for c in cases:
        for tag in ("C1", "C2"):
            M = np.array(c[tag])
            if not np.allclose(M, M.T, atol=1e-9 * max(1.0, abs(M).max())):
                print(f"  WARN case {c['case']} {tag} not symmetric")
            w = np.linalg.eigvalsh(M[:3, :3])
            if w.min() <= 0:
                print(f"  WARN case {c['case']} {tag} position block not PD: {w}")

    cols = (["case", "hbr", "r1x", "r1y", "r1z", "v1x", "v1y", "v1z",
             "r2x", "r2y", "r2z", "v2x", "v2y", "v2z"]
            + [f"c1_{i}{j}" for i in range(1, 7) for j in range(i, 7)]
            + [f"c2_{i}{j}" for i in range(1, 7) for j in range(i, 7)]
            + ["published_pc2d"])
    rows = []
    for c in cases:
        r = {"case": c["case"], "hbr": c["hbr"], "published_pc2d": c["published_pc2d"]}
        for k, v in (("r1", c["r1"]), ("v1", c["v1"]), ("r2", c["r2"]), ("v2", c["v2"])):
            for i, x in enumerate(v, 1):
                r[f"{k}{['x','y','z'][i-1]}"] = x
        for tag in ("C1", "C2"):
            M = c[tag]
            for i in range(6):
                for j in range(i, 6):
                    r[f"{tag.lower()}_{i+1}{j+1}"] = M[i][j]
        rows.append(r)
    pd.DataFrame(rows)[cols].to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(rows)} rows, {len(cols)} cols)")
    for c in cases:
        print(f"  case {c['case']:2d}  HBR={c['hbr']:3d} m  "
              f"published CARA Pc2D = {c['published_pc2d']:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
