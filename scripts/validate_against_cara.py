"""Validate OUR 2-D Pc implementations against NASA CARA's own released code.

The reference is `Pc2D_Foster.m` from github.com/nasa/CARA_Analysis_Tools
(NOSA GSC-18593-1), executed under GNU Octave on the Alfano (2009) conjunction
test cases whose published expected values ship with the SDK. This closes the
"validation against CARA tools on known cases" requirement: the reference is
CARA's code, not ours, and the test cases are the published ones.

Run order:
    python3 scripts/extract_alfano_cases.py     # .xls -> /tmp/alfano_cases.csv
    octave --no-gui --quiet scripts/run_cara.m  # CARA's code -> /tmp/cara_reference.csv
    python3 scripts/validate_against_cara.py    # this file: ours vs CARA

Reference: S. Alfano, "Satellite Conjunction Monte Carlo Analysis", AAS 09-233 (2009).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.covariance import encounter_plane_basis, project_to_plane
from src.foster2d import cara_monte_carlo
from src.pc2d import pc2d, pc2d_max, pc2d_numeric

ROOT = Path(__file__).resolve().parents[1]
CASES = Path("/tmp/alfano_cases.csv")
REF = Path("/tmp/cara_reference.csv")
OCTAVE = ROOT / "scripts" / "run_cara.m"
OUT = ROOT / "results" / "cara_alfano_validation.csv"

#: published CARA 2-D Pc values (CARA's own unit test + the A09 output CSV)
PUBLISHED = {1: 0.146749500506248, 2: 0.00622226705532899, 3: 0.100351017066071,
             4: 0.0493220745447709, 5: 0.044492344523551, 6: 0.00433545396136356,
             7: 0.000158146485930144, 8: 0.036947965085717, 9: 0.290161522245745,
             10: 0.290161522245745, 11: 0.00267203364642589, 12: float("nan")}


def sym(cols: dict, prefix: str, n: int) -> np.ndarray:
    """Rebuild a symmetric n x n matrix from its upper triangle in `cols`."""
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            M[i, j] = M[j, i] = cols[f"{prefix}_{i+1}{j+1}"]
    return M


def main() -> int:
    # The whole chain must be self-contained: extract the .xls inputs, then run
    # CARA's own code over them, then compare. Each step is skipped only if its
    # output is already present, so a partial /tmp never yields a silent pass.
    if not CASES.exists():
        print(f"{CASES} missing -- extracting the Alfano .xls inputs")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_alfano_cases.py")],
                       check=True)
    if not REF.exists():
        print(f"{REF} missing -- running CARA's Pc2D_Foster under Octave")
        subprocess.run(["octave", "--no-gui", "--quiet", str(OCTAVE)], check=True)

    cases = pd.read_csv(CASES)
    ref = pd.read_csv(REF)
    ref = ref.set_index("case")

    print("=" * 108)
    print("OUR 2-D Pc IMPLEMENTATIONS vs NASA CARA's Pc2D_Foster.m")
    print("Alfano (2009) published test cases, executed under GNU Octave")
    print("=" * 108)

    rows = []
    print(f"{'case':>4} {'HBR':>4} {'CARA Foster':>13} {'ours: N-2':>13} "
          f"{'ours: 2D quad':>13} {'ours: closed':>13} {'ours: MC':>11} "
          f"{'N-2 rel':>9} {'quad rel':>9}")
    print("-" * 108)

    for _, c in cases.iterrows():
        nc = int(c["case"])
        cara = float(ref.loc[nc, "cara_pc2d_foster"]) if nc in ref.index else np.nan
        if not np.isfinite(cara):
            print(f"{nc:4d} {int(c['hbr']):4d}   (CARA returned NaN -- skipped)")
            continue

        r1 = np.array([c.r1x, c.r1y, c.r1z]); v1 = np.array([c.v1x, c.v1y, c.v1z])
        r2 = np.array([c.r2x, c.r2y, c.r2z]); v2 = np.array([c.v2x, c.v2y, c.v2z])
        C1 = sym(c.to_dict(), "c1", 6)
        C2 = sym(c.to_dict(), "c2", 6)
        hbr = float(c["hbr"])

        # relative geometry in ECI, exactly as CARA builds it
        r_rel, v_rel = r1 - r2, v1 - v2
        C3 = C1[:3, :3] + C2[:3, :3]

        # encounter plane perpendicular to the relative velocity
        P = encounter_plane_basis(v_rel)
        C2p = project_to_plane(C3, P)
        m2 = P.T @ r_rel

        p_n2 = pc2d_numeric(m2, C2p, hbr)      # placeholder, replaced below
        from src.foster2d import pc2d_1d
        p_n2 = pc2d_1d(m2, C2p, hbr)            # Eq. N-2 (Alfano / Elrod)
        p_quad = pc2d_numeric(m2, C2p, hbr)     # Foster & Estes 1992 2-D quadrature
        p_closed = pc2d(m2, C2p, hbr)           # Chan small-HBR closed form
        p_max = pc2d_max(hbr, C2p)

        # CARA-style 6-D Monte Carlo from the FULL 6x6 covariance
        C6 = C1 + C2
        mc = cara_monte_carlo(r_rel, v_rel, C6, hbr, n_samples=2_000_000,
                              seed=nc, dt_s=0.0)

        rows.append({
            "case": nc, "hbr_m": hbr, "cara_pc2d_foster": cara,
            "ours_n2": p_n2, "ours_quadrature": p_quad, "ours_closed": p_closed,
            "ours_mc": mc["pc_mc"], "sigma_mc": mc["sigma_mc"],
            "pc_max": p_max, "published_pc2d": PUBLISHED.get(nc, np.nan),
            "miss_distance_m": float(np.linalg.norm(r_rel)),
            "rel_speed_m_s": float(np.linalg.norm(v_rel)),
        })

        print(f"{nc:4d} {int(hbr):4d} {cara:13.6e} {p_n2:13.6e} {p_quad:13.6e} "
              f"{p_closed:13.6e} {mc['pc_mc']:11.3e} {p_n2/cara-1:+9.1e} "
              f"{p_quad/cara-1:+9.1e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("-" * 108)
    print("\nSUMMARY vs CARA's Pc2D_Foster (the reference):")
    for col, label in (("ours_n2", "Eq. N-2 (Alfano/Elrod 1-D erf)"),
                       ("ours_quadrature", "Foster & Estes 1992 2-D quadrature"),
                       ("ours_closed", "Chan small-HBR closed form"),
                       ("ours_mc", "CARA-style 6-D Monte Carlo")):
        rel = (df[col] / df["cara_pc2d_foster"] - 1).abs()
        # MC: compare against its own Poisson error, not a ratio
        if col == "ours_mc":
            z = ((df[col] - df["cara_pc2d_foster"]).abs()
                 / df["sigma_mc"].clip(lower=1e-30))
            print(f"  {label:38s} worst |z| = {z.max():6.2f}  "
                  f"(all within 4 sigma: {bool((z < 4).all())})")
        else:
            print(f"  {label:38s} worst rel err = {rel.max():.3e}  "
                  f"median = {rel.median():.3e}")

    print("\nCARA's own unit test asserts RelTol = 1e-3 against these same cases.")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
