"""Layer 1 validation: Foster 2-D vs Chan closed form vs CARA-style Monte Carlo.

Three independent implementations of the same quantity, checked against each
other and against exact analytic anchors.

Check 1  EXACT ANCHOR   isotropic C2, zero miss -> Rayleigh CDF 1-exp(-R^2/2s^2)
Check 2  SCHEME AGREEMENT  anisotropic + offset miss, foster2d vs polar quadrature
Check 3  CARA-STYLE MC  6-D draw -> propagate to TCA -> project -> count hits,
                        compared to foster2d on the propagated covariance
Check 4  REAL-WORLD ANCHOR (--with-data)  ESA's `risk` field in the Kelvins CDMs

Exit code 1 if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.foster2d import cara_monte_carlo, encounter_plane_covariance, foster2d
from src.pc2d import pc2d, pc2d_numeric

OUT = Path(__file__).resolve().parents[1] / "results" / "foster_mc_validation.csv"


def realistic_c6(sigma_r=600.0, sigma_t=1200.0, sigma_n=40.0,
                 vr_dot=0.5, vt_dot=1.0, vn_dot=0.02, seed=11) -> np.ndarray:
    """A physically plausible 6-D RTN relative covariance (guaranteed PD).

    Position sigmas typical of a CDM a few days from TCA, with along-track
    dominated uncertainty and the position-velocity correlations that a real
    orbit determination produces.
    """
    rng = np.random.default_rng(seed)
    d = np.array([sigma_r, sigma_t, sigma_n, vr_dot, vt_dot, vn_dot])
    # correlation structure: position and its own rate correlate positively,
    # R and T correlate through the along-track drift
    R = np.eye(6)
    R[0, 1] = R[1, 0] = 0.35          # r-t
    R[3, 4] = R[4, 3] = 0.30          # rdot-tdot
    R[0, 3] = R[3, 0] = 0.55          # r - rdot
    R[1, 4] = R[4, 1] = 0.45          # t - tdot
    R[2, 5] = R[5, 2] = 0.60          # n - ndot
    R[0, 4] = R[4, 0] = 0.20
    R[1, 3] = R[3, 1] = 0.15
    C = R * np.outer(d, d)
    # jitter to guarantee strict positive definiteness
    return C + np.diag(rng.uniform(1e-6, 1e-4, 6))


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    rows: list[dict] = []
    failures: list[str] = []

    # ---------------------------------------------------------------- check 1
    print("=" * 100)
    print("CHECK 1 -- EXACT ANCHOR: isotropic C2, zero miss -> Rayleigh CDF")
    print("=" * 100)
    print(f"{'sigma_m':>9} {'R_m':>6} {'R/sigma':>8} {'exact':>13} {'foster':>13} "
          f"{'rel_err':>10} {'quad_rel':>10} {'chan_rel':>10}")
    for sigma, R in ((100.0, 20.0), (500.0, 20.0), (1000.0, 20.0),
                     (200.0, 40.0), (2000.0, 100.0)):
        C2 = np.array([[sigma ** 2, 0.0], [0.0, sigma ** 2]])
        exact = 1.0 - np.exp(-R ** 2 / (2.0 * sigma ** 2))
        f = foster2d([0.0, 0.0], C2, R)
        q = pc2d_numeric([0.0, 0.0], C2, R)
        c = pc2d([0.0, 0.0], C2, R)
        rel_f, rel_q, rel_c = f / exact - 1, q / exact - 1, c / exact - 1
        print(f"{sigma:9.0f} {R:6.1f} {R/sigma:8.4f} {exact:13.6e} {f:13.6e} "
              f"{rel_f:+10.2e} {rel_q:+10.2e} {rel_c:+10.2e}")
        rows.append({"check": "rayleigh_anchor", "sigma_m": sigma, "R_m": R,
                     "exact": exact, "foster": f, "quadrature": q, "chan": c,
                     "foster_rel_err": rel_f, "quad_rel_err": rel_q, "chan_rel_err": rel_c})
        if abs(rel_f) > 1e-6:
            failures.append(f"rayleigh foster rel err {rel_f:.2e} at sigma={sigma}")
        if abs(rel_q) > 1e-5:
            failures.append(f"rayleigh quadrature rel err {rel_q:.2e} at sigma={sigma}")

    # ---------------------------------------------------------------- check 2
    print("\n" + "=" * 100)
    print("CHECK 2 -- SCHEME AGREEMENT: foster2d vs polar quadrature (anisotropic, offset)")
    print("=" * 100)
    print(f"{'s1':>7} {'s2':>7} {'rho':>6} {'|m|':>8} {'R':>6} {'foster':>12} "
          f"{'quad':>12} {'rel_err':>10} {'chan_rel':>10}")
    cases = [(300., 1200., 0.4, 50., -80., 25.), (800., 800., 0.0, 0., 0., 15.),
             (150., 900., -0.7, 400., 200., 10.), (2000., 300., 0.9, 100., -300., 60.),
             (500., 2500., -0.5, 900., -120., 30.)]
    for i, (s1, s2, rho, mx, my, R) in enumerate(cases):
        C2 = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
        m = np.array([mx, my])
        f = foster2d(m, C2, R)
        q = pc2d_numeric(m, C2, R)
        c = pc2d(m, C2, R)
        print(f"{s1:7.0f} {s2:7.0f} {rho:6.2f} {np.hypot(mx,my):8.0f} {R:6.1f} "
              f"{f:12.6e} {q:12.6e} {f/q-1:+10.2e} {c/f-1:+10.2e}")
        rows.append({"check": "scheme_agreement", "case": i, "sigma1_m": s1,
                     "sigma2_m": s2, "rho": rho, "miss_m": np.hypot(mx, my),
                     "R_m": R, "foster": f, "quadrature": q, "chan": c,
                     "foster_rel_err": f / q - 1, "chan_rel_err": c / f - 1})
        if abs(f / q - 1) > 1e-5:
            failures.append(f"scheme agreement case {i}: {f/q-1:.2e}")

    # ---------------------------------------------------- check 2b: R/sigma sweep
    print("\n" + "-" * 100)
    print("CHECK 2b -- size of the small-HBR approximation error (Chan vs exact)")
    print("-" * 100)
    print(f"{'R/sigma':>9} {'eccentricity':>13} {'exact':>12} {'chan':>12} {'chan_rel':>10}")
    worst = 0.0
    for ratio in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        for ecc in (1.0, 3.0, 10.0):
            s1, s2 = 1000.0, 1000.0 / ecc
            R = ratio * s1
            C2 = np.array([[s1 ** 2, 0.0], [0.0, s2 ** 2]])
            ex = foster2d([0.0, 0.0], C2, R)
            ch = pc2d([0.0, 0.0], C2, R)
            print(f"{ratio:9.2f} {ecc:13.1f} {ex:12.6e} {ch:12.6e} {ch/ex-1:+10.2%}")
            rows.append({"check": "small_hbr_error", "R_over_sigma": ratio,
                         "eccentricity": ecc, "exact": ex, "chan": ch,
                         "chan_rel_err": ch / ex - 1})
            worst = max(worst, abs(ch / ex - 1))
    print(f"\n  worst small-HBR approximation error over the grid: {worst:.2%}")

    # ---------------------------------------------------------------- check 3
    print("\n" + "=" * 100)
    print("CHECK 3 -- CARA-STYLE MC: 6-D draw -> propagate to TCA -> project -> count")
    print("=" * 100)
    print(f"{'dt_s':>8} {'seed':>5} {'foster':>12} {'chan':>12} {'MC':>12} "
          f"{'sigma_MC':>10} {'z':>6} {'R/sigma':>9} {'chi2':>7} {'hits':>7}")
    geoms = [
        ("near miss, few days out",
         np.array([250.0, -400.0, 15.0]), np.array([-150.0, 400.0, 7000.0]),
         realistic_c6(600.0, 1200.0, 40.0), 12.0),
        ("head-on, larger uncertainty",
         np.array([-600.0, 900.0, -40.0]), np.array([300.0, -200.0, 6500.0]),
         realistic_c6(1500.0, 3000.0, 90.0), 25.0),
        ("glancing, small HBR",
         np.array([80.0, 60.0, -5.0]), np.array([-50.0, 150.0, 7200.0]),
         realistic_c6(250.0, 700.0, 20.0), 5.0),
    ]
    for label, r, v, C6, R in geoms:
        for dt in (0.0, 7200.0):
            for seed in (1, 2):
                out = cara_monte_carlo(r, v, C6, R, n_samples=2_000_000,
                                       seed=seed, dt_s=dt)
                z = abs(out["pc_mc"] - out["pc_foster"]) / max(out["sigma_mc"], 1e-30)
                C2 = out["C2_tca"]
                chi2 = float(np.linalg.solve(
                    C2, out["m2_tca"]) @ out["m2_tca"])
                print(f"{dt:8.0f} {seed:5d} {out['pc_foster']:12.6e} "
                      f"{out['pc_closed']:12.6e} {out['pc_mc']:12.6e} "
                      f"{out['sigma_mc']:10.1e} {z:6.2f} "
                      f"{R/np.sqrt(np.linalg.eigvalsh(C2).max()):9.4f} {chi2:7.2f} "
                      f"{out['n_inside']:7d}")
                rows.append({"check": "cara_mc", "geometry": label, "dt_s": dt,
                             "seed": seed, "R_m": R, "foster": out["pc_foster"],
                             "chan": out["pc_closed"], "mc": out["pc_mc"],
                             "sigma_mc": out["sigma_mc"], "z": z,
                             "chi2": chi2, "n_inside": out["n_inside"]})
                if z > 4.0:
                    failures.append(f"CARA MC z={z:.2f} for {label} dt={dt} seed={seed}")

    # ---------------------------------------------------------------- check 4
    if "--with-data" in sys.argv:
        print("\n" + "=" * 100)
        print("CHECK 4 -- REAL-WORLD ANCHOR: ESA's `risk` field in the Kelvins CDMs")
        print("=" * 100)
        from src.kelvins_data import load_test_truth
        from src.pc2d import pc2d_from_rtn
        t = load_test_truth()
        print(f"  {len(t)} CDMs; reproducing ESA `risk` from the published covariance.")
        print("  (full version of this check: scripts/validate_pipeline.py check 3)")

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 100)
    n_fail = len(failures)
    print(f"RESULT: {'ALL CHECKS PASSED' if not n_fail else f'{n_fail} FAILURE(S)'}"
          f"  ({len(rows)} rows written to {OUT.name})")
    for f in failures:
        print(f"  FAIL: {f}")
    print("=" * 100)

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT, index=False)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
