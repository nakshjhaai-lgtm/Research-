"""End-to-end walkthrough of the learning path.

Runs the whole chain on one worked example:

  1. parse a TLE and recover the classical elements
  2. propagate with the two-body model and with SGP4, and show the drift
  3. build a synthetic conjunction: two objects, RTN covariances, a miss vector
  4. combine the covariances, project onto the encounter plane, compute the 2-D Pc
  5. show how Pc moves as the miss distance and the covariance shrink

Run:  python3 scripts/demo_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.covariance import (covariance_ellipse, encounter_plane_basis,
                            mahalanobis, project_to_plane, rtn_basis)
from src.orbital import (DEG, ClassicalElements, elements_from_state,
                         j2_secular_rates, parse_tle, propagate_kepler,
                         sgp4_propagate, state_from_elements)
from src.pc2d import pc2d, pc2d_max, pc2d_numeric

ISS_TLE = """1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""


def banner(txt: str) -> None:
    print(f"\n{'=' * 74}\n{txt}\n{'=' * 74}")


def step1_tle() -> None:
    banner("1. TLE -> classical elements")
    tle = parse_tle(ISS_TLE)
    print(f"satnum        : {tle.satnum}")
    print(f"epoch         : {tle.epoch} UTC")
    print(f"inclination   : {tle.inclination_deg:.4f} deg")
    print(f"eccentricity  : {tle.eccentricity:.7f}   (raw field '0006703')")
    print(f"mean motion   : {tle.mean_motion_revday:.8f} rev/day")
    print(f"a from n      : {tle.semi_major_axis_km:.4f} km   "
          f"(Kepler's 3rd law, a = (mu/n^2)^(1/3))")
    print(f"period        : {2 * np.pi / tle.mean_motion_rad_s / 60:.3f} min")

    el = tle.to_elements()
    print(f"true anomaly  : {el.nu / DEG:.4f} deg  (converted from the MEAN anomaly)")
    rv = state_from_elements(el)
    print(f"state         : r = {rv[:3].round(3)} km,  |r| = {np.linalg.norm(rv[:3]):.3f} km")
    print(f"                v = {rv[3:].round(6)} km/s")

    back = elements_from_state(rv)
    print(f"round trip    : a={back.a:.6f}  e={back.e:.9f}  i={back.i / DEG:.6f} deg")


def step2_propagation() -> None:
    banner("2. Two-body vs SGP4, and why J2 matters")
    tle = parse_tle(ISS_TLE)
    el = tle.to_elements()
    rv0 = state_from_elements(el)
    dt = 86400.0

    rk = propagate_kepler(rv0, dt)
    rs = sgp4_propagate(tle, dt)
    r0_sgp4 = sgp4_propagate(tle, 0.0)
    print(f"after {dt / 86400:.0f} day:")
    print(f"  two-body |r| = {np.linalg.norm(rk[:3]):10.4f} km")
    print(f"  SGP4     |r| = {np.linalg.norm(rs[:3]):10.4f} km")
    print(f"  separation   = {np.linalg.norm(rk[:3] - rs[:3]) * 1000:10.1f} m")
    print(f"\n  two sources of that error:")
    print(f"   (a) initial state: TLEs carry MEAN elements, SGP4 converts them")
    print(f"       to osculating internally. At the epoch the two states already")
    print(f"       differ by {np.linalg.norm(rv0[:3] - r0_sgp4[:3]) * 1000:.1f} m.")
    print(f"   (b) dynamics: two-body ignores J2 (dRAAN/dt = "
          f"{j2_secular_rates(el)['raan_dot'] / DEG * 86400:+.3f} deg/day) and drag.")
    print("  -> operational screening must use SGP4, not a Keplerian propagator.")

    rates = j2_secular_rates(el)
    print(f"\nJ2 secular rates: dRAAN/dt = {rates['raan_dot'] / DEG * 86400:+.5f} deg/day, "
          f"dargp/dt = {rates['argp_dot'] / DEG * 86400:+.5f} deg/day")


def step3_frames() -> None:
    banner("3. RTN frame and the encounter plane")
    r = np.array([7000.0, 1200.0, -900.0])
    v = np.array([-0.4, 7.1, 0.6])
    B = rtn_basis(r, v)
    print("RTN basis rows (R, T, N):")
    for name, vec in zip("RTN", B):
        print(f"  {name} = {vec.round(6)}")
    print(f"  orthonormal: {np.allclose(B @ B.T, np.eye(3))}")

    # a relative velocity mostly along-track, tilted
    v_rel = np.array([0.05, -7400.0, 320.0])
    P = encounter_plane_basis(v_rel)
    print("\nencounter plane basis (perpendicular to the relative velocity):")
    print(f"  e1 = {P[:, 0].round(6)}")
    print(f"  e2 = {P[:, 1].round(6)}")
    print(f"  e1 . v_rel = {np.dot(P[:, 0], v_rel):.3e}   "
          f"e2 . v_rel = {np.dot(P[:, 1], v_rel):.3e}")


def step4_pc() -> None:
    banner("4. A synthetic conjunction: covariance -> 2-D Pc")
    # target: a well-tracked satellite; chaser: a debris object
    sig_t = np.array([35.0, 2400.0, 22.0])        # m  (R, T, N)
    sig_c = np.array([180.0, 5200.0, 95.0])
    # build the 3x3 RTN covariance from sigmas and correlation coefficients
    Ct = np.diag(sig_t ** 2)
    Cc = np.diag(sig_c ** 2)
    for (i, j), rho in zip([(1, 0), (2, 0), (2, 1)], [0.10, 0.55, 0.20]):
        Ct[i, j] = Ct[j, i] = rho * sig_t[i] * sig_t[j]
    for (i, j), rho in zip([(1, 0), (2, 0), (2, 1)], [-0.30, 0.05, 0.60]):
        Cc[i, j] = Cc[j, i] = rho * sig_c[i] * sig_c[j]

    C3 = Ct + Cc
    print(f"target sigmas (R,T,N)     : {sig_t} m")
    print(f"chaser sigmas (R,T,N)     : {sig_c} m")
    print(f"combined C (RTN), km^2    :\n{(C3 / 1e6).round(6)}")
    print(f"  eigenvalues (m^2)       : {np.linalg.eigvalsh(C3).round(1)}")
    print(f"  det(C)                  : {np.linalg.det(C3):.6e} m^6")

    # a miss 4 km mostly along-track, relative velocity across-track
    m_rtn = np.array([120.0, 3950.0, -80.0])
    v_rel = np.array([-15.0, -60.0, 7400.0])
    R = 6.5  # m combined hard-body radius

    P = encounter_plane_basis(v_rel)
    C2 = project_to_plane(C3, P)
    m2 = P.T @ m_rtn
    chi2 = mahalanobis(m2, C2)
    p = pc2d(m2, C2, R)
    pn = pc2d_numeric(m2, C2, R)

    print(f"\nmiss vector (RTN)         : {m_rtn} m   |m| = {np.linalg.norm(m_rtn):.1f} m")
    print(f"relative speed            : {np.linalg.norm(v_rel):.1f} m/s")
    print(f"hard-body radius          : {R} m")
    print(f"projected miss (plane)    : {m2.round(2)} m")
    print(f"chi^2 (encounter plane)   : {chi2:.4f}")
    print(f"sigma_1 * sigma_2         : {np.sqrt(np.linalg.det(C2)):.1f} m^2")
    print(f"\n  Pc (closed form)        : {p:.6e}   log10 = {np.log10(p):.4f}")
    print(f"  Pc (exact quadrature)   : {pn:.6e}   log10 = {np.log10(pn):.4f}")
    print(f"  Pc_max = R^2/(2 s1 s2)  : {pc2d_max(R, C2):.6e}")
    print(f"  closed/exact ratio      : {p / pn:.6f}")


def step5_sensitivity() -> None:
    banner("5. How Pc responds to the miss and to shrinking covariance")
    Ct = np.diag(np.array([35.0, 2400.0, 22.0]) ** 2)
    Cc = np.diag(np.array([180.0, 5200.0, 95.0]) ** 2)
    C3 = Ct + Cc
    v_rel = np.array([-15.0, -60.0, 7400.0])
    P = encounter_plane_basis(v_rel)
    C2 = project_to_plane(C3, P)
    R = 6.5

    print("Pc as the miss distance shrinks (covariance fixed):")
    for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        m_rtn = np.array([120.0, 3950.0, -80.0]) * scale
        m2 = P.T @ m_rtn
        p = pc2d(m2, C2, R)
        lg = f"{np.log10(p):7.3f}" if p > 0 else "  -inf"
        print(f"  |miss| = {np.linalg.norm(m_rtn):9.1f} m   "
              f"chi2 = {mahalanobis(m2, C2):8.4f}   Pc = {p:.6e}   log10 = {lg}")

    print("\nPc as the covariance shrinks (miss fixed at 3950 m):")
    m2 = P.T @ np.array([120.0, 3950.0, -80.0])
    for s in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        C2s = C2 * s * s
        p = pc2d(m2, C2s, R)
        lg = f"{np.log10(p):7.3f}" if p > 0 else "  -inf"
        print(f"  sigma scale = {s:5.2f}   chi2 = {mahalanobis(m2, C2s):9.3f}   "
              f"Pc = {p:.6e}   log10 = {lg}")
    print("\n  This is the whole story of a conjunction time series: as the")
    print("  objects approach, both the miss and the covariance shrink, and")
    print("  the reported risk moves by many orders of magnitude.")


if __name__ == "__main__":
    step1_tle()
    step2_propagation()
    step3_frames()
    step4_pc()
    step5_sensitivity()
