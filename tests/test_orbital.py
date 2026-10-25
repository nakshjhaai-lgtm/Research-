"""Regression tests for the orbital-mechanics layer.

Run:  python3 -m pytest tests/ -q      (or)      python3 tests/test_orbital.py

These caught two real bugs during development, both of which produced
plausible-looking but wrong orbits:

  1. ``true_anomaly_from_eccentric`` had the atan2 arguments swapped, mapping
     nu ~ 325 deg to ~215 deg.
  2. ``state_from_elements`` used ``r * cos(E)`` for the perifocal x instead of
     ``a (cos E - e)``, corrupting every eccentric orbit.

Both are guarded here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.orbital import (DEG, MU_EARTH, ClassicalElements, elements_from_state,
                         eccentric_anomaly_from_mean, eccentric_anomaly_from_true,
                         j2_secular_rates, parse_tle, propagate_kepler,
                         sgp4_propagate, state_from_elements,
                         true_anomaly_from_eccentric)

ISS_TLE = """1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"""


def _wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def test_anomaly_round_trip() -> None:
    """nu -> E -> nu must be exact for every eccentricity and quadrant."""
    for e in (0.0, 1e-6, 0.0006703, 0.1, 0.5, 0.9, 0.99):
        for nu_deg in (0, 1, 37, 90, 179, 180, 215, 300, 359):
            nu = nu_deg * DEG
            E = eccentric_anomaly_from_true(e, nu)
            nu2 = true_anomaly_from_eccentric(e, E)
            assert abs(_wrap(nu2 - nu)) < 1e-9, (e, nu_deg)


def test_kepler_equation() -> None:
    """M = E - e sin E must hold after the Newton solve (modulo 2*pi)."""
    for e in (0.0, 0.1, 0.5, 0.9, 0.99):
        for M in (0.0, 0.7, 3.0, 6.2, -5.9):
            E = eccentric_anomaly_from_mean(e, M)
            assert abs(_wrap((E - e * math.sin(E)) - M)) < 1e-10, (e, M)


def test_elements_state_round_trip() -> None:
    """Elements -> state -> elements must recover the same orbit."""
    for e in (1e-6, 0.0006703, 0.2, 0.7, 0.95):
        el = ClassicalElements(a=7000.0, e=e, i=51.0 * DEG,
                               raan=100 * DEG, argp=30 * DEG, nu=200 * DEG)
        back = elements_from_state(state_from_elements(el))
        assert abs(back.a - el.a) < 1e-8
        assert abs(back.e - el.e) < 1e-12
        assert abs(back.i - el.i) < 1e-12
        assert abs(_wrap(back.raan - el.raan)) < 1e-9
        assert abs(_wrap(back.argp - el.argp)) < 1e-9
        assert abs(_wrap(back.nu - el.nu)) < 1e-9


def test_kepler_propagation_conserves_integrals() -> None:
    """Two-body propagation must conserve specific energy and angular momentum."""
    el = ClassicalElements(a=7000.0, e=0.3, i=51 * DEG,
                           raan=100 * DEG, argp=30 * DEG, nu=200 * DEG)
    rv0 = state_from_elements(el)
    e0 = np.linalg.norm(rv0[3:]) ** 2 / 2 - MU_EARTH / np.linalg.norm(rv0[:3])
    h0 = np.linalg.norm(np.cross(rv0[:3], rv0[3:]))
    for dt in (600.0, 3600.0, 86400.0, 7 * 86400.0):
        rv = propagate_kepler(rv0, dt)
        e1 = np.linalg.norm(rv[3:]) ** 2 / 2 - MU_EARTH / np.linalg.norm(rv[:3])
        h1 = np.linalg.norm(np.cross(rv[:3], rv[3:]))
        assert abs(e1 - e0) < 1e-9, dt
        assert abs(h1 - h0) < 1e-9, dt


def test_perifocal_position_formula() -> None:
    """Guard the a(cos E - e) vs r cos E bug: check the perigee/apogee radii."""
    el = ClassicalElements(a=7000.0, e=0.3, i=0.0, raan=0.0, argp=0.0, nu=0.0)
    rv = state_from_elements(el)
    assert abs(np.linalg.norm(rv[:3]) - el.rp) < 1e-9          # nu = 0 -> perigee
    el_ap = ClassicalElements(a=7000.0, e=0.3, i=0.0, raan=0.0, argp=0.0, nu=math.pi)
    rv_ap = state_from_elements(el_ap)
    assert abs(np.linalg.norm(rv_ap[:3]) - el.ra) < 1e-9       # nu = pi -> apogee
    # at E = 90 deg the radius must equal a, which r*cos(E) would violate
    el_q = ClassicalElements(a=7000.0, e=0.3, i=0.0, raan=0.0, argp=0.0,
                             nu=true_anomaly_from_eccentric(0.3, math.pi / 2))
    rv_q = state_from_elements(el_q)
    assert abs(np.linalg.norm(rv_q[:3]) - 7000.0) < 1e-6


def test_tle_parsing() -> None:
    tle = parse_tle(ISS_TLE)
    assert tle.satnum == 25544
    assert tle.inclination_deg == 51.6416
    assert tle.eccentricity == 0.0006703          # implied decimal point
    assert tle.mean_motion_revday == 15.72125391
    assert tle.bstar == -0.11606e-4               # implied exponent
    assert tle.epoch.year == 2008 and tle.epoch.month == 9 and tle.epoch.day == 20
    assert abs(tle.semi_major_axis_km - 6730.9607) < 1e-3
    assert abs(2 * math.pi / tle.mean_motion_rad_s / 60 - 91.596) < 1e-3


def test_tle_mean_anomaly_is_not_true_anomaly() -> None:
    """nu recovered from a TLE's mean anomaly must differ from M for e > 0."""
    tle = parse_tle(ISS_TLE)
    nu = tle.to_elements().nu / DEG
    assert abs(((nu - tle.mean_anomaly_deg + 180) % 360) - 180) > 1e-3


def test_j2_rates_signs() -> None:
    """Prograde orbit: RAAN regresses, argp advances for i < 63.4 deg."""
    el = ClassicalElements(a=6778.0, e=0.0006703, i=51.6416 * DEG,
                           raan=0.0, argp=0.0, nu=0.0)
    r = j2_secular_rates(el)
    assert r["raan_dot"] < 0
    assert r["argp_dot"] > 0
    assert r["M_dot"] > el.mean_motion


def test_two_body_drifts_from_sgp4() -> None:
    """A Keplerian propagator must diverge from SGP4 over a day."""
    tle = parse_tle(ISS_TLE)
    rv0 = state_from_elements(tle.to_elements())
    sep = np.linalg.norm(propagate_kepler(rv0, 86400.0)[:3] - sgp4_propagate(tle, 86400.0)[:3])
    assert sep > 1.0, sep          # km


def test_sgp4_epoch_matches_tle_epoch() -> None:
    """SGP4 at dt = 0 must return the state at the TLE epoch."""
    tle = parse_tle(ISS_TLE)
    r = sgp4_propagate(tle, 0.0)[:3]
    assert abs(np.linalg.norm(r) - 6720.189) < 1e-2
    # and the state must lie on the orbit implied by the mean motion
    assert abs(np.linalg.norm(r) - tle.semi_major_axis_km) < 20.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  {fn.__name__:45s} OK")
    print(f"\n{len(fns)} orbital tests passed.")
