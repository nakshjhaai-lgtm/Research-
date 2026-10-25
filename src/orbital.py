"""Orbital mechanics primitives: Keplerian orbits, TLEs and SGP4.

Layer 1 of the learning path (Keplerian orbits -> TLE/SGP4 -> covariance ->
2D Pc).  Everything here is deliberately dependency-light (numpy + sgp4) so the
conventions stay visible.

Conventions
-----------
* Time scale: UTC, handled as ``datetime`` (naive) or ``astropy``-free.
* Frame: J2000 / ECI (Earth-centred inertial), right-handed.
* Angles: radians internally, degrees in the public API where noted.
* Distances: km.  Velocities: km/s.

References
----------
* D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.
  (2013) -- ch. 2 (two-body), ch. 9 (SGP4), ch. 11 (covariance/transformations).
* CelesTrak, "Frequently Asked Questions: Two-Line Element Set Format"
  (https://celestrak.org/NORAD/documentation/tle-fmt.php).
* CelesTrak, "SPACETRACK Report #3: Models for Propagation of the NORAD
  Element Sets" (Hoots & Roehrich, 1980) -- the SGP4 theory document.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------------------
# Constants (WGS-72 / EGM-96 as used by SGP4, and WGS-84 Earth radius)
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418  # km^3 / s^2  (geocentric gravitational constant)
R_EARTH = 6378.137  # km  (equatorial radius)
J2 = 1.08262668e-3  # second zonal harmonic
J3 = -2.53265648530e-6

DEG = math.pi / 180.0
RAD = 180.0 / math.pi

#: Number of seconds per day, used throughout for TLE epoch handling.
SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# Classical (Keplerian) elements
# ---------------------------------------------------------------------------
@dataclass
class ClassicalElements:
    """Classical orbital elements (osculating, two-body).

    ``a`` semi-major axis [km]; ``e`` eccentricity; ``i`` inclination [rad];
    ``raan`` right ascension of the ascending node [rad]; ``argp`` argument of
    perigee [rad]; ``nu`` true anomaly [rad].
    """

    a: float
    e: float
    i: float
    raan: float
    argp: float
    nu: float

    # -- derived quantities -------------------------------------------------
    @property
    def p(self) -> float:
        """Semi-latus rectum [km]."""
        return self.a * (1.0 - self.e ** 2)

    @property
    def rp(self) -> float:
        """Radius of perigee [km]."""
        return self.a * (1.0 - self.e)

    @property
    def ra(self) -> float:
        """Radius of apogee [km]."""
        return self.a * (1.0 + self.e)

    @property
    def mean_motion(self) -> float:
        """Two-body mean motion [rad/s]."""
        return math.sqrt(MU_EARTH / self.a ** 3)

    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return 2.0 * math.pi / self.mean_motion

    @property
    def specific_energy(self) -> float:
        """Specific orbital energy [km^2/s^2]."""
        return -MU_EARTH / (2.0 * self.a)

    @property
    def h(self) -> np.ndarray:
        """Specific angular momentum vector in ECI [km^2/s]."""
        r, v = state_from_elements(self)[:3], state_from_elements(self)[3:]
        return np.cross(r, v)

    def mean_anomaly(self) -> float:
        E = eccentric_anomaly_from_true(self.e, self.nu)
        return E - self.e * math.sin(E)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ClassicalElements(a={self.a:.4f} km, e={self.e:.7f}, "
            f"i={self.i * RAD:.4f} deg, raan={self.raan * RAD:.4f} deg, "
            f"argp={self.argp * RAD:.4f} deg, nu={self.nu * RAD:.4f} deg)"
        )




# ---------------------------------------------------------------------------
# Anomaly conversions
# ---------------------------------------------------------------------------
def eccentric_anomaly_from_mean(e: float, M: float, tol: float = 1e-12, itmax: int = 100) -> float:
    """Solve Kepler's equation  M = E - e sin E  (Newton-Raphson)."""
    M = (M + math.pi) % (2.0 * math.pi) - math.pi
    E = M if e < 0.8 else math.pi
    for _ in range(itmax):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break
    else:  # pragma: no cover - only for pathological e
        raise RuntimeError("Kepler's equation did not converge")
    return E


def true_anomaly_from_eccentric(e: float, E: float) -> float:
    """nu from E:  tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2).

    Written with atan2 so the quadrant is recovered correctly:
    nu = 2 * atan2( sqrt(1+e) sin(E/2), sqrt(1-e) cos(E/2) ).
    """
    return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                            math.sqrt(1.0 - e) * math.cos(E / 2.0))


def eccentric_anomaly_from_true(e: float, nu: float) -> float:
    return 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu / 2.0),
                            math.sqrt(1.0 + e) * math.cos(nu / 2.0))


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def perifocal_to_eci(el: ClassicalElements) -> np.ndarray:
    """3x3 matrix PQW -> ECI (columns are the perifocal basis in ECI)."""
    return rot_z(el.raan) @ rot_x(el.i) @ rot_z(el.argp)


# ---------------------------------------------------------------------------
# Two-body propagation
# ---------------------------------------------------------------------------
def state_from_elements(el: ClassicalElements) -> np.ndarray:
    """Classical elements -> 6-vector [r(3), v(3)] in ECI [km, km/s].

    Perifocal (PQW) coordinates, with E the eccentric anomaly:

        x = a (cos E - e)
        y = a sqrt(1 - e^2) sin E
        v = sqrt(mu a)/r * (-sin E, sqrt(1 - e^2) cos E, 0),  r = a(1 - e cos E)

    Note that x = a(cos E - e) is *not* r cos E; using r cos E silently
    corrupts the orbit for every eccentricity.
    """
    E = eccentric_anomaly_from_true(el.e, el.nu)
    r = el.a * (1.0 - el.e * math.cos(E))
    r_pqw = np.array([el.a * (math.cos(E) - el.e),
                      el.a * math.sqrt(1.0 - el.e ** 2) * math.sin(E), 0.0])
    fac = math.sqrt(MU_EARTH * el.a) / r
    v_pqw = np.array([-math.sin(E), math.sqrt(1.0 - el.e ** 2) * math.cos(E), 0.0]) * fac
    Q = perifocal_to_eci(el)
    return np.concatenate([Q @ r_pqw, Q @ v_pqw])


def elements_from_state(rv: np.ndarray, mu: float = MU_EARTH) -> ClassicalElements:
    """State vector [km, km/s] -> classical elements (Vallado alg. 9)."""
    r, v = rv[:3], rv[3:]
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n_vec = np.cross([0.0, 0.0, 1.0], h)  # node vector
    e_vec = ((vn ** 2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)
    energy = vn ** 2 / 2.0 - mu / rn
    a = -mu / (2.0 * energy) if abs(energy) > 1e-14 else float("inf")

    i = math.acos(np.clip(h[2] / hn, -1.0, 1.0))
    raan = math.atan2(n_vec[1], n_vec[0]) % (2.0 * math.pi)
    argp = math.acos(np.clip(np.dot(n_vec, e_vec) / (np.linalg.norm(n_vec) * e), -1.0, 1.0))
    if e_vec[2] < 0:
        argp = 2.0 * math.pi - argp
    nu = math.acos(np.clip(np.dot(e_vec, r) / (e * rn), -1.0, 1.0))
    if np.dot(r, v) < 0:
        nu = 2.0 * math.pi - nu
    return ClassicalElements(a=float(a), e=float(e), i=float(i), raan=float(raan),
                             argp=float(argp), nu=float(nu))


def propagate_kepler(state0: np.ndarray, dt: float, mu: float = MU_EARTH) -> np.ndarray:
    """Two-body (Keplerian) propagation of a state vector by ``dt`` seconds."""
    el = elements_from_state(state0, mu)
    n = el.mean_motion
    M0 = el.mean_anomaly()
    E = eccentric_anomaly_from_mean(el.e, M0 + n * dt)
    nu = true_anomaly_from_eccentric(el.e, E)
    return state_from_elements(ClassicalElements(el.a, el.e, el.i, el.raan, el.argp, nu))


# ---------------------------------------------------------------------------
# J2 secular drift (why a pure Keplerian propagator diverges from reality)
# ---------------------------------------------------------------------------
def j2_secular_rates(el: ClassicalElements) -> dict:
    """First-order J2 secular rates of RAAN, argp and mean anomaly [rad/s]."""
    n = el.mean_motion
    k = 1.5 * J2 * (R_EARTH / el.a) ** 2 * n / (1.0 - el.e ** 2) ** 2
    ci = math.cos(el.i)
    return {
        "raan_dot": -k * ci,
        "argp_dot": 0.75 * k * (5.0 * ci ** 2 - 1.0),
        "M_dot": 0.25 * k * math.sqrt(1.0 - el.e ** 2) * (3.0 * ci ** 2 - 1.0) + n,
    }


# ---------------------------------------------------------------------------
# Two-Line Element sets
# ---------------------------------------------------------------------------
@dataclass
class TLE:
    """A parsed NORAD two-line element set."""

    name: str = ""
    line1: str = ""
    line2: str = ""
    satnum: int = 0
    classification: str = "U"
    intl_designator: str = ""
    epoch_year: int = 0
    epoch_day: float = 0.0
    ndot_over2: float = 0.0  # rev/day^2 / 2
    nddot_over6: float = 0.0  # rev/day^3 / 6
    bstar: float = 0.0  # 1/earth_radii
    inclination_deg: float = 0.0
    raan_deg: float = 0.0
    eccentricity: float = 0.0  # decimal point implied
    argp_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    mean_motion_revday: float = 0.0
    rev_at_epoch: int = 0
    element_set_no: int = 0

    @property
    def epoch(self) -> datetime:
        """Epoch as a UTC ``datetime`` (TLE years are 2-digit, <57 => 20xx)."""
        year = self.epoch_year
        year += 2000 if year < 57 else 1900
        base = datetime(year, 1, 1)
        return base + timedelta(days=self.epoch_day - 1.0)

    @property
    def mean_motion_rad_s(self) -> float:
        return self.mean_motion_revday * 2.0 * math.pi / SECONDS_PER_DAY

    @property
    def semi_major_axis_km(self) -> float:
        """Semi-major axis recovered from the mean motion (Kepler's 3rd law)."""
        n = self.mean_motion_rad_s
        return (MU_EARTH / n ** 2) ** (1.0 / 3.0)

    def to_elements(self) -> ClassicalElements:
        """Mean elements -> classical elements.

        TLEs carry the *mean* anomaly, so it is converted to the true anomaly
        through Kepler's equation before building the state vector.
        """
        E = eccentric_anomaly_from_mean(self.eccentricity, self.mean_anomaly_deg * DEG)
        nu = true_anomaly_from_eccentric(self.eccentricity, E)
        return ClassicalElements(
            a=self.semi_major_axis_km,
            e=self.eccentricity,
            i=self.inclination_deg * DEG,
            raan=self.raan_deg * DEG,
            argp=self.argp_deg * DEG,
            nu=nu,
        )


def _parse_float(field: str, implied_decimal: bool = False) -> float:
    """Parse a TLE fixed-width float.

    Handles the implied exponent sign (' 12345-3' -> 0.12345e-3) and, for the
    drag/derivative fields, the implied leading decimal point ('-11606-4' ->
    -0.11606e-4).  When ``implied_decimal`` is set and the mantissa carries no
    explicit '.', a '0.' is prepended.
    """
    field = field.strip()
    if not field:
        return 0.0
    if field[0] in "+-":
        sign = -1.0 if field[0] == "-" else 1.0
        mant = field[1:].strip()
    else:
        sign, mant = 1.0, field
    if implied_decimal and "." not in mant:
        mant = "0." + mant
    if "e" in mant.lower():
        return sign * float(mant)
    if "-" in mant[1:]:
        idx = mant.rindex("-")
        return sign * float(mant[:idx] + "e" + mant[idx:])
    if "+" in mant[1:]:
        idx = mant.rindex("+")
        return sign * float(mant[:idx] + "e" + mant[idx:])
    return sign * float(mant or "0")


def parse_tle(text: str) -> TLE:
    """Parse a 2- or 3-line element set from text."""
    lines = [ln.rstrip("\n").rstrip("\r") for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) == 3:
        name, l1, l2 = lines
    elif len(lines) == 2:
        name, (l1, l2) = "", lines
    else:
        raise ValueError(f"expected 2 or 3 lines, got {len(lines)}")
    if not (l1.startswith("1 ") and l2.startswith("2 ")):
        raise ValueError("lines do not start with '1' and '2'")

    return TLE(
        name=name.strip(),
        line1=l1,
        line2=l2,
        satnum=int(l1[2:7]),
        classification=l1[7:8],
        intl_designator=l1[9:17].strip(),
        epoch_year=int(l1[18:20]),
        epoch_day=float(l1[20:32]),
        ndot_over2=_parse_float(l1[33:43], implied_decimal=True),
        nddot_over6=_parse_float(l1[44:52], implied_decimal=True),
        bstar=_parse_float(l1[53:61], implied_decimal=True),
        element_set_no=int(l1[64:68]),
        inclination_deg=float(l2[8:16]),
        raan_deg=float(l2[17:25]),
        eccentricity=float("0." + l2[26:33].strip()),
        argp_deg=float(l2[34:42]),
        mean_anomaly_deg=float(l2[43:51]),
        mean_motion_revday=float(l2[52:63]),
        rev_at_epoch=int(l2[63:68]),
    )


# ---------------------------------------------------------------------------
# SGP4 (Vallado's implementation, via the `sgp4` package)
# ---------------------------------------------------------------------------
def sgp4_propagate(tle: TLE, dt_seconds: float) -> np.ndarray:
    """Propagate a TLE with SGP4 and return [r(3), v(3)] in TEME [km, km/s].

    TEME (true equator, mean equinox) is the native frame of SGP4/TLE output;
    it differs from J2000 by a few tens of metres at most, but the distinction
    matters for covariance work -- see Vallado ch. 11 on frame conversions.
    """
    from sgp4.api import Satrec

    sat = Satrec.twoline2rv(tle.line1, tle.line2)
    jd, fr = sat.jdsatepoch, sat.jdsatepochF
    jd += int(dt_seconds // SECONDS_PER_DAY)
    fr += (dt_seconds % SECONDS_PER_DAY) / SECONDS_PER_DAY
    if fr >= 1.0:
        jd, fr = jd + 1.0, fr - 1.0
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        raise RuntimeError(f"SGP4 error code {e}")
    return np.array([*r, *v])


def sgp4_error_codes() -> dict:
    """Meaning of the SGP4 error codes returned by ``sgp4()``."""
    return {
        0: "ok",
        1: "mean elements, secular effect (mean motion too small)",
        2: "mean motion between 0 and 1 rev/day (sub-orbital decay)",
        3: "perturbed eccentricity < 0 or > 1",
        4: "semi-latus rectum < 0",
        5: "epoch elements are sub-orbital",
        6: "satellite has decayed",
    }
