"""Parser for Conjunction Data Messages (CDM).

Two front ends:

1. :func:`parse_cdm_kvl` -- the operational CCSDS / 18th SDS CDM as actually
   distributed: a KVL (keyword = value) ASCII message per CCSDS 508.0-B-1,
   one block per object (OBJECT_1 / OBJECT_2) carrying the OD parameters,
   the 6x6 covariance in RTN and the screening result.

2. :class:`CDMTable` -- adapter for the *anonymised tabular* CDM database
   released for the ESA Kelvins Collision Avoidance Challenge, where each row
   is a flattened CDM and the covariance has been reduced to sigmas plus
   correlation coefficients (the full matrices cannot be published).

References
----------
* CCSDS 508.0-B-1, *Conjunction Data Message*, blue book.
* NASA CARA Best Practices Handbook (2023) -- Appendices M and N on the CDM
  fields used for risk assessment.
* CelesTrak CDM documentation (https://celestrak.org/NORAD/documentation/cdm-formats.php).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# KVL (CCSDS) CDM
# ---------------------------------------------------------------------------


@dataclass
class CDMObject:
    """One of the two objects of a conjunction (OBJECT_1 or OBJECT_2)."""

    object_designator: str = ""
    catalog_name: str = ""
    object_name: str = ""
    international_designator: str = ""
    object_type: str = "PAYLOAD"
    ephemeris_name: str = "NONE"
    covariance_method: str = "CALC"
    maneuverable: str = "N/A"
    ref_frame: str = "ITRF"

    # orbit-determination context
    time_lastob_start: datetime | None = None
    time_lastob_end: datetime | None = None
    recommended_od_span: float | None = None  # days
    actual_od_span: float | None = None  # days
    obs_available: int | None = None
    obs_used: int | None = None
    residuals_accepted: float | None = None  # percent
    weighted_rms: float | None = None

    # physical
    area_pc: float | None = None  # m^2 (radar cross section)
    area_drag: float | None = None  # m^2
    area_srp: float | None = None  # m^2
    mass: float | None = None  # kg
    cd_area_over_mass: float | None = None  # m^2/kg
    cr_area_over_mass: float | None = None  # m^2/kg
    sedr: float | None = None  # W/kg
    geo_alt: float | None = None  # km, GEO objects only

    # state at TCA
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    x_dot: float = 0.0
    y_dot: float = 0.0
    z_dot: float = 0.0

    #: 6x6 covariance in RTN order: km^2 (position block), km^2/s^2 (velocity).
    covariance_rtn: np.ndarray | None = None

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def velocity(self) -> np.ndarray:
        return np.array([self.x_dot, self.y_dot, self.z_dot])

    def position_covariance(self) -> np.ndarray | None:
        """Upper-left 3x3 block of the RTN covariance [km^2]."""
        if self.covariance_rtn is None:
            return None
        return self.covariance_rtn[:3, :3]


@dataclass
class CDM:
    """A parsed Conjunction Data Message."""

    ccsds_cdm_vers: str = ""
    creation_date: datetime | None = None
    originator: str = ""
    message_id: str = ""
    tca: datetime | None = None
    miss_distance: float | None = None  # m
    relative_speed: float | None = None  # m/s
    relative_position: np.ndarray | None = None  # m
    relative_velocity: np.ndarray | None = None  # m/s
    start_screen_period: datetime | None = None
    stop_screen_period: datetime | None = None
    screen_volume_frame: str = ""
    screen_volume_shape: str = ""
    screen_volume_x: float | None = None
    screen_volume_y: float | None = None
    screen_volume_z: float | None = None
    screen_entry_time: datetime | None = None
    screen_exit_time: datetime | None = None
    collision_probability: float | None = None
    collision_probability_method: str = ""
    object1: CDMObject = field(default_factory=CDMObject)
    object2: CDMObject = field(default_factory=CDMObject)
    extra: dict = field(default_factory=dict)

    @property
    def combined_position_covariance(self) -> np.ndarray | None:
        """C = C1 + C2 in RTN [km^2], assuming uncorrelated object errors."""
        c1 = self.object1.position_covariance()
        c2 = self.object2.position_covariance()
        if c1 is None or c2 is None:
            return None
        return c1 + c2


def _parse_epoch(value: str) -> datetime | None:
    """CDM timestamps are ISO-8601 UTC, e.g. 2021-03-19T08:22:15.123456."""
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _to_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


#: Keys handled explicitly per object block; anything else lands in ``extra``.
_OBJECT_KEYS = {
    "OBJECT_DESIGNATOR", "CATALOG_NAME", "OBJECT_NAME", "INTERNATIONAL_DESIGNATOR",
    "OBJECT_TYPE", "EPHEMERIS_NAME", "COVARIANCE_METHOD", "MANEUVERABLE",
    "REF_FRAME", "TIME_LASTOB_START", "TIME_LASTOB_END", "RECOMMENDED_OD_SPAN",
    "ACTUAL_OD_SPAN", "OBS_AVAILABLE", "OBS_USED", "RESIDUALS_ACCEPTED",
    "WEIGHTED_RMS", "AREA_PC", "AREA_DRAG", "AREA_SRP", "MASS", "CD_AREA_OVER_MASS",
    "CR_AREA_OVER_MASS", "SEDR", "GEO_ALT", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT",
}

#: 6x6 RTN covariance, lower triangle as published (index pairs, RTN order).
_COV_KEYS = {
    "CR_R": (0, 0), "CT_R": (1, 0), "CT_T": (1, 1), "CN_R": (2, 0),
    "CN_T": (2, 1), "CN_N": (2, 2),
    "CRDOT_R": (3, 0), "CRDOT_T": (3, 1), "CRDOT_N": (3, 2), "CRDOT_RDOT": (3, 3),
    "CTDOT_R": (4, 0), "CTDOT_T": (4, 1), "CTDOT_N": (4, 2), "CTDOT_RDOT": (4, 3),
    "CTDOT_TDOT": (4, 4),
    "CNDOT_R": (5, 0), "CNDOT_T": (5, 1), "CNDOT_N": (5, 2), "CNDOT_RDOT": (5, 3),
    "CNDOT_TDOT": (5, 4), "CNDOT_NDOT": (5, 5),
}


def parse_cdm_kvl(text: str) -> CDM:
    """Parse one CCSDS KVL CDM message.

    Lines are ``KEY = value`` with an optional ``COMMENT`` after the value.
    ``OBJECT_1 = YES`` / ``OBJECT_2 = YES`` toggle which object the following
    OD / covariance keys belong to.
    """
    cdm = CDM()
    target: CDMObject | None = None
    cov: dict[tuple[int, int], float] = {}

    def _flush() -> None:
        """Attach the covariance accumulated so far to the current object."""
        if target is not None and cov:
            target.covariance_rtn = _assemble_covariance(cov)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.upper().startswith("COMMENT") or "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip().upper()
        value = rest.split("COMMENT")[0].strip()

        if key == "OBJECT_1":
            if value.upper() == "YES":
                _flush()
                target, cov = cdm.object1, {}
            continue
        if key == "OBJECT_2":
            if value.upper() == "YES":
                _flush()
                target, cov = cdm.object2, {}
            continue

        # --- object-scoped keys -------------------------------------------
        if target is not None and (key in _OBJECT_KEYS or key in _COV_KEYS):
            if key in _COV_KEYS:
                f = _to_float(value)
                if f is not None:
                    cov[_COV_KEYS[key]] = f
            else:
                _set_object_field(target, key, value)
            continue

        # --- header keys ---------------------------------------------------
        if key == "CCSDS_CDM_VERS":
            cdm.ccsds_cdm_vers = value
        elif key == "CREATION_DATE":
            cdm.creation_date = _parse_epoch(value)
        elif key == "ORIGINATOR":
            cdm.originator = value
        elif key == "MESSAGE_ID":
            cdm.message_id = value
        elif key == "TCA":
            cdm.tca = _parse_epoch(value)
        elif key == "MISS_DISTANCE":
            cdm.miss_distance = _to_float(value)
        elif key == "RELATIVE_SPEED":
            cdm.relative_speed = _to_float(value)
        elif key.startswith("RELATIVE_POSITION_"):
            cdm.relative_position = _vec3(cdm.relative_position, key[-1], value)
        elif key.startswith("RELATIVE_VELOCITY_"):
            cdm.relative_velocity = _vec3(cdm.relative_velocity, key[-1], value)
        elif key == "COLLISION_PROBABILITY":
            cdm.collision_probability = _to_float(value)
        elif key == "COLLISION_PROBABILITY_METHOD":
            cdm.collision_probability_method = value
        elif key in ("START_SCREEN_PERIOD", "STOP_SCREEN_PERIOD",
                     "SCREEN_ENTRY_TIME", "SCREEN_EXIT_TIME"):
            setattr(cdm, key.lower(), _parse_epoch(value))
        elif key.startswith("SCREEN_"):
            setattr(cdm, key.lower(), value)
        else:
            cdm.extra[key] = value

    _flush()
    return cdm


def _vec3(current: np.ndarray | None, axis: str, value: str) -> np.ndarray:
    idx = {"R": 0, "T": 1, "N": 2}[axis]
    f = _to_float(value) or 0.0
    arr = np.zeros(3) if current is None else current.copy()
    arr[idx] = f
    return arr


#: Object keys whose value is a string rather than a number.
_STR_OBJECT_KEYS = {
    "OBJECT_DESIGNATOR", "CATALOG_NAME", "OBJECT_NAME", "INTERNATIONAL_DESIGNATOR",
    "OBJECT_TYPE", "EPHEMERIS_NAME", "COVARIANCE_METHOD", "MANEUVERABLE", "REF_FRAME",
}


def _set_object_field(obj: CDMObject, key: str, value: str) -> None:
    if key in _STR_OBJECT_KEYS:
        setattr(obj, key.lower(), value)
    elif key in ("TIME_LASTOB_START", "TIME_LASTOB_END"):
        setattr(obj, key.lower(), _parse_epoch(value))
    elif key in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"):
        setattr(obj, key.lower(), _to_float(value) or 0.0)
    elif key in ("OBS_AVAILABLE", "OBS_USED"):
        setattr(obj, key.lower(), _to_int(value))
    else:
        setattr(obj, key.lower(), _to_float(value))


def _assemble_covariance(cov: dict[tuple[int, int], float]) -> np.ndarray:
    """Lower-triangle entries -> symmetric 6x6 matrix."""
    C = np.zeros((6, 6))
    for (i, j), v in cov.items():
        C[i, j] = v
        if i != j:
            C[j, i] = v
    return C


def parse_cdm_file(path: str | Path) -> CDM:
    return parse_cdm_kvl(Path(path).read_text())


def iter_cdm_files(directory: str | Path) -> Iterator[CDM]:
    """Yield parsed CDMs from every .cdm / .txt file in ``directory``."""
    d = Path(directory)
    for p in sorted(list(d.glob("*.cdm")) + list(d.glob("*.txt"))):
        yield parse_cdm_file(p)


# ---------------------------------------------------------------------------
# Kelvins tabular CDM adapter
# ---------------------------------------------------------------------------
class CDMTable:
    """Adapter over the flattened, anonymised CDM table of the Kelvins dataset.

    The published database keeps only a *reduced* description of the
    covariance: the three position sigmas and the three position-position
    correlation coefficients per object, in RTN.  :meth:`covariance_rtn`
    rebuilds the full 3x3 position covariance from those, which is all the
    2-D Pc calculation needs (the velocity block is not required).
    """

    #: RTN position sigma columns per object.
    SIGMA_COLS = {
        "t": ("t_sigma_r", "t_sigma_t", "t_sigma_n"),
        "c": ("c_sigma_r", "c_sigma_t", "c_sigma_n"),
    }
    #: RTN position-position correlation columns per object.
    CORR_COLS = {
        "t": (("t_ct_r", (1, 0)), ("t_cn_r", (2, 0)), ("t_cn_t", (2, 1))),
        "c": (("c_ct_r", (1, 0)), ("c_cn_r", (2, 0)), ("c_cn_t", (2, 1))),
    }
    REL_POS_COLS = ("relative_position_r", "relative_position_t", "relative_position_n")
    REL_VEL_COLS = ("relative_velocity_r", "relative_velocity_t", "relative_velocity_n")

    def __init__(self, row: dict):
        self.row = row

    @classmethod
    def from_series(cls, row) -> "CDMTable":
        return cls(dict(row))

    def covariance_rtn(self, obj: str) -> np.ndarray:
        """3x3 RTN position covariance [m^2] for 't' (target) or 'c' (chaser)."""
        sig = np.array([self.row.get(c, np.nan) for c in self.SIGMA_COLS[obj]], dtype=float)
        C = np.diag(sig ** 2)
        for col, (i, j) in self.CORR_COLS[obj]:
            rho = self.row.get(col, np.nan)
            if np.isfinite(rho):
                C[i, j] = C[j, i] = rho * sig[i] * sig[j]
        return C

    def combined_covariance_rtn(self) -> np.ndarray:
        """C = C_target + C_chaser [m^2]."""
        return self.covariance_rtn("t") + self.covariance_rtn("c")

    def relative_position_rtn(self) -> np.ndarray:
        return np.array([self.row.get(c, np.nan) for c in self.REL_POS_COLS], dtype=float)

    def relative_velocity_rtn(self) -> np.ndarray:
        return np.array([self.row.get(c, np.nan) for c in self.REL_VEL_COLS], dtype=float)

    def hard_body_radius(self) -> float:
        """Combined hard-body radius [m]: HBR = (span_target + span_chaser) / 2."""
        return 0.5 * (float(self.row["t_span"]) + float(self.row["c_span"]))
