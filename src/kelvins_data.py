"""Loader for the ESA Kelvins Spacecraft Collision Avoidance Challenge dataset.

Dataset: "Collision Avoidance Challenge dataset", Zenodo 10.5281/zenodo.4463683
(mirror of https://kelvins.esa.int/collision-avoidance-challenge/data/).

Layout inside the released zip::

    kelvins_competition_data/
        train_data.csv           CDMs of the TRAINING events, time_to_tca >= 0
                                 (the last CDM of each event is included -> it
                                 carries the label in the `risk` column)
        test_data.csv            CDMs of the TEST events with time_to_tca >= 2 days
                                 (the last CDM, i.e. the target, is removed)
        test_data_private.csv    the removed last CDM of every test event, with
                                 its risk renamed to `true_risk`  ->  GROUND TRUTH
    raw_data/
        raw_data_2015-2019.gz    all 199,082 CDMs of the original database
        raw_data_2015-2019.txt   column dictionary

Each row is one CDM (103 attributes). Rows are grouped by `event_id`; an event
is a time series of CDMs ordered by decreasing `time_to_tca` (days before the
time of closest approach). `risk` is the log10 of the self-reported collision
probability at that CDM's epoch.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

#: Root of the unpacked "Collision Avoidance Challenge - Dataset" folder.
DATA_ROOT = Path(
    os.environ.get(
        "KELVINS_DATA_ROOT",
        "/home/user/data/kelvins/Collision Avoidance Challenge - Dataset",
    )
)

COMP_DIR = DATA_ROOT / "kelvins_competition_data"
RAW_DIR = DATA_ROOT / "raw_data"

TRAIN_CSV = COMP_DIR / "train_data.csv"
TEST_CSV = COMP_DIR / "test_data.csv"
TEST_PRIVATE_CSV = COMP_DIR / "test_data_private.csv"
RAW_CSV_GZ = RAW_DIR / "raw_data_2015-2019.gz"

#: Columns whose values are log10 of a probability.
LOG_RISK_COLS = ("risk", "true_risk", "max_risk_estimate")

#: Covariance attributes are stored as sigmas and correlation coefficients.
RTN = ("r", "t", "n")


def _check(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Set KELVINS_DATA_ROOT to the unpacked dataset folder."
        )
    return path


def _freeze(cols):
    return None if cols is None else tuple(cols)


@lru_cache(maxsize=None)
def _cached_csv(path: Path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(_check(path), usecols=usecols)


def load_train(usecols=None) -> pd.DataFrame:
    """Training events (all CDMs, label available in the `risk` column)."""
    return _cached_csv(TRAIN_CSV, _freeze(usecols))


def load_test(usecols=None) -> pd.DataFrame:
    """Test-event inputs: only CDMs at least 2 days before TCA."""
    return _cached_csv(TEST_CSV, _freeze(usecols))


def load_test_truth(usecols=None) -> pd.DataFrame:
    """Ground truth: last CDM of each test event, risk in `true_risk`."""
    return _cached_csv(TEST_PRIVATE_CSV, _freeze(usecols))


def load_raw(usecols=None) -> pd.DataFrame:
    """The complete raw database (199,082 CDMs, 2015-2019)."""
    return _cached_csv(RAW_CSV_GZ, _freeze(usecols))


def latest_cdm_per_event(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per event: the CDM with the smallest ``time_to_tca``.

    This is the 'latest available CDM' (r_{-2} in the paper's notation) that the
    LRP baseline and every other forecaster is allowed to see.

    NOTE: rows are sorted by ASCENDING ``time_to_tca``, so the *first* row of
    each event is the CDM closest to TCA -- i.e. the one released last.
    ``head(1)`` (not ``tail(1)``) is therefore the correct selector; using
    tail(1) silently picks the earliest CDM of the series instead.
    """
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    df = df.sort_values(["event_id", "time_to_tca", "_row"], kind="mergesort")
    latest = df.groupby("event_id", as_index=False).head(1)
    return latest.drop(columns="_row").sort_values("event_id").reset_index(drop=True)


def build_lrp_frame() -> pd.DataFrame:
    """Join the latest available test CDM with the hidden ground truth.

    Returns one row per test event with columns:
    ``event_id, time_to_tca_last, r_latest, true_risk``.
    """
    test = load_test(usecols=["event_id", "time_to_tca", "risk"])
    truth = load_test_truth(usecols=["event_id", "time_to_tca", "true_risk"])

    latest = latest_cdm_per_event(test)[["event_id", "time_to_tca", "risk"]].rename(
        columns={"time_to_tca": "time_to_tca_last", "risk": "r_latest"}
    )
    truth = truth.rename(columns={"time_to_tca": "time_to_tca_truth"})
    out = latest.merge(truth, on="event_id", how="inner", validate="one_to_one")
    return out.sort_values("event_id").reset_index(drop=True)
