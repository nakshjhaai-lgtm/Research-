"""THE HEADLINE TABLE -- threshold-explicit FP/FN rates across all 199,082 raw CDMs.

Every CDM in the Kelvins release is treated as a forecast of its own event's
final risk (the risk on the last CDM of that event).  For each threshold we then
report the false-positive and false-negative rates.

Two views, because they answer different questions:

  EVENT level  one row per event, using the last CDM that is *not* the final
               one -- the honest forecasting setting, comparable to LRP.
  CDM level    every non-final CDM, stratified by lead time -- how the error
               rates evolve as the conjunction approaches.

Why this table has not been published: the competition evaluates only its
2,167-event test split at a single 1e-6 threshold.  The raw 199k-CDM release is
usually used for structural checks only.  Reporting FP/FN across 1e-7 ... 1e-3
over the whole corpus makes the threshold dependence visible at full sample
size.

Run:  python3 scripts/threshold_fpfn_199k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.kelvins_data import load_raw
from src.thresholds import PROVIDER_THRESHOLDS, all_thresholds

OUT = Path(__file__).resolve().parents[1] / "results"

#: the release's risk field is floored at -30; every threshold we use is well
#: above that, so a floored value is unambiguously "not high risk"
RISK_FLOOR = -30.0


def load_frame() -> pd.DataFrame:
    df = load_raw(usecols=["event_id", "time_to_tca", "risk"])
    df = df.sort_values(["event_id", "time_to_tca"], kind="mergesort").reset_index(drop=True)
    df["_rank"] = df.groupby("event_id").cumcount()      # 0 = final CDM
    # truth = risk on the final CDM (rank 0)
    truth = df[df["_rank"] == 0][["event_id", "risk"]].rename(columns={"risk": "r_true"})
    df = df.merge(truth, on="event_id", how="left", validate="many_to_one")
    df["is_final"] = df["_rank"] == 0
    return df


def rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    n_pos, n_neg = tp + fn, fp + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_pos": n_pos, "n_neg": n_neg,
            "fp_rate": fp / n_neg if n_neg else np.nan,
            "fn_rate": fn / n_pos if n_pos else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / n_pos if n_pos else np.nan,
            "f2": (5 * tp / (5 * tp + 4 * fn + fp)) if (5 * tp + 4 * fn + fp) else np.nan,
            "flagged_rate": (tp + fp) / (n_pos + n_neg) if n_pos + n_neg else np.nan}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    df = load_frame()
    print("=" * 112)
    print("THRESHOLD-EXPLICIT FP/FN RATES ACROSS THE FULL KELVINS RAW CORPUS")
    print("=" * 112)
    print(f"  CDMs           : {len(df):,}")
    print(f"  events         : {df.event_id.nunique():,}")
    print(f"  final CDMs     : {int(df.is_final.sum()):,}")
    print(f"  risk range     : [{df.risk.min():.2f}, {df.risk.max():.2f}] log10 Pc")
    print(f"  floored at -30 : {(df.risk == RISK_FLOOR).mean() * 100:.1f}% of rows "
          "(harmless: every threshold used is far above -30)")

    thresholds = list(PROVIDER_THRESHOLDS)
    # dense grid from 1e-7 to 1e-3 in 0.25 dex steps, deduplicated
    for lg in np.arange(-7.0, -2.99, 0.25):
        if not any(abs(lg - t.log10_risk) < 1e-9 for t in thresholds):
            thresholds.append(type(thresholds[0])(f"grid_1e{lg:.2f}", f"grid 1e{lg:.2f}",
                                                  float(lg), "sweep grid", role="grid"))
    thresholds = sorted(thresholds, key=lambda t: t.log10_risk)

    # ---------------------------------------------------------------- EVENT level
    ev = df[~df["is_final"]].copy()
    ev = ev.sort_values(["event_id", "time_to_tca"], kind="mergesort")
    last_visible = ev.groupby("event_id").tail(1)          # closest to TCA, not final
    print(f"\n  events with >= 2 CDMs (usable for forecasting): {last_visible.event_id.nunique():,}")

    rows = []
    print("\n" + "=" * 112)
    print("TABLE A -- EVENT LEVEL: last non-final CDM as the forecast (one row per event)")
    print("=" * 112)
    print(f"{'threshold':>26} {'log10T':>8} {'n_pos':>7} {'TP':>6} {'FP':>7} {'FN':>6} "
          f"{'FP rate':>9} {'FN rate':>9} {'prec':>7} {'rec':>7} {'F2':>7}")
    for T in thresholds:
        yt = last_visible["r_true"].values >= T.log10_risk
        yp = last_visible["risk"].values >= T.log10_risk
        r = rates(yt, yp)
        r.update({"view": "event", "provider": T.provider, "key": T.key,
                  "log10_threshold": T.log10_risk})
        rows.append(r)
        print(f"{T.provider:>26} {T.log10_risk:8.3f} {r['n_pos']:7d} {r['tp']:6d} "
              f"{r['fp']:7d} {r['fn']:6d} {r['fp_rate']:9.5f} {r['fn_rate']:9.5f} "
              f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f2']:7.4f}")

    # ---------------------------------------------------------------- CDM level
    nonfinal = df[~df["is_final"]]
    print("\n" + "=" * 112)
    print("TABLE B -- CDM LEVEL: every non-final CDM as a forecast "
          f"({len(nonfinal):,} CDMs)")
    print("=" * 112)
    print(f"{'threshold':>26} {'log10T':>8} {'n_pos':>7} {'TP':>7} {'FP':>7} {'FN':>7} "
          f"{'FP rate':>9} {'FN rate':>9} {'prec':>7} {'rec':>7} {'F2':>7}")
    for T in thresholds:
        yt = nonfinal["r_true"].values >= T.log10_risk
        yp = nonfinal["risk"].values >= T.log10_risk
        r = rates(yt, yp)
        r.update({"view": "cdm", "provider": T.provider, "key": T.key,
                  "log10_threshold": T.log10_risk})
        rows.append(r)
        print(f"{T.provider:>26} {T.log10_risk:8.3f} {r['n_pos']:7d} {r['tp']:7d} "
              f"{r['fp']:7d} {r['fn']:7d} {r['fp_rate']:9.5f} {r['fn_rate']:9.5f} "
              f"{r['precision']:7.4f} {r['recall']:7.4f} {r['f2']:7.4f}")

    # ------------------------------------------------- lead time (Layer 2 attribution)
    print("\n" + "=" * 112)
    print("TABLE C -- FP/FN BY LEAD TIME (CDM level, non-final CDMs)")
    print("=" * 112)
    bins = [0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 99.0]
    labels = ["<0.5 d", "0.5-1 d", "1-2 d", "2-3 d", "3-4 d", "4-5 d", "5-7 d", ">7 d"]
    nonfinal = nonfinal.assign(lead=pd.cut(nonfinal["time_to_tca"], bins=bins,
                                           labels=labels, right=False))
    rows_lt = []
    for T in PROVIDER_THRESHOLDS:
        print(f"\n  --- {T.provider} (Pc >= 1e{T.log10_risk:.3f}) ---")
        print(f"  {'lead time':>10} {'n_cdm':>8} {'n_pos':>7} {'TP':>6} {'FP':>7} "
              f"{'FN':>6} {'FP rate':>9} {'FN rate':>9}")
        for lab in labels:
            sub = nonfinal[nonfinal["lead"] == lab]
            if sub.empty:
                continue
            yt = sub["r_true"].values >= T.log10_risk
            yp = sub["risk"].values >= T.log10_risk
            r = rates(yt, yp)
            r.update({"view": "lead_time", "provider": T.provider, "key": T.key,
                      "log10_threshold": T.log10_risk, "lead_time": lab})
            rows_lt.append(r)
            print(f"  {lab:>10} {len(sub):8d} {r['n_pos']:7d} {r['tp']:6d} {r['fp']:7d} "
                  f"{r['fn']:6d} {r['fp_rate']:9.5f} {r['fn_rate']:9.5f}")

    # ---------------------------------------------------------------- write
    pd.DataFrame(rows).to_csv(OUT / "fpfn_thresholds_199k.csv", index=False)
    pd.DataFrame(rows_lt).to_csv(OUT / "fpfn_by_leadtime_199k.csv", index=False)
    print("\n" + "=" * 112)
    print("wrote fpfn_thresholds_199k.csv and fpfn_by_leadtime_199k.csv")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
