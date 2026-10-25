"""Layer 0 in practice: report every result as a threshold sweep.

Takes the LRP baseline (and a few no-ML reference predictors) and reports them
across the provider thresholds, plus the cross-provider disagreement of the
predictions and of the ground truth.

Run:  python3 scripts/threshold_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.kelvins_data import build_lrp_frame, load_test
from src.metrics import clip_low_risk
LRP_KEY = "LRP (latest CDM)"

from src.thresholds import (PROVIDER_THRESHOLDS, all_thresholds, disagreement,
                            format_sweep, operational_cost, sweep)


def build_predictors() -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Reference predictors over the visible CDMs, aligned to the LRP frame."""
    frame = build_lrp_frame()
    test = load_test(usecols=["event_id", "risk"])
    g = test.groupby("event_id")["risk"]

    agg = pd.DataFrame({
        "event_id": g.max().index,
        "max_visible": g.max().values,
        "mean_visible": g.mean().values,
    }).set_index("event_id")
    frame = frame.set_index("event_id").join(agg)

    # NOTE: raw (unclipped) predictions are passed to sweep(); the clip floor is
    # threshold-relative and is applied inside sweep(), per threshold.
    lrp = "LRP (latest CDM)"
    return {
        lrp: frame["r_latest"].values,
        "CRP (constant -5)": np.full(len(frame), -5.0),
        "MAX (max over visible CDMs)": frame["max_visible"].values,
        "MEAN (mean over visible CDMs)": frame["mean_visible"].values,
    }, frame


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    sweep_csv = out_dir / "threshold_sweep.csv"
    if sweep_csv.exists():
        sweep_csv.unlink()

    predictors, frame = build_predictors()
    r_true = frame["true_risk"].values

    # ------------------------------------------------------------------
    print("=" * 104)
    print("LAYER 0 -- THRESHOLD SWEEP  (no result below is a single number)")
    print("=" * 104)
    print("\nThreshold set (ascending in risk):")
    for T in all_thresholds():
        print(f"  {T.provider:44s} Pc >= 1e{T.log10_risk:7.4f}   [{T.role}]")
    print(f"\n  {len(PROVIDER_THRESHOLDS)} provider thresholds + the Kelvins competition")
    print("  threshold (1e-6), kept for continuity with the LRP reproduction.")

    # ------------------------------------------------------------------
    # 1. sweep for each predictor
    # ------------------------------------------------------------------
    sweeps = {}
    for name, r_hat in predictors.items():
        sweeps[name] = sweep(r_true, r_hat)

    for name, df in sweeps.items():
        print("\n" + "-" * 104)
        print(f"SWEEP: {name}")
        print("-" * 104)
        print(format_sweep(df))
        df.assign(predictor=name).to_csv(sweep_csv, mode="a",
                                         header=not sweep_csv.exists(), index=False)

    # ------------------------------------------------------------------
    # 2. ranking instability across thresholds
    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("RANKING INSTABILITY: F2 by predictor x threshold")
    print("=" * 104)
    pivot = pd.DataFrame({name: df.set_index("key")["F2"] for name, df in sweeps.items()})
    pivot = pivot.reindex([T.key for T in all_thresholds()])
    print(pivot.round(4).to_string())
    print("\nBest predictor at each threshold:")
    for k in pivot.index:
        row = pivot.loc[k].dropna()
        if len(row):
            print(f"  {k:16s} -> {row.idxmax()}  (F2 = {row.max():.4f})")
        else:
            print(f"  {k:16s} -> not evaluable (empty positive class)")

    # ------------------------------------------------------------------
    # 3. cross-provider disagreement
    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("CROSS-PROVIDER DISAGREEMENT (first-class metric)")
    print("=" * 104)

    for label, values in (("PREDICTED risk (LRP)", predictors["LRP (latest CDM)"]),
                          ("TRUE final risk", r_true)):
        d = disagreement(values)
        print(f"\n--- {label} ---")
        print(f"  events                        : {d['n_events']}")
        print(f"  overall disagreement rate     : {d['disagreement_rate'] * 100:.2f}%"
              "   (providers do not all reach the same decision)")
        print(f"  all providers act             : {d['all_act_rate'] * 100:.2f}%")
        print(f"  no provider acts              : {d['none_act_rate'] * 100:.2f}%")
        print(f"  max pairwise disagreement     : {d['max_pairwise_disagreement'] * 100:.2f}%"
              f"   ({d['most_divergent_pair'][0]} vs {d['most_divergent_pair'][1]})")
        print(f"  mean providers acting / event : {d['mean_providers_acting']:.3f}"
              f" of {d['n_providers']}")
        print(f"  histogram of #providers acting: {d['n_acting_histogram']}")
        print("\n  pairwise disagreement matrix:")
        print(d["pairwise"].round(4).to_string())

    # ------------------------------------------------------------------
    # 4. definition sensitivity of the problem itself
    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("DEFINITION SENSITIVITY: how the problem changes with the threshold")
    print("=" * 104)
    lrp_pred = predictors["LRP (latest CDM)"]
    rows = []
    for T in all_thresholds():
        n_hr_true = int(np.sum(r_true >= T.log10_risk))
        rows.append({
            "provider": T.provider,
            "log10_threshold": T.log10_risk,
            "true_high_risk_events": n_hr_true,
            "true_high_risk_share_%": 100.0 * n_hr_true / len(r_true),
            "events_LRP_would_flag": int(np.sum(lrp_pred >= T.log10_risk)),
        })
    sens = pd.DataFrame(rows)
    print(sens.round(4).to_string(index=False))
    lo, hi = sens.true_high_risk_events.min(), sens.true_high_risk_events.max()
    print(f"\n  -> the positive class ranges from {lo} to {hi} events across the sweep")
    print("     (a factor of "
          f"{hi / lo if lo else float('inf'):.0f}). 'Accuracy' and 'F2' are NOT comparable")
    print("     between rows of this table.")
    sens.to_csv(out_dir / "threshold_definition_sensitivity.csv", index=False)

    # ------------------------------------------------------------------
    # 5. operational cost
    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("OPERATIONAL COST (FP = wasted manoeuvre, FN = missed conjunction)")
    print("=" * 104)
    for cost_fn in (1.0, 10.0, 100.0, 1000.0):
        print(f"\n  cost(FN) / cost(FP) = {cost_fn:g}")
        tbl = operational_cost(sweeps[LRP_KEY],
                               cost_fp=1.0, cost_fn=cost_fn)
        print(tbl[["provider", "log10_threshold", "fp", "fn",
                   "expected_cost", "cost_per_event"]].round(3).to_string(index=False))
    print("\n  NOTE: the cost ratio is a POLICY input, passed in explicitly.")
    print("  It is not a measurement and must never be hidden inside a default.")

    # ------------------------------------------------------------------
    # 6. the margin knob
    # ------------------------------------------------------------------
    print("\n" + "=" * 104)
    print("THE MARGIN KNOB: trading FP for FN at fixed model output")
    print("=" * 104)
    print("  (LRP predictions; the *prediction* threshold is shifted while the")
    print("   provider's definition of high risk stays fixed. margin > 0 = more")
    print("   reluctant to call high risk.)")
    rows = []
    for T in PROVIDER_THRESHOLDS:
        for margin in (-0.5, 0.0, 0.5):
            df = sweep(r_true, lrp_pred, thresholds=[T], margin=margin)
            r = df.iloc[0]
            rows.append({"provider": T.provider, "log10_threshold": T.log10_risk,
                         "margin": margin, "fp": r["fp"], "fn": r["fn"],
                         "precision": r["precision"], "recall": r["recall"],
                         "F2": r["F2"]})
    mg = pd.DataFrame(rows)
    print(mg.round(4).to_string(index=False))
    mg.to_csv(out_dir / "threshold_margin_sweep.csv", index=False)

    print("\n  -> a single 'F2' number hides all of this. The sweep IS the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
