"""Layer 2 -- calibration and precision-recall on the Kelvins data.

Answers one question: **does the predicted Pc mean anything?**  A Pc of 1e-4 is
a claim that 1 in 10,000 such encounters collides.  If the events forecast at
1e-4 actually collide 1 in 100 times, the forecast is wrong by two orders of
magnitude -- and no threshold tuning repairs that.

Every curve is reported **per threshold** (Layer 0): the binary label is defined
by the threshold, so a single "the model is well calibrated" line is as
meaningless as a single F2.

Two splits:
  * train -- ~15k events, used for the calibration statistics (sample size)
  * test  -- 2,167 events with hidden ground truth, used for the headline

Run:  python3 scripts/calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration import (brier_score, calibration_regression, log_loss,
                             platt_calibrate, pr_curve, reliability_curve,
                             reliability_summary)
from src.kelvins_data import latest_cdm_per_event, load_test, load_test_truth, load_train
from src.thresholds import PROVIDER_THRESHOLDS, COMPETITION_THRESHOLD, all_thresholds

OUT = Path(__file__).resolve().parents[1] / "results"


# ----------------------------------------------------------------------
def build_frames() -> dict[str, dict[str, np.ndarray]]:
    """Predictor / truth pairs for the train and test splits."""
    # --- test split: latest visible CDM vs the hidden final CDM
    test = load_test(usecols=["event_id", "time_to_tca", "risk"])
    truth = load_test_truth(usecols=["event_id", "true_risk"])
    latest = latest_cdm_per_event(test)[["event_id", "risk"]].rename(
        columns={"risk": "r_latest"})
    tf = latest.merge(truth, on="event_id", how="inner", validate="one_to_one")

    # --- train split: the last CDM is present in train_data.csv, so the
    #     minimum-time_to_tca row IS the ground truth and the second-smallest
    #     is the latest one a forecaster could see.
    tr = load_train(usecols=["event_id", "time_to_tca", "risk"])
    tr = tr.sort_values(["event_id", "time_to_tca"], kind="mergesort")
    tr["_rank"] = tr.groupby("event_id").cumcount()          # 0 = last CDM
    keep = tr["_rank"] <= 1                                  # need >= 2 CDMs
    t_last = tr[keep & (tr["_rank"] == 0)][["event_id", "risk"]].rename(
        columns={"risk": "r_true"})
    t_prev = tr[keep & (tr["_rank"] == 1)][["event_id", "risk"]].rename(
        columns={"risk": "r_latest"})
    trf = t_last.merge(t_prev, on="event_id", how="inner", validate="one_to_one")

    n = len(trf)
    print(f"  train split: {n:,} events with >= 2 CDMs "
          f"(last CDM = ground truth, previous CDM = prediction)")

    return {
        "test": {"true": tf["true_risk"].values, "lrp": tf["r_latest"].values,
                 "crp": np.full(len(tf), -5.0)},
        "train": {"true": trf["r_true"].values, "lrp": trf["r_latest"].values,
                  "crp": np.full(n, -5.0)},
    }


# ----------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("=" * 104)
    print("LAYER 2 -- CALIBRATION AND PRECISION-RECALL ON KELVINS")
    print("=" * 104)
    frames = build_frames()

    rel_rows, reg_rows, score_rows = [], [], []
    curves: dict[tuple, dict] = {}

    # Platt-recalibrated LRP: fit on train, applied to whichever split we are
    # reporting.  Never fit and evaluate on the same events.
    train_true, train_lrp = frames["train"]["true"], frames["train"]["lrp"]
    platt_cache: dict[float, tuple] = {}

    def platt_for(split: str, r_pred: np.ndarray, thr: float):
        """Return (recalibrated risks, boolean mask of the scored events)."""
        if split == "train":
            # honest within-train number: fit on the second half, score the
            # first, so the fit never sees the events it is scored on
            n = len(r_pred)
            half = n // 2
            mask = np.zeros(n, dtype=bool)
            mask[:half] = True
            vals = platt_calibrate(r_pred[half:], train_true[half:],
                                   r_pred[:half], thr)[0]
            return vals, mask
        if thr not in platt_cache:
            platt_cache[thr] = platt_calibrate(train_lrp, train_true, r_pred, thr)
        return platt_cache[thr][0], np.ones(len(r_pred), dtype=bool)

    for split, d in frames.items():
        r_true = d["true"]
        print(f"\n{'#' * 104}\n# SPLIT: {split}   ({len(r_true):,} events)\n{'#' * 104}")
        for T in all_thresholds():
            n_hr = int(np.sum(r_true >= T.log10_risk))
            print(f"\n--- {T.provider}  (Pc >= 1e{T.log10_risk:.4f}) ---"
                  f"   high-risk events: {n_hr:,}")
            if n_hr < 10:
                print("    (fewer than 10 positive events: reliability is not")
                print("     estimable at a useful resolution -- reported for")
                print("     completeness, not as a result)")

            # mask of events that get scored (train: the Platt fit is held out)
            _, mask = platt_for(split, d["lrp"], T.log10_risk)
            y_true = r_true[mask]

            platt_vals, _ = platt_for(split, d["lrp"], T.log10_risk)
            preds = [("LRP", d["lrp"][mask]), ("CRP", d["crp"][mask]),
                     ("LRP+Platt", platt_vals)]

            for name, r_pred in preds:
                if name == "CRP" and n_hr == 0:
                    continue
                if name == "LRP+Platt" and n_hr < 10:
                    continue   # not enough positives to fit a 2-parameter map
                cur = reliability_curve(y_true, r_pred, T.log10_risk, n_bins=8)
                reg = calibration_regression(y_true, r_pred, T.log10_risk)
                bs = brier_score(y_true, r_pred, T.log10_risk)
                ll = log_loss(y_true, r_pred, T.log10_risk)
                pr = pr_curve(y_true, r_pred, T.log10_risk, n_points=120)

                summ = reliability_summary(cur)
                auc_s = ("n/a (const)" if pr.get("degenerate")
                         else f"{pr['pr_auc']:.4f}")
                print(f"    {name:4s} ECE={summ['ece']:.4f}  Brier={bs:.3e}  "
                      f"log10loss={ll:.4f}  calib_slope={reg['slope']:+.2f} "
                      f"intercept={reg['intercept']:+.2f}  PR-AUC={auc_s}"
                      f"{'  [separation risk]' if reg['perfect_separation_risk'] else ''}")
                rel_rows.append({"split": split, "predictor": name,
                                 "n_scored": int(len(y_true)),
                                 **{k: v for k, v in summ.items()}})
                reg_rows.append({"split": split, "predictor": name, "n_scored": int(len(y_true)),
                                 "threshold": T.log10_risk,
                                 "slope": reg["slope"], "intercept": reg["intercept"],
                                 "slope_standardised": reg["slope_standardised"],
                                 "log_likelihood": reg["log_likelihood"],
                                 "perfect_separation_risk": reg["perfect_separation_risk"]})
                score_rows.append({"split": split, "predictor": name,
                                   "n_scored": int(len(y_true)),
                                   "threshold": T.log10_risk, "ece": summ["ece"],
                                   "brier": bs, "log10_loss": ll,
                                   "pr_auc": pr["pr_auc"]})
                if name == "LRP":
                    curves[(split, name, T.key)] = {"curve": cur, "pr": pr, "T": T}

    # ---------------------------------------------------------------- tables
    rel_df = pd.DataFrame(rel_rows)
    reg_df = pd.DataFrame(reg_rows)
    score_df = pd.DataFrame(score_rows)
    for df, name in ((rel_df, "calibration_reliability"),
                     (reg_df, "calibration_regression"),
                     (score_df, "calibration_scores")):
        df.to_csv(OUT / f"{name}.csv", index=False)

    print("\n" + "=" * 104)
    print("TABLE 1 -- CALIBRATION SCORES (per threshold, per split)")
    print("=" * 104)
    print(score_df.pivot_table(index=["threshold"], columns=["split", "predictor"],
                               values=["ece", "brier", "pr_auc"]).round(4).to_string())

    print("\n" + "=" * 104)
    print("TABLE 2 -- CALIBRATION REGRESSION (slope 1 / intercept 0 = calibrated)")
    print("=" * 104)
    print(reg_df.round(4).to_string(index=False))

    # ---------------------------------------------------------------- plots
    plot_reliability(curves, OUT / "calibration_reliability.png")
    plot_pr(curves, OUT / "calibration_pr_curves.png")

    print("\n" + "=" * 104)
    print(f"wrote calibration_reliability.png, calibration_pr_curves.png and 3 CSVs to {OUT}")
    print("=" * 104)
    return 0


def plot_reliability(curves, path: Path) -> None:
    """Reliability diagrams: predicted Pc vs observed high-risk frequency."""
    keys = [k for k in curves if k[1] == "LRP" and k[0] == "train"]
    if not keys:
        keys = [k for k in curves if k[1] == "LRP"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        cur = curves[k]["curve"]
        b = pd.DataFrame(cur["bins"])
        if b.empty:
            ax.set_visible(False)
            continue
        ax.plot([1e-7, 1], [1e-7, 1], "k--", lw=1, label="perfect")
        ax.plot(b["mean_pred_p"], b["observed_freq"], "o-", color="#c0392b",
                lw=1.8, ms=6, label="LRP")
        for _, r in b.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r["mean_pred_p"], r["observed_freq"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("predicted Pc (bin mean)")
        ax.set_ylabel(f"observed freq (Pc >= 1e{cur['threshold']:.1f})")
        ax.set_title(f"{curves[k]['T'].provider}\n1e{cur['threshold']:.2f}  "
                     f"ECE={cur['ece']:.3f}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Calibration / reliability of the LRP Pc forecast "
                 f"({keys[0][0]} split)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pr(curves, path: Path) -> None:
    """Precision-recall curves per threshold."""
    keys = [k for k in curves if k[1] == "LRP" and k[0] == "train"]
    if not keys:
        keys = [k for k in curves if k[1] == "LRP"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    cmap = plt.get_cmap("viridis")
    for i, k in enumerate(keys):
        pr = curves[k]["pr"]
        r, p = pr["recall"], pr["precision"]
        ok = ~np.isnan(r) & ~np.isnan(p)
        ax.plot(r[ok], p[ok], lw=1.8, color=cmap(i / max(len(keys) - 1, 1)),
                label=f"1e{pr['threshold']:.2f}  AUC={pr['pr_auc']:.3f} "
                      f"(n+={pr['n_high_risk']})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Precision-recall over a continuous decision threshold\n"
                 "(LRP predictor; each curve is one Layer 0 threshold)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
