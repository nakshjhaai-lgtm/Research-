"""Layer 2 attribution over the full 199,082-CDM corpus.

Answers the Layer 2 question -- *where does the error come from?* -- on three
axes, all at full sample size and all reported as threshold sweeps (Layer 0):

  Table D  GEOMETRY    FP/FN by relative speed (low-relative-velocity encounters)
  Table E  HBR CHOICE  FP/FN under flat-5 m, catalogue and RCS-uncertainty HBR
  Table F  COMBINED    the FP/FN frontier across thresholds, all axes at once

Table E works by rescaling ESA's own `risk` field rather than recomputing Pc.
ESA's functional form is

    risk = log10( R^2 / (2 sqrt(det C2)) ) - chi^2 / (2 ln 10),  R = (t_span+c_span)/2

so under a different combined radius R' the probability transforms exactly as

    risk' = risk + 2 log10(R' / R)

which is exact within the same small-HBR approximation ESA itself uses.  This
keeps the comparison on ESA's own terms instead of mixing two Pc definitions.

Run:  python3 scripts/attribution_199k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.kelvins_data import load_raw
from src.thresholds import PROVIDER_THRESHOLDS

OUT = Path(__file__).resolve().parents[1] / "results"


def load() -> pd.DataFrame:
    df = load_raw(usecols=["event_id", "time_to_tca", "risk", "t_span", "c_span",
                           "relative_speed", "miss_distance"])
    df = df.sort_values(["event_id", "time_to_tca"], kind="mergesort").reset_index(drop=True)
    df["_rank"] = df.groupby("event_id").cumcount()
    truth = df[df["_rank"] == 0][["event_id", "risk"]].rename(columns={"risk": "r_true"})
    df = df.merge(truth, on="event_id", how="left", validate="many_to_one")
    df["is_final"] = df["_rank"] == 0
    df["R_cat"] = 0.5 * (df["t_span"] + df["c_span"])
    return df


def rates(y_true, y_pred) -> dict:
    tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred)); tn = int(np.sum(~y_true & ~y_pred))
    n_pos, n_neg = tp + fn, fp + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_pos": n_pos,
            "fp_rate": fp / n_neg if n_neg else np.nan,
            "fn_rate": fn / n_pos if n_pos else np.nan,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / n_pos if n_pos else np.nan}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    df = load()
    nf = df[~df["is_final"]].copy()
    print("=" * 108)
    print("LAYER 2 ATTRIBUTION -- 199,082-CDM CORPUS")
    print("=" * 108)
    print(f"  non-final CDMs scored: {len(nf):,}")
    print(f"  relative_speed range : {nf.relative_speed.min():.1f} - "
          f"{nf.relative_speed.max():.1f} m/s (median {nf.relative_speed.median():.1f})")
    print(f"  catalogue HBR range  : {nf.R_cat.min():.2f} - {nf.R_cat.max():.2f} m "
          f"(median {nf.R_cat.median():.2f})")

    rows = []

    # ---------------------------------------------------------------- Table D
    print("\n" + "=" * 108)
    print("TABLE D -- GEOMETRY: FP/FN by relative speed")
    print("=" * 108)
    sbins = [0, 500, 1000, 2000, 4000, 8000, 1e9]
    slabels = ["<500", "500-1k", "1k-2k", "2k-4k", "4k-8k", ">8k"]
    nf["vbin"] = pd.cut(nf["relative_speed"], bins=sbins, labels=slabels, right=False)
    for T in PROVIDER_THRESHOLDS:
        print(f"\n  --- {T.provider} (Pc >= 1e{T.log10_risk:.3f}) ---")
        print(f"  {'v_rel m/s':>10} {'n_cdm':>8} {'n_pos':>7} {'TP':>6} {'FP':>7} "
              f"{'FN':>6} {'FP rate':>9} {'FN rate':>9}")
        for lab in slabels:
            sub = nf[nf["vbin"] == lab]
            if sub.empty:
                continue
            yt = (sub["r_true"].values >= T.log10_risk)
            yp = (sub["risk"].values >= T.log10_risk)
            r = rates(yt, yp)
            r.update({"axis": "relative_speed", "provider": T.provider,
                      "key": T.key, "log10_threshold": T.log10_risk, "bin": lab})
            rows.append(r)
            print(f"  {lab:>10} {len(sub):8d} {r['n_pos']:7d} {r['tp']:6d} {r['fp']:7d} "
                  f"{r['fn']:6d} {r['fp_rate']:9.5f} {r['fn_rate']:9.5f}")

    # ---------------------------------------------------------------- Table E
    print("\n" + "=" * 108)
    print("TABLE E -- HBR CHOICE: FP/FN under different combined hard-body radii")
    print("=" * 108)
    # rescale ESA's own risk field: risk' = risk + 2 log10(R'/R)
    nf["R_flat5"] = 5.0
    rcs_rel = 0.30
    nf["R_rcs"] = nf["R_cat"] * np.sqrt(1.0 + rcs_rel ** 2)
    variants = {
        "catalogue (t_span+c_span)/2": nf["risk"],
        "flat 5 m (Space-Track EVR)": nf["risk"] + 2 * np.log10(5.0 / nf["R_cat"]),
        f"RCS-uncertainty (+{rcs_rel:.0%} rel sigma)":
            nf["risk"] + 2 * np.log10(nf["R_rcs"] / nf["R_cat"]),
    }
    for T in PROVIDER_THRESHOLDS:
        print(f"\n  --- {T.provider} (Pc >= 1e{T.log10_risk:.3f}) ---")
        print(f"  {'HBR variant':>36} {'n_pos':>7} {'TP':>6} {'FP':>7} {'FN':>6} "
              f"{'FP rate':>9} {'FN rate':>9} {'prec':>7}")
        for name, rp in variants.items():
            yt = (nf["r_true"].values >= T.log10_risk)
            yp = (rp.values >= T.log10_risk)
            r = rates(yt, yp)
            r.update({"axis": "hbr_variant", "provider": T.provider, "key": T.key,
                      "log10_threshold": T.log10_risk, "bin": name})
            rows.append(r)
            print(f"  {name:>36} {r['n_pos']:7d} {r['tp']:6d} {r['fp']:7d} "
                  f"{r['fn']:6d} {r['fp_rate']:9.5f} {r['fn_rate']:9.5f} "
                  f"{r['precision']:7.4f}")

    # ---------------------------------------------------------------- Table F
    print("\n" + "=" * 108)
    print("TABLE F -- THE FP/FN FRONTIER: CDM-level rates across 1e-7 ... 1e-3")
    print("=" * 108)
    grid = np.arange(-7.0, -2.99, 0.25)
    print(f"{'log10T':>8} {'n_pos':>7} {'FP rate':>9} {'FN rate':>9} {'flagged':>9} {'F2':>7}")
    for lg in grid:
        yt = (nf["r_true"].values >= lg)
        yp = (nf["risk"].values >= lg)
        tp = int(np.sum(yt & yp)); fp = int(np.sum(~yt & yp)); fn = int(np.sum(yt & ~yp))
        n_pos, n_neg = tp + fn, fp + int(np.sum(~yt & ~yp))
        fpr = fp / n_neg if n_neg else np.nan
        fnr = fn / n_pos if n_pos else np.nan
        d = 5 * tp
        f2 = d / (d + 4 * fn + fp) if (d + 4 * fn + fp) else np.nan
        print(f"{lg:8.2f} {n_pos:7d} {fpr:9.5f} {fnr:9.5f} {(tp+fp)/len(nf):9.5f} {f2:7.4f}")
        rows.append({"axis": "frontier", "provider": f"grid_1e{lg:.2f}",
                     "key": f"grid_{lg:.2f}", "log10_threshold": lg, "bin": "all",
                     "tp": tp, "fp": fp, "fn": fn, "n_pos": n_pos,
                     "fp_rate": fpr, "fn_rate": fnr,
                     "flagged_rate": (tp + fp) / len(nf), "f2": f2})

    pd.DataFrame(rows).to_csv(OUT / "attribution_199k.csv", index=False)
    print("\n" + "=" * 108)
    print(f"wrote attribution_199k.csv ({len(rows)} rows)")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
