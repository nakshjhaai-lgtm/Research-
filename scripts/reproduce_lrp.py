"""Reproduce the Latest Risk Prediction (LRP) baseline of the Kelvins challenge.

LRP (Eq. 6 of Uriot et al. 2021) is the *clipped naive forecast*: the final risk
of a conjunction is predicted with the risk value reported in the latest CDM
that is still visible to the competitor (i.e. the newest CDM at least 2 days
before TCA), clipped at -6.001 so that low-risk predictions sit just below the
classification threshold:

    r_hat_i = r_i        if r_i > -6
            = -6.001     otherwise

Published reference values on the complete test set (Table 3 of the paper):

    LRP baseline | Score L = 0.694 | MSE_HR = 0.513 | F2 = 0.739

Gate for this project: F2 within +/- 0.02 of 0.739.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.kelvins_data import build_lrp_frame, load_test, load_test_truth
from src.metrics import HIGH_RISK_THRESHOLD, RISK_CLIP_FLOOR, challenge_metrics, clip_low_risk

TARGET_F2 = 0.739
TOL = 0.02


def sanity_checks() -> None:
    """Confirm the split matches the three eligibility rules of Sec. 4.2."""
    test = load_test(usecols=["event_id", "time_to_tca"])
    truth = load_test_truth(usecols=["event_id", "time_to_tca", "true_risk"])

    per_ev = test.groupby("event_id")["time_to_tca"]
    print("--- structural sanity checks ---")
    print(f"test CDMs                : {len(test):>7d}")
    print(f"test events              : {test.event_id.nunique():>7d}")
    print(f"truth rows (last CDMs)   : {len(truth):>7d}")
    print(f"event_id sets identical  : {set(test.event_id) == set(truth.event_id)}")
    print(f"min time_to_tca in input : {test.time_to_tca.min():.6f} d  (rule iii: >= 2)")
    print(f"max time_to_tca in truth : {truth.time_to_tca.max():.6f} d  (rule ii: < 1)")
    print(f"CDMs per event  min/mean/max: {per_ev.count().min():.0f} / "
          f"{per_ev.count().mean():.2f} / {per_ev.count().max():.0f}")
    print(f"high-risk truth events   : {(truth.true_risk >= HIGH_RISK_THRESHOLD).sum()}")
    print(f"events w/ >= 2 input CDMs: {(per_ev.count() >= 2).sum()}")


def main() -> int:
    sanity_checks()

    df = build_lrp_frame()
    print(f"\njoined frame: {len(df)} events x {list(df.columns)}")

    r_true = df["true_risk"].to_numpy()
    r_hat_raw = df["r_latest"].to_numpy()
    r_hat = clip_low_risk(r_hat_raw)

    res = challenge_metrics(r_true, r_hat)
    print("\n--- LRP baseline (clipped naive forecast) ---")
    print(res.report())

    # Cross-check the classification half against sklearn.
    try:
        from sklearn.metrics import fbeta_score
        skl = fbeta_score(r_true >= HIGH_RISK_THRESHOLD, r_hat >= HIGH_RISK_THRESHOLD, beta=2)
        print(f"\nsklearn fbeta_score(beta=2) cross-check: {skl:.6f} "
              f"(delta {abs(skl - res.F2):.2e})")
    except ImportError:
        pass

    # Unclipped variant, for diagnostics only (not the paper's definition).
    res_noclip = challenge_metrics(r_true, r_hat_raw)
    print(f"\n[diagnostic] without the -6.001 clip: F2={res_noclip.F2:.6f} "
          f"MSE_HR={res_noclip.MSE_HR:.6f} L={res_noclip.L:.6f}")

    # Independent cross-check: the Constant Risk Prediction baseline (Sec. 4.4),
    # r_hat = -5 for every event, is documented to score L = 2.5.
    res_crp = challenge_metrics(r_true, np.full_like(r_true, -5.0))
    print(f"[cross-check] CRP baseline (r_hat = -5): L = {res_crp.L:.4f} "
          f"(paper: 2.5), F2 = {res_crp.F2:.4f}, MSE_HR = {res_crp.MSE_HR:.4f}")

    delta = res.F2 - TARGET_F2
    verdict = "PASS" if abs(delta) <= TOL else "FAIL"
    print(f"\n=== GATE: F2 = {res.F2:.4f} vs target {TARGET_F2} "
          f"(delta {delta:+.4f}, tol +/-{TOL}) -> {verdict} ===")

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)
    df.assign(r_hat_lrp=r_hat).to_csv(out_dir / "lrp_predictions.csv", index=False)
    pd.DataFrame([res.as_dict()]).to_csv(out_dir / "lrp_metrics.csv", index=False)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
