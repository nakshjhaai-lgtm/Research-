"""Layer 2: the interpretability scorecard.

CARA's Best Practices Handbook asks not only *what* Pc is but whether the method
supports *physical reasoning about why*.  This module turns that into a measured
scorecard rather than an opinion.

Each method is scored on five dimensions, computed on the real 183,761-CDM
corpus (the Monte Carlo and the 2-D quadrature on subsamples, since they are the
expensive paths).  Every score is in [0, 1] with an explicit definition, so a
reader can dispute a number by disputing its definition rather than a vibe.

  D1 DECOMPOSABILITY   Does the method's FORMULA literally factor into named
                       physical terms?  Structural, scored by inspection and
                       justified in notes/layer2_evaluation.md.

  D2 MONOTONICITY      Does Pc move in the physically correct direction with
                       every driver?  Scored by the sign of the PARTIAL
                       sensitivity fitted in the multivariate physical model.
                       Marginal correlations are confounded here and are not
                       used -- see the note in d2_monotonicity.

  D3 VALIDITY          Does the method expose when its own assumptions break?
                       Appendix N gives the small-HBR limit
                       R_lim = 4*sigma_x - min(R).  A method with no such
                       assumption scores 1; the closed form is scored by its
                       accuracy on CDMs inside the limit.

  D4 TRACEABILITY      Can the NUMBER be attributed to named physical causes on
                       the CDMs that drive an operational decision?  R^2 of the
                       two-factor model log10 Pc ~ log10(R/sqrt(det C)) + chi^2
                       on flagged CDMs (Pc >= 1e-5).

  D5 AUDITABILITY      Does it agree with an independent reference?  Agreement
                       with NASA CARA's own Pc2D_Foster on the published
                       Alfano cases, from results/cara_alfano_validation.csv.

Run:  python3 scripts/interpretability_scorecard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.cdm_covariance import (NEEDED, combined_position_covariance,
                                covariance_6x6, plane_basis, project)
from src.pc2d import pc2d_numeric
from src.foster2d import cara_monte_carlo
from scripts_helpers import pc2d_closed_batch, pc2d_exact_batch

OUT = Path(__file__).resolve().parents[1] / "results"
RAW = Path("/home/user/data/kelvins/Collision Avoidance Challenge - Dataset/"
           "raw_data/raw_data_2015-2019.gz")

COLS = ["event_id", "time_to_tca", "risk", "t_span", "c_span", "miss_distance",
        "relative_speed",
        "relative_position_r", "relative_position_t", "relative_position_n",
        "relative_velocity_r", "relative_velocity_t",
        "relative_velocity_n"] + NEEDED

METHODS = ("closed_form", "exact_N2", "quadrature_N1", "monte_carlo")

#: Theoretical partial sensitivities of log10 Pc, from the small-HBR form
#:     log10 Pc = 2 log10 R - log10(2 sqrt(det C)) - chi^2 / (2 ln 10)
#: so d log10 Pc / d log10(R/sqrt(det C)) = +2 and d log10 Pc / d chi^2 =
#: -1/(2 ln 10) = -0.2173.
THEORY = {"log10_R": 2.0, "log10_sqrt_det": -1.0,
          "chi2": -1.0 / (2.0 * np.log(10.0))}

#: Does the method's formula literally factor into named physical terms?
#: Structural property, scored by inspection.
D1_STRUCTURAL = {
    # Pc = R^2/(2 sqrt(det C)) * exp(-chi^2/2): a literal product of a radius
    # term, a covariance term and a Mahalanobis term.
    "closed_form": 1.0,
    # An integral.  The radius enters the limits and the erf arguments, the
    # covariance the widths; the decomposition is present in the integrand but
    # the result is not a closed product.
    "exact_N2": 0.5,
    "quadrature_N1": 0.5,
    # A sample proportion.  No analytic form at all -- which is exactly why it
    # is the trustworthy reference and the poor explainer.
    "monte_carlo": 0.0,
}


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #
def _fit(pc: np.ndarray, x: dict[str, np.ndarray]):
    """Least-squares fit of log10 Pc on the two physical drivers."""
    live = pc > 1e-250
    y = np.log10(pc[live])
    A = np.column_stack([np.ones(live.sum())] + [x[k][live] for k in THEORY])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(zip(THEORY, coef[1:])), live.sum()


def d2_monotonicity(pc: np.ndarray, x: dict[str, np.ndarray]) -> float:
    """Fraction of drivers whose PARTIAL sensitivity has the correct sign.

    Marginal correlations are confounded in this corpus: R/sigma and chi^2
    covary strongly across CDMs, so the raw Spearman of log(R/sigma) against Pc
    comes out *negative* even though theory says Pc rises with R at fixed chi^2
    (a wider covariance raises chi^2 and Pc at once).  The unconfounded test is
    the sign of the fitted coefficient, which is what this returns.
    """
    fitted, _ = _fit(pc, x)
    return float(np.mean([np.sign(fitted[k]) == np.sign(v)
                          for k, v in THEORY.items()]))


#: Which methods carry a small-HBR domain limit they can silently violate?
#: Exact quadrature and Monte Carlo have none, so they must not be penalised
#: for the Monte Carlo's sampling noise -- that is precision, not validity.
HAS_DOMAIN_LIMIT = {"closed_form": True, "exact_N2": False,
                    "quadrature_N1": False, "monte_carlo": False}


def d3_validity(method: str, pc: np.ndarray, exact: np.ndarray,
                past_limit: np.ndarray) -> float:
    """Does the method break silently outside its own domain?

    Appendix N's limit for the small-HBR / Gauss-Chebyshev form is
    R_lim = 4*sigma_x - min(R).  A method with no such assumption scores 1.0
    outright; the closed form is scored by its accuracy on CDMs *inside* the
    limit, i.e. by how much of its error is assumption violation rather than
    something else.
    """
    if not HAS_DOMAIN_LIMIT[method]:
        return 1.0
    inside = ~past_limit
    if not inside.any():
        return 0.0
    err = np.abs(np.log10(np.clip(pc[inside], 1e-300, None))
                 - np.log10(np.clip(exact[inside], 1e-300, None)))
    return float(np.mean(err <= 0.05))


def d4_traceability(pc: np.ndarray, x: dict[str, np.ndarray]) -> float:
    """R^2 of the two-factor physical model on FLAGGED CDMs (Pc >= 1e-5).

    Distinct from D1: D1 asks whether the *formula* factors, this asks whether
    the *number* is attributable to named physical causes on the CDMs that
    actually drive an operational decision.
    """
    flagged = pc >= 1e-5
    if flagged.sum() < 10:
        return float("nan")
    y = np.log10(pc[flagged])
    A = np.column_stack([np.ones(flagged.sum()),
                         x["log_R_over_sigma"][flagged], x["chi2"][flagged]])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(max(0.0, 1.0 - resid.var() / y.var()))


def d5_auditability(method: str) -> float:
    """Agreement with NASA CARA's own Pc2D_Foster on the published Alfano
    cases, from results/cara_alfano_validation.csv."""
    f = OUT / "cara_alfano_validation.csv"
    if not f.exists():
        return float("nan")
    d = pd.read_csv(f)
    col = {"closed_form": "ours_closed", "exact_N2": "ours_n2",
           "quadrature_N1": "ours_quadrature", "monte_carlo": "ours_mc"}[method]
    if col == "ours_mc":
        z = ((d[col] - d.cara_pc2d_foster).abs()
             / d.sigma_mc.clip(lower=1e-30))
        return float(np.mean(z < 4.0))
    rel = (d[col] / d.cara_pc2d_foster - 1).abs()
    return float(np.mean(rel <= 1e-3))


def _coef(pc: np.ndarray, x: dict[str, np.ndarray], key: str) -> float:
    fitted, _ = _fit(pc, x)
    return float(fitted[key])


# --------------------------------------------------------------------------- #
def main() -> int:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(RAW, usecols=COLS)
    df = df.sort_values(["event_id", "time_to_tca"], kind="mergesort")
    df["_rank"] = df.groupby("event_id").cumcount()
    truth = df[df["_rank"] == 0][["event_id", "risk"]].rename(
        columns={"risk": "r_true"})
    df = df.merge(truth, on="event_id", how="left", validate="many_to_one")
    nf = df[df["_rank"] > 0].copy()
    nf = nf[nf[NEEDED].notna().all(axis=1)]

    pos = nf[["relative_position_r", "relative_position_t",
              "relative_position_n"]].to_numpy()
    vel = nf[["relative_velocity_r", "relative_velocity_t",
              "relative_velocity_n"]].to_numpy()
    hbr = (0.5 * (nf.t_span + nf.c_span)).to_numpy()

    C3 = combined_position_covariance(nf)
    E = plane_basis(vel)
    C2 = project(C3, E)
    miss = np.einsum("nia,ni->na", E, pos)

    w = np.linalg.eigvalsh(C2)
    s_maj, s_min = np.sqrt(w[:, 1]), np.sqrt(w[:, 0])
    det = np.clip(w[:, 0] * w[:, 1], 1e-300, None)
    chi2 = np.einsum("ni,nij,nj->n", miss, np.linalg.inv(C2), miss)

    x = {
        "miss_m": np.linalg.norm(miss, axis=1),
        "chi2": chi2,
        "log10_R": np.log10(hbr),
        "log10_sqrt_det": 0.5 * np.log10(det),
        "log_R_over_sigma": np.log(hbr / np.sqrt(det)),
        # Appendix N's small-HBR / Gauss-Chebyshev validity limit
        "past_R_lim": hbr > (4.0 * s_maj - np.minimum(hbr, s_min)),
    }

    p_closed = pc2d_closed_batch(C2, miss, hbr)
    p_exact = pc2d_exact_batch(C2, miss, hbr)

    # 2-D quadrature on a subsample: it costs ~10x the 1-D form and every
    # scorecard dimension is a rank/distribution statistic, so a large
    # subsample suffices.  The subsample size is reported in the table.
    rng = np.random.default_rng(0)
    sub = rng.choice(len(nf), 4000, replace=False)
    p_quad = np.array([pc2d_numeric(miss[i], C2[i], hbr[i]) for i in sub])

    # Monte Carlo on a smaller subsample: 6-D sampling is the expensive path and
    # the scorecard only needs its distribution.  Uses the FULL assembled 6x6
    # covariance of both objects, which is what makes it a ground-truth-style
    # estimator rather than another 2-D method.
    msub = rng.choice(len(nf), 300, replace=False)
    C6 = (covariance_6x6(nf, "t") + covariance_6x6(nf, "c"))[msub]
    p_mc = np.array([
        cara_monte_carlo(pos[i], vel[i], C6[k], hbr[i],
                         n_samples=100_000, seed=int(i))["pc_mc"]
        for k, i in enumerate(msub)])

    pcs = {"closed_form": p_closed, "exact_N2": p_exact,
           "quadrature_N1": p_quad, "monte_carlo": p_mc}

    print("=" * 104)
    print("LAYER 2 INTERPRETABILITY SCORECARD")
    print("=" * 104)
    print(f"  CDMs scored              : {len(nf):,}")
    print(f"  CDMs past R_lim          : {100*x['past_R_lim'].mean():.2f}% "
          "(Appendix N small-HBR limit: R_lim = 4*sigma_x - min(R))")
    print(f"  median R                 : {np.median(hbr):.2f} m")
    print(f"  median sigma_major       : {np.median(s_maj):.1f} m")
    print(f"  CDMs flagged at 1e-5     : {int((p_exact >= 1e-5).sum()):,}")
    print("  -> R << sigma throughout this corpus, so the small-HBR assumption "
          "is\\n     valid HERE even though it fails catastrophically on CARA's "
          "Alfano cases\\n     (where R > sigma).  Both facts are true; they "
          "are different regimes.")

    rows = []
    for m in METHODS:
        pc = pcs[m]
        mask = (sub if m == "quadrature_N1"
                else msub if m == "monte_carlo" else np.arange(len(nf)))
        xs = {k: (v[mask] if isinstance(v, np.ndarray) else v)
              for k, v in x.items()}
        ex = p_exact[mask]
        rows.append({
            "method": m, "n_scored": int(len(mask)),
            "D1_decomposability": D1_STRUCTURAL[m],
            "D2_monotonicity": d2_monotonicity(pc, xs),
            "D3_validity": d3_validity(m, pc, ex, xs["past_R_lim"]),
            "D4_traceability": d4_traceability(pc, xs),
            "D5_auditability": d5_auditability(m),
            "coef_log10_R": _coef(pc, xs, "log10_R"),
            "coef_log10_sqrt_det": _coef(pc, xs, "log10_sqrt_det"),
            "coef_chi2": _coef(pc, xs, "chi2"),
        })

    sc = pd.DataFrame(rows)
    dcols = [c for c in sc.columns if c.startswith("D")]
    sc["mean_score"] = sc[dcols].mean(axis=1)
    sc.to_csv(OUT / "interpretability_scorecard.csv", index=False)

    print("\n" + "=" * 104)
    print(f"  {'method':<15}{'n':>8}{'D1':>7}{'D2':>7}{'D3':>7}{'D4':>7}"
          f"{'D5':>7}{'MEAN':>8}   fitted partials (theory +2.000, -1.000, -0.217)")
    print("-" * 104)
    for _, r in sc.iterrows():
        print(f"  {r.method:<15}{r.n_scored:8d}{r.D1_decomposability:7.3f}"
              f"{r.D2_monotonicity:7.3f}{r.D3_validity:7.3f}"
              f"{r.D4_traceability:7.3f}{r.D5_auditability:7.3f}"
              f"{r.mean_score:8.3f}   {r.coef_log10_R:+.3f}, "
              f"{r.coef_log10_sqrt_det:+.3f}, {r.coef_chi2:+.4f}")
    print(f"\nwrote results/interpretability_scorecard.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
