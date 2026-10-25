# Layer 2 — the evaluation core

Code: `src/calibration.py`, `scripts/calibration.py`,
`scripts/threshold_fpfn_199k.py`, `scripts/attribution_199k.py`.

Results:
- `results/calibration_reliability.csv`, `calibration_regression.csv`, `calibration_scores.csv`
- `results/calibration_reliability.png`, `calibration_pr_curves.png`
- `results/fpfn_thresholds_199k.csv`, `fpfn_by_leadtime_199k.csv`
- `results/attribution_199k.csv`

---

## 1. The headline table: threshold-explicit FP/FN across 199,082 CDMs

The competition evaluates only its 2,167-event test split at a single 1e-6
threshold. The raw 199k-CDM release is normally used for structural checks only.
Scoring **every** CDM against its own event's final risk makes the threshold
dependence visible at full sample size.

**Corpus:** 199,082 CDMs · 15,321 events · 183,761 non-final CDMs scored ·
risk range [−30, −1.44] log10 Pc · 41.4% of rows sit exactly on the −30 floor
(harmless — every threshold used is far above it).

### Table A — event level (last non-final CDM as the forecast, 15,096 events)

| threshold | log10T | n_pos | TP | FP | FN | **FP rate** | **FN rate** | F₂ |
|---|---|---|---|---|---|---|---|---|
| 1e-7 | −7.00 | 1133 | 907 | 2566 | 226 | **0.18377** | **0.19947** | 0.5665 |
| 1e-6.75 | −6.75 | 968 | 751 | 2226 | 217 | 0.15756 | 0.22417 | 0.5483 |
| **3e-7 (SpaceX)** | −6.523 | 804 | 608 | 1949 | 196 | **0.13637** | **0.24378** | 0.5266 |
| 1e-6 | −6.00 | 505 | 352 | 1326 | 153 | 0.09088 | 0.30297 | 0.4759 |
| 1e-5.75 | −5.75 | 395 | 253 | 1088 | 142 | 0.07401 | 0.35949 | 0.4331 |
| 1e-5.5 | −5.50 | 280 | 176 | 849 | 104 | 0.05730 | 0.37143 | 0.4103 |
| **1e-5 (ESA lower)** | −5.00 | 128 | 70 | 446 | 58 | **0.02980** | **0.45312** | 0.3405 |
| 1e-4.75 | −4.75 | 103 | 48 | 313 | 55 | 0.02088 | 0.53398 | 0.3105 |
| **1e-4 (ESA/DLR/CARA)** | −4.00 | 30 | 9 | 87 | 21 | **0.00577** | **0.70000** | 0.2083 |
| 1e-3.5 | −3.50 | 14 | 2 | 32 | 12 | 0.00212 | 0.85714 | 0.1111 |
| **1e-3 (JAXA)** | −3.00 | **7** | 2 | 9 | 5 | **0.00060** | **0.71429** | 0.2564 |

### Table B — CDM level (183,761 non-final CDMs)

| threshold | log10T | n_pos | TP | FP | FN | **FP rate** | **FN rate** | F₂ |
|---|---|---|---|---|---|---|---|---|
| 1e-7 | −7.00 | 8691 | 7611 | 18974 | 1080 | **0.10838** | **0.12427** | 0.6203 |
| **3e-7 (SpaceX)** | −6.523 | 6469 | 5465 | 14788 | 1004 | **0.08341** | **0.15520** | 0.5924 |
| 1e-6 | −6.00 | 4457 | 3625 | 10217 | 832 | 0.05698 | 0.18667 | 0.5723 |
| **1e-5 (ESA lower)** | −5.00 | 1448 | 1039 | 3715 | 409 | **0.02038** | **0.28246** | 0.4926 |
| **1e-4 (ESA/DLR/CARA)** | −4.00 | 375 | 177 | 858 | 198 | **0.00468** | **0.52800** | 0.3491 |
| **1e-3 (JAXA)** | −3.00 | 79 | 31 | 64 | 48 | **0.00035** | **0.60759** | 0.3771 |

**The trade-off, stated once:** moving the reaction threshold from 1e-3 to 1e-7
cuts the false-negative rate from **61% to 12%** and raises the false-positive
rate from **0.035% to 10.8%** — a 300× swing in FP for a 5× cut in FN. There is
no setting that is good on both. That is the whole content of the table, and it
is invisible in any single-threshold evaluation.

**Note the JAXA row.** At 1e-3 the event-level positive class is **7 events**.
The FP/FN rates there (0.00060 / 0.71429) are 2 significant figures on a sample
of seven. They are reported because the sweep must be complete, not because they
are measurements.

---

## 2. Calibration: does Pc mean anything?

**No — not in the range where most events live.**

Reliability diagrams (`results/calibration_reliability.png`), train split, LRP
predictor. At the SpaceX threshold the bins are:

| predicted Pc (bin mean) | observed freq of final Pc ≥ 3e-7 | n |
|---|---|---|
| ~1e-22 | ~1.5e-3 | 4040 |
| ~1e-14 | ~1.0e-3 | 808 |
| ~1e-11 | ~3.0e-3 | 808 |
| ~1e-6 | ~2.0e-1 | 808 |

The predicted probability spans **sixteen orders of magnitude** while the
observed frequency is **flat at ~1e-3**. The log-risk carries essentially no
information about the outcome until the very top bin. A forecast of 1e-22 is
wrong by nineteen orders of magnitude.

(The ~1e-22 values are partly the release's −30 floor, not real measurements —
41.4% of rows sit exactly there. The flat observed rate of ~1e-3 for those
events is nonetheless real and is the operative fact: those events *do* end up
high-risk about 1 in 1000 times, regardless of what the CDM claimed.)

### Calibration regression (slope 1 / intercept 0 = calibrated)

Test split, 2,167 events:

| threshold | LRP slope | LRP intercept | LRP+Platt slope | LRP log₁₀loss | CRP log₁₀loss | LRP+Platt log₁₀loss |
|---|---|---|---|---|---|---|
| 3e-7 | **0.34** | +1.44 | 0.94 | 0.5047 | 0.3876 | **0.0759** |
| 1e-6 | **0.34** | +1.25 | 0.76 | 0.4468 | 0.3461 | **0.0789** |
| 1e-5 | **0.28** | −0.36 | 0.43 | 0.1621 | 0.1292 | **0.0505** |
| 1e-4 | **0.32** | −1.40 | 0.34 | 0.0354 | 0.0392 | **0.0128** |
| 1e-3 | **0.20** | −3.49 | — (5 positives) | 0.0080 | 0.0115 | — |

Train split reproduces the same pattern (slopes 0.08–0.80).

**Three findings, all reproducible:**

1. **LRP is over-confident.** Its calibration slope is 0.2–0.8, far below the
   ideal 1.0, on both splits and at every threshold. The log-odds move far less
   than the prediction implies.

2. **A constant predictor beats it on log loss at every threshold with enough
   positives to tell.** At 3e-7, CRP (constant −5) scores 0.3876 against LRP's
   0.5047. LRP only wins at 1e-4 and below, where the positive class is too
   small for the comparison to mean much.

3. **A two-parameter Platt map fixes most of it.** Fit on the train split,
   applied to test: log loss drops **6.6×** at 3e-7 (0.5047 → 0.0759) and the
   calibration slope moves 0.34 → 0.94. **PR-AUC is unchanged** (0.6674 →
   0.6651), as it must be — Platt scaling is monotone. So LRP's *ranking* is
   fine; its *probabilities* are not.

> **Methodological note.** The calibration regression is L2-penalised, and that
> is essential rather than cosmetic. The unpenalised MLE diverges to ±∞ whenever
> the predictor perfectly separates the classes, which happens routinely here.
> An earlier version of this function returned slopes of ~1e10 from exactly
> that; those numbers are not estimates.

> **The Brier score is useless at these thresholds.** It is identical to three
> decimals for LRP and CRP in every cell, because the probabilities are all
> tiny relative to the base rate. Log loss discriminates; Brier does not.

---

## 3. Attribution: where does the error come from?

All three axes, full corpus, reported as threshold sweeps.

### Axis 1 — covariance staleness (CDM age). **This is the dominant one.**

CDM level, SpaceX threshold (3e-7):

| lead time | n_cdm | n_pos | TP | FP | FN | **FP rate** | **FN rate** |
|---|---|---|---|---|---|---|---|
| <0.5 d | 5,526 | 145 | 140 | 27 | 5 | **0.00502** | 0.03448 |
| 0.5–1 d | 14,942 | 427 | 393 | 146 | 34 | 0.01006 | 0.07963 |
| 1–2 d | 29,920 | 890 | 798 | 782 | 92 | 0.02694 | 0.10337 |
| 2–3 d | 28,870 | 918 | 769 | 1438 | 149 | 0.05145 | 0.16231 |
| 3–4 d | 27,715 | 961 | 798 | 2223 | 163 | 0.08309 | 0.16961 |
| 4–5 d | 26,647 | 992 | 822 | 2941 | 170 | 0.11464 | 0.17137 |
| 5–7 d | 50,141 | 2136 | 1745 | 7231 | 391 | **0.15063** | 0.18305 |

**The false-positive rate rises 30× from <0.5 days (0.50%) to 5–7 days (15.1%).**
The false-negative rate is far flatter (3.4% → 18.3%). Early CDMs cry wolf; late
CDMs miss.

At 1e-4 the same table gives FP 0.00109 → 0.00651 (6×) and FN 0.091 → 0.598
(6.6×): both rates degrade with age, FN faster.

### Axis 2 — geometry (relative speed). **A minor effect; the hypothesis is rejected.**

CDM level, SpaceX threshold (3e-7):

| v_rel (m/s) | n_cdm | n_pos | TP | FP | FN | **FP rate** | **FN rate** |
|---|---|---|---|---|---|---|---|
| <500 | 2,495 | 89 | 87 | 197 | 2 | 0.08188 | 0.02247 |
| 500–1k | 1,539 | 37 | 21 | 183 | 16 | **0.12184** | **0.43243** |
| 1k–2k | 3,806 | 202 | 161 | 307 | 41 | 0.08518 | 0.20297 |
| 2k–4k | 12,984 | 401 | 348 | 1270 | 53 | 0.10093 | 0.13217 |
| 4k–8k | 29,611 | 1168 | 957 | 2527 | 211 | 0.08884 | 0.18065 |
| >8k | 133,326 | 4572 | 3891 | 10304 | 681 | 0.08003 | 0.14895 |

The FP rate varies only 0.080 → 0.122 across a 34× range of relative speed, with
no monotone trend — **low relative speed is not a major FP driver in this
corpus**. The FN rate does vary substantially (0.022 at <500 m/s, peaking at
0.43 at 500–1k m/s), so slow encounters are a *recall* problem, not a precision
problem.

**But this must not be over-read, and an earlier draft of this note over-read
it.** CARA's Appendix N states that relative velocity "alone is not sufficient to
identify situations in which the conjunction plane Pc approximation will
miscarry", and gives a worked example of a ~3 m/s conjunction where the 2-D
conjunction-plane Pc is **wrong by a factor of 330** — the 3-D Nc method matches
the Monte Carlo there instead. So the effect is real and large *when it occurs*.

The reconciliation is frequency, not magnitude: only **2,495 of 183,761 CDMs
(1.4%)** in this corpus have v_rel < 500 m/s, so even a 330× error on those
cannot move the corpus-wide FP rate. The correct statement is:

> Low relative velocity does not drive false positives *in aggregate* in this
> corpus, because such encounters are 1.4% of it — not because the 2-D method
> handles them correctly. CARA documents errors up to 330× for exactly these
> conjunctions, and its four stated assumptions (rectilinear motion, Gaussian
> positions with negligible velocity uncertainty, temporal isolation,
> independence) are the right diagnostic, not v_rel itself.

### Axis 3 — HBR choice. **A large systematic effect that has nothing to do with skill.**

Done by rescaling ESA's own `risk` field — `risk' = risk + 2·log10(R'/R)` — which
is exact within the same small-HBR approximation ESA itself uses, so the
comparison stays on ESA's terms. Catalogue HBR median is 5.55 m (range
1.05–18.35 m), **not** 5 m.

CDM level:

| threshold | HBR variant | FP rate | FN rate | precision |
|---|---|---|---|---|
| 3e-7 | catalogue (t_span+c_span)/2 | 0.08341 | 0.15520 | 0.2698 |
| 3e-7 | **flat 5 m (Space-Track EVR)** | **0.10863** | **0.20575** | 0.2106 |
| 3e-7 | RCS-uncertainty (+30% rel σ) | 0.08595 | 0.14531 | 0.2662 |
| 1e-5 | catalogue | 0.02038 | 0.28246 | 0.2186 |
| 1e-5 | **flat 5 m** | **0.02757** | **0.38674** | 0.1502 |
| 1e-5 | RCS-uncertainty | 0.02158 | 0.27141 | 0.2114 |
| 1e-4 | catalogue | 0.00468 | 0.52800 | 0.1710 |
| 1e-4 | **flat 5 m** | 0.00524 | **0.72267** | 0.0977 |
| 1e-4 | RCS-uncertainty | 0.00508 | 0.50133 | 0.1673 |

**Switching the HBR convention moves the false-negative rate by ~35%** at every
threshold, with no change whatsoever in forecaster skill. Two operators running
identical CDMs through identical code will report systematically different FP/FN
rates purely because they disagree about how big the satellites are. RCS
uncertainty at ±30% moves rates by only 3–5%, so the *choice of convention*
dominates the *uncertainty within* one.

---

## 4. What this means

| Question | Answer from this layer |
|---|---|
| Does the predicted Pc mean anything? | **Not below ~1e-6.** Observed frequency is flat at ~1e-3 across sixteen decades of predicted Pc. |
| Is LRP good? | Good at ranking (PR-AUC 0.67), badly over-confident (slope 0.34), beaten on log loss by a constant. |
| Can it be fixed? | Yes — a 2-parameter Platt map cuts test log loss 6.6× with PR-AUC unchanged. |
| Where do false positives come from? | **CDM age**, overwhelmingly: FP rate rises 30× from <0.5 d to 5–7 d lead. |
| Where do false negatives come from? | CDM age and slow encounters; not precision but recall. |
| Does geometry drive FPs? | No — flat across a 34× range of v_rel. Hypothesis rejected. |
| Does the HBR matter? | **Yes, ~35% on the FN rate.** More than the RCS uncertainty within one convention. |
| Is a single F₂ meaningful? | No. It ranges 0.62 → 0.35 across 1e-7…1e-3 on the same data. |

---

## 5. Attribution axis 4 — method error (added later, closes the spec)

The Layer 2 specification names four attribution axes: HBR choice, covariance
staleness, method error, geometry. Axes 1–3 were covered above and in
`attribution_199k.py` / `fpfn_by_leadtime_199k.csv`. **Method error was the
missing one** and is now measured by `scripts/attribution_method.py`.

The raw CDMs carry the complete 6×6 RTN covariance element-by-element
(`src/cdm_covariance.py` assembles it). That makes it possible to compute Pc
*exactly* (Eq. N-2, no small-HBR approximation) on every CDM and compare three
definitions on identical inputs:

| definition | meaning |
|---|---|
| `esa_risk` | ESA's operational field |
| `closed_form` | the small-HBR closed form on **our** covariance |
| `exact_N2` | Eq. N-2 on **the same** covariance |

so `exact − closed` isolates method error alone.

### Two gates, both of which must pass

**G1 covariance reconstruction.** The assembled 6×6 reproduces the published
miss vector to **4.4e-05 relative** and ESA's own functional form to
**−0.009 dex median on near conjunctions** (χ² ≈ 3.6). On far conjunctions
(χ² ≈ 22.5) ESA's `risk` is *not* reproduced — the published block is evidently
not the block behind `risk` there. The 2-D convention beats the 3-D one by
~100×, so the element mapping is confirmed correct rather than accidentally
close. 5.6% of rows have missing covariance elements; 95% are complete.

**G2 vectorisation.** The batched Eq. N-2 agrees with the scalar `pc2d_1d`
reference to **1.4e-14** on 300 random CDMs, compared in log space because both
paths underflow identically near 1e-300.

### Result

Method error is the **smallest** of the four axes. On the validated subset
(35,429 CDMs where our covariance reproduces ESA within 0.25 dex), switching the
closed form for the exact integral changes:

| threshold | ΔFP | of total FP |
|---|---|---|
| 3×10⁻⁷ (SpaceX) | **+5** | 9,135 |
| 10⁻⁵ (ESA) | **+8** | 2,659 |
| 10⁻⁴ (ESA/DLR) | **+3** | 682 |
| 10⁻³ (JAXA) | **−1** | 53 |

Median method error is 1×10⁻⁴ dex, but the tail is heavy: **11.4% of CDMs are
off by more than 0.1 dex and 5.9% by more than 1.0 dex.**

### Why this reconciles with the catastrophic Alfano result

On this corpus `R << σ` throughout — median R = 5.55 m against median
σ_major = 2,019 m, and **0.00% of CDMs exceed Appendix N's `R_lim =
4σ_x − min(R)`**. On CARA's published Alfano cases `R > σ` and the same closed
form is wrong by up to 24 orders of magnitude (§ Layer 1). Both are true. The
small-HBR approximation is not "a bit off" — it is *valid exactly when R << σ*,
and this corpus happens to live there while CARA's stress cases do not.

**Do not generalise either regime.** The correct summary is: use Eq. N-2, which
has no domain limit at all, rather than reasoning about whether a given corpus
is inside the closed form's.

### Two traps found while building this

- The raw `mahalanobis_distance` column is **not** the χ² in ESA's formula. It is
  ~120× smaller than the 3-D relative-position χ². Compute the 2-D χ² from the
  miss vector and the encounter-plane covariance.
- The published `t_position_covariance_det` is the **3-D RTN** determinant, not
  the 2-D encounter-plane one. Feeding it into ESA's formula is off by ~10 dex.

---

## 6. Interpretability scorecard (added later, closes the spec)

CARA's handbook asks whether a method supports *physical reasoning about why*,
not just what Pc is. `scripts/interpretability_scorecard.py` measures that on
five dimensions, each with an explicit definition so a number can be disputed by
disputing its definition rather than a vibe.

| method | D1 decomp | D2 monot | D3 valid | D4 trace | D5 audit | **MEAN** |
|---|---|---|---|---|---|---|
| closed_form | 1.000 | 1.000 | 0.857 | 0.725 | 0.000 | 0.716 |
| **exact_N2** | 0.500 | 1.000 | 1.000 | 0.722 | 1.000 | **0.844** |
| quadrature_N1 | 0.500 | 1.000 | 1.000 | 0.683 | 1.000 | 0.837 |
| monte_carlo | 0.000 | 1.000 | 1.000 | 0.857 | 1.000 | 0.771 |

- **D1 decomposability** — does the *formula* factor into named physical terms?
  Structural. The closed form literally is
  `R²/(2√det C)·exp(−χ²/2)`; the integral forms do not factor; the MC has no
  analytic form at all.
- **D2 monotonicity** — does Pc move the right way with each driver? Scored on
  **partial** sensitivities. The fitted partials of log₁₀ Pc against
  log₁₀ R, log₁₀ √det and χ² are, for the closed form,
  **+2.000, −1.000, −0.2171** against theory **+2.000, −1.000, −0.217** — an
  exact match, which is itself the strongest evidence the physical model is
  right.
- **D3 validity** — does the method break silently outside its domain? Methods
  with no small-HBR assumption score 1; the closed form is scored by accuracy
  inside `R_lim`.
- **D4 traceability** — R² of the two-factor model on *flagged* CDMs
  (Pc ≥ 10⁻⁵), the ones that drive an operational decision.
- **D5 auditability** — agreement with NASA CARA's own `Pc2D_Foster` on the
  published Alfano cases.

**The scorecard's message:** analytic transparency and correctness are in
tension. The closed form is the most transparent and the least correct (D5 = 0).
Eq. N-2 is the best overall compromise and the recommended default.

**Trap:** D2 must use *partial* sensitivities. The marginal Spearman of
`log(R/σ)` against Pc is **−0.65**, the wrong sign, because R and χ² covary
across CDMs. Rewriting D2 as a marginal correlation silently inverts it.
