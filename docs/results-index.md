# Results index

Every number quoted in the write-up comes from one of these files, and every file
is produced by a script that exits 0. Nothing in `results/` was edited by hand.
If a fresh run disagrees with a value in the documentation, the fresh run wins.

Paths are relative to the repository root. Each entry says what the file answers,
which script writes it, and what the columns mean.

---

## The gate and the predictions

### `results/lrp_metrics.csv`
One row. The baseline score on the 2,167-event test split.
Written by `scripts/reproduce_lrp.py`.

| Column | Meaning |
|---|---|
| `n_events` | test events with a usable latest CDM (2,167) |
| `n_high_risk` | events with `true_risk ≥ −6` (150) |
| `F2` | competition F-score, β = 2 |
| `precision`, `recall` | conventional definitions, at 10⁻⁶ |
| `tp`, `fp`, `fn`, `tn` | 115 / 63 / 35 / 1954 |
| `MSE_HR` | mean squared error over high-risk events only |
| `L` | challenge loss, `MSE_HR / F2` |

### `results/lrp_predictions.csv`
2,167 rows, one per test event. The forecast behind the row above, event by
event, and the file to open when a number needs checking.
Written by `scripts/reproduce_lrp.py`.

| Column | Meaning |
|---|---|
| `event_id` | anonymous event identifier |
| `time_to_tca_last` | days to closest approach of the CDM used as the forecast |
| `r_latest` | log₁₀ Pc in that CDM |
| `time_to_tca_truth` | days to closest approach in the ground-truth row |
| `true_risk` | log₁₀ Pc of the final CDM (the label) |
| `r_hat_lrp` | the clipped prediction, `r_latest` or −6.001 |

### `results/pc2d_validation.csv`
Per-CDM comparison of our closed-form Pc against the `risk` column ESA shipped.
Written by `scripts/validate_pipeline.py`.

| Column | Meaning |
|---|---|
| `risk` | ESA's reported log₁₀ Pc |
| `pc2d_log10` | our closed-form Pc from the published covariance |
| `chi2` | 2-D Mahalanobis distance in the encounter plane |
| `hbr_m`, `sigma_prod` | hard-body radius, √det of the encounter-plane covariance |
| `miss_distance_m`, `rel_pos_norm_m` | miss distance, for cross-checking |
| `max_risk_estimate`, `max_risk_scaling` | ESA's independent maxima fields |
| `max_est_implied`, `scaling_implied` | what those fields imply, back-computed |
| `det_t_ratio`, `det_c_ratio` | our determinant over the published one (1.000000) |
| `err_log10` | our Pc minus ESA's |

---

## Layer 0: thresholds

### `results/threshold_sweep.csv`
20 rows. The LRP baseline scored under five provider policies plus intermediate
margins, on the competition test split.
Written by `scripts/threshold_sweep.py`.

| Column | Meaning |
|---|---|
| `key`, `provider` | policy identifier and human name |
| `log10_threshold`, `pc_threshold` | where that policy draws the line |
| `n_events`, `n_true_high_risk` | events scored, events above the line |
| `flagged`, `tp`, `fp`, `fn`, `tn` | confusion matrix |
| `precision`, `recall`, `F2`, `F1` | scores |
| `MSE_HR`, `L` | challenge metrics, when defined at that threshold |
| `evaluable` | false where the positive class is too small to mean anything |
| `predictor` | which forecast this row describes |

### `results/threshold_definition_sensitivity.csv`
Five rows, one per provider. How far the definition of "high risk" can move the
positive class. Written by `scripts/threshold_sweep.py`.

`true_high_risk_events` runs 168 down to 5 across the five policies;
`events_LRP_would_flag` is the same count for the baseline's own decisions.

### `results/threshold_margin_sweep.csv`
12 rows. What happens when the reaction threshold is nudged by ±0.5 dex around
each provider value. Written by `scripts/threshold_sweep.py`.

A margin of −0.5 makes the baseline flag everything (recall 1.000, precision
0.078, F₂ 0.296), which is the clearest demonstration of why the clip floor has
to be recomputed per threshold instead of reused.

---

## Layer 1: Pc methods

### `results/foster_mc_validation.csv`
40 rows across several check families, on synthetic cases where the answer is
known. Written by `scripts/validate_foster_mc.py`.

| Column | Meaning |
|---|---|
| `check` | `rayleigh_anchor`, anisotropic quadrature agreement, or MC comparison |
| `sigma_m`, `R_m`, `R_over_sigma`, `eccentricity` | case geometry |
| `exact`, `foster`, `quadrature`, `chan` | the four implementations |
| `foster_rel_err`, `quad_rel_err`, `chan_rel_err` | error against the exact value |
| `mc`, `sigma_mc`, `z`, `chi2`, `n_inside` | Monte Carlo estimate and its agreement |

### `results/cara_alfano_validation.csv`
11 usable rows, one per published Alfano case (case 12 is degenerate and skipped
explicitly). Our three methods against NASA's `Pc2D_Foster.m`, which was executed
under Octave. Written by `scripts/validate_against_cara.py`.

| Column | Meaning |
|---|---|
| `case` | Alfano case number |
| `cara_pc2d_foster` | CARA's own output |
| `published_pc2d` | the value published with the test suite |
| `ours_n2` | Eq. N-2, 1-D error function |
| `ours_quadrature` | Eq. N-1, 2-D quadrature |
| `ours_closed` | small-HBR closed form |
| `ours_mc`, `sigma_mc` | 6-D Monte Carlo and its standard error |
| `pc_max`, `miss_distance_m`, `rel_speed_m_s` | case context |
| `hbr_m` | hard-body radius used |

Case 4 is the famous one: `ours_closed` reads 1.4 × 10⁻²⁵ against CARA's
4.9 × 10⁻².

---

## Layer 2: evaluation

### `results/fpfn_thresholds_199k.csv`
36 rows: 18 thresholds × 2 views. The headline table over all 199,082 CDMs.
Written by `scripts/threshold_fpfn_199k.py`.

`view` is either `cdm` (every non-final CDM scored against its own event's final
risk) or `event` (the last non-final CDM per event). `provider` is the named
policy where the threshold matches one, otherwise `grid <value>`.
`fp_rate` and `fn_rate` are the two columns the whole argument rests on.

### `results/fpfn_by_leadtime_199k.csv`
28 rows: 7 lead-time bins × 4 thresholds. Same scoring, split by how early the
CDM was issued. Written by `scripts/attribution_199k.py`.

`lead_time` runs from `<0.5 d` to `5-7 d`. At 3 × 10⁻⁷ the false-positive rate
goes from 0.005018 to 0.150630 across those bins.

### `results/attribution_199k.csv`
53 rows. Three attribution axes in one long table, selected by `axis`:
`relative_speed`, `hbr_variant` and `frontier`. Written by
`scripts/attribution_199k.py`.

`bin` holds the category within the axis, so a row reads as "at this threshold,
for this speed bin, the false-positive rate was this".

### `results/attribution_method_199k.csv`
12 rows: four thresholds × three Pc definitions (`esa_risk`, `closed_form`,
`exact_n2`). Method error as an attribution axis, over the full corpus.
Written by `scripts/attribution_method.py`.

### `results/attribution_method_validated.csv`
12 rows, same structure plus a `subset` column fixed to `validated`. The same
comparison restricted to the 35,429 CDMs where our covariance reproduces ESA's
risk to within 0.25 dex, which is the version to quote.
Written by `scripts/attribution_method.py`.

### `results/calibration_scores.csv`
28 rows. Log loss, Brier, ECE and PR-AUC for the forecasts under each threshold
and each calibrator. Written by `scripts/calibration.py`.

`predictor` is `LRP`, `CRP`, `MAX`, `MEAN` or `LRP+Platt`; `split` is `train` or
`test`. Note that `brier` is nearly identical for every predictor, which is the
reason the Brier score is not used in the argument.

### `results/calibration_regression.csv`
28 rows. The slope test: fit the outcome against the forecast, and check whether
the slope is 1 and the intercept 0. Written by `scripts/calibration.py`.

`slope` for LRP runs 0.20–0.80 where honesty requires 1.0, and
`perfect_separation_risk` flags the cells where an unpenalised fit would diverge.

### `results/calibration_reliability.csv`
28 rows. Binned reliability data behind the PNG. `ece` is expected calibration
error, and `n_bins_populated` is how many bins actually received events. For a
constant predictor that number is one, which is why ECE alone is a weak summary.

### `results/calibration_reliability.png`, `results/calibration_pr_curves.png`
The two plots: reliability at five thresholds, and precision–recall curves.
Written by `scripts/calibration.py`.

### `results/interpretability_scorecard.csv`
4 rows, one per Pc method. Five dimensions, each defined explicitly in the
script, plus the fitted partial sensitivities of log₁₀ Pc against
log₁₀ R, log₁₀ √det C and χ².
Written by `scripts/interpretability_scorecard.py`.

Theory says those coefficients should be +2.000, −1.000 and −0.217. The closed
form hits them exactly; Eq. N-2 deviates slightly (+2.166, −1.016, −0.2167);
Monte Carlo is attenuated by sampling noise (+1.178, −0.573, −0.0932).
