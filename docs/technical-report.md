# Kelvins Collision-Avoidance: CDM pipeline + LRP baseline

Reproduction of the ESA Kelvins *Spacecraft Collision Avoidance Challenge*
baseline, plus the orbital-mechanics and covariance layers underneath it.

```
Learn:      Keplerian orbits -> TLE/SGP4 -> covariance basics -> 2-D Pc
Download:   Kelvins dataset (Zenodo 10.5281/zenodo.4463683)
Parse:      CCSDS KVL CDM parser + adapter for the anonymised tabular CDMs
GATE:       reproduce the LRP baseline, target F2 = 0.739 +/- 0.02   --> PASS
```

## Result

| Quantity | Published (Uriot et al. 2021, Table 3) | Reproduced here | Δ |
|---|---|---|---|
| **F₂** | **0.739** | **0.7391** | **+0.0001** |
| MSE_HR | 0.513 | 0.5129 | −0.0001 |
| Score `L = (1/F₂)·MSE_HR` | 0.694 | 0.6940 | 0.0000 |
| CRP baseline `L` (independent check) | 2.5 | 2.5041 | +0.0041 |

**GATE: PASS** — F₂ = 0.7391 against a target of 0.739 ± 0.02.

Confusion matrix on the 2,167 test events (150 high-risk):
`TP=115  FP=63  FN=35  TN=1954`, precision 0.646, recall 0.767.

Two independent published baselines reproduce to <0.5%, which validates the
metric implementation as well as the prediction.

## What the LRP baseline is

`Latest Risk Prediction` is the *clipped naive forecast* (Eq. 6 of the paper):
predict the final risk of a conjunction with the risk reported in the **last
CDM that is still visible to the competitor** — i.e. the newest CDM at least
two days before TCA — clipped so that low-risk predictions sit just below the
classification threshold:

```
r̂_i = r_i        if r_i >  -6
     = -6.001     otherwise
```

`r` is log₁₀ of the collision probability; `r ≥ -6` (Pc ≥ 1e-6) defines a
high-risk event. β = 2 in Fβ weights recall twice as heavily as precision,
because a missed high-risk conjunction is far more costly than a false alarm.

The competition metric is `L = (1/F₂) · MSE_HR`, where `MSE_HR` is the mean
squared error **over high-risk events only**.

## Bugs the checks caught

Four defects were found and fixed during this build. Two of them produced
plausible-looking output and would have silently corrupted results:

| # | Where | Bug | Effect | Caught by |
|---|-------|-----|--------|-----------|
| 1 | `kelvins_data.latest_cdm_per_event` | `sort(ascending)` + `tail(1)` picked the **earliest** CDM, not the latest | F₂ = 0.519 instead of 0.739 | **the gate** |
| 2 | `orbital.state_from_elements` | perifocal x used `r·cos E` instead of `a(cos E − e)` | every eccentric orbit wrong | round-trip test |
| 3 | `orbital.true_anomaly_from_eccentric` | `atan2` arguments swapped | ν = 325° mapped to 215° | anomaly round-trip test |
| 4 | `orbital._parse_float` | B*/nddot implied leading decimal point ignored | B* = −1.1606 instead of −1.1606e−05 | comparison with the `sgp4` reference library |

Bug 1 is precisely the failure the gate is designed to surface: a wrong CDM
selector still yields a well-formed F₂. Bugs 2–4 were caught by the regression
tests in `tests/`.

## The bug that the gate caught

The first run gave **F₂ = 0.519** — far outside tolerance. The pipeline had
selected the wrong CDM:

```python
df = df.sort_values(["event_id", "time_to_tca"])     # ascending
latest = df.groupby("event_id").tail(1)              # <-- BUG
```

Sorting ascending and taking `tail(1)` returns the CDM with the **largest**
`time_to_tca` — the *earliest* CDM of the series, roughly a week before TCA —
not the latest one. The fix is `head(1)`. This is exactly the failure mode the
gate is designed to catch: a silently wrong selector that still produces
plausible-looking numbers.

```
selector              F2      MSE_HR   L
latest CDM  (fixed)  0.7391   0.5129   0.6940   <-- matches the paper
earliest CDM (bug)   0.5193   0.7841   1.5099
max over visible     0.5915   0.5596   0.9460
mean over visible    0.6688   0.5904   0.8827
```

## Layer 0 — definitions, not yet numbers

`src/thresholds.py` + `scripts/threshold_sweep.py`. Every result is reported as a
**threshold sweep**, never as one number, because the positive class is a policy
choice that varies by 3.5 orders of magnitude across operators.

**Thresholds** (all verified against primary sources — see `notes/layer0_thresholds.md`):

| Operator | Pc threshold | Source |
|---|---|---|
| SpaceX Starlink (autonomous) | 3 x 10^-7 | SpaceX published autonomous-avoidance policy |
| Kelvins competition | 10^-6 | Uriot et al. 2021 |
| ESA (lower bound) | 10^-5 | ESA mission range 1e-5 ... 1e-4 |
| ESA / DLR / CSA / 18 SPCS / NASA CARA | 10^-4 | ESA SDC8 (2021), NASA CARA handbook |
| JAXA | 10^-3 | JAXA-JMR-016(E) section 6.2.2, risk level 3 |

**LRP baseline as a sweep** — same code, five different definitions of "high risk":

| Threshold | true HR events | flag | TP | FP | FN | precision | recall | F2 |
|---|---|---|---|---|---|---|---|---|
| 3 x 10^-7 (SpaceX) | 168 | 237 | 142 | 95 | 26 | 0.599 | 0.845 | **0.781** |
| 10^-6 (Kelvins) | 150 | 178 | 115 | 63 | 35 | 0.646 | 0.767 | **0.739** |
| 10^-5 (ESA lower) | 56 | 73 | 40 | 33 | 16 | 0.548 | 0.714 | **0.673** |
| 10^-4 (ESA/DLR) | 17 | 23 | 9 | 14 | 8 | 0.391 | 0.529 | **0.495** |
| 10^-3 (JAXA) | **5** | 2 | 2 | 0 | 3 | 1.000 | 0.400 | **0.455** |

Quoting "F2 = 0.739" without the threshold is quoting a number with no meaning.
At 10^-3 the metric rests on five positive examples -- two significant figures
on a sample of five is numerology, not measurement.

**Cross-provider disagreement** (first-class metric, `disagreement()`):

| | on LRP predictions | on true final risk |
|---|---|---|
| overall disagreement | 10.84 % | 7.52 % |
| all providers act | 0.09 % | 0.23 % |
| no provider act | 89.06 % | 92.25 % |
| widest pair | SpaceX vs JAXA 10.84 % | SpaceX vs JAXA 7.52 % |

Even with perfect knowledge, ~1 event in 13 is classified differently depending
on whose rulebook you use. No model can remove that floor.

**The clip floor is threshold-relative.** `r_hat = r if r >= T else T - eps`.
Reusing the fixed -6.001 floor at the 3 x 10^-7 threshold makes every low-risk
event look high-risk (recall 1.000, precision 0.078, F2 0.296). `sweep()`
applies the clip internally -- pass it raw model output.

## Roadmap

Full plan of record: `notes/ROADMAP.md`. Status:

| Layer | Status |
|---|---|
| 0 — Definitions (threshold sweeps, disagreement) | **DONE** |
| 1 — Pc methods (2-D closed form, quadrature, MC, HBR variants) | **DONE** — validated against CARA's own code |
| 2 — Evaluation core (calibration, attribution, interpretability) | **DONE** except interpretability scorecard |
| 3 — Triage method (the forecaster that must pass the harness) | not started |
| 4 — Live replay (SOCRATES ingest, alert-trustworthiness report) | not started |
| 5 — Reports (one command → markdown/PDF) | not started |

## Layer 1 -- Pc methods

`src/foster2d.py`, `scripts/validate_foster_mc.py`, `notes/layer1_pc_methods.md`.

Four implementations spanning the CARA Appendix N family, so they check each
other rather than one formula: the Chan 2008 small-HBR closed form (`pc2d`), the
**Eq. N-2** 1-D error-function reduction (`pc2d_1d`, alias `foster2d`), the
**Foster & Estes 1992 2-D quadrature of Eq. N-1** (`pc2d_numeric`), and a
CARA-style 6-D Monte Carlo that draws the full state, propagates to TCA and
counts hits.

| check | result |
|---|---|
| isotropic, zero miss -> exact Rayleigh CDF | pc2d_1d rel err **6.5e-08** |
| pc2d_1d vs polar quadrature, 5 anisotropic cases | agree to **4e-07** |
| CARA-style 6-D MC vs analytic, 12 cases x 2M samples | worst \|z\| **1.81** |
| ESA's `risk` field reproduced from published covariances | median err **+0.0012 dex** |
| CARA's own 63,603-conjunction MC benchmark (Fig. N-7) | 98.9% agree, ~1.1% violate the 2-D assumptions |

**The finding:** the small-HBR closed form everyone uses operationally is not
free. It deviates from the exact integral by **+0.5%** at R/sigma=0.02 with
eccentricity 10, **+12.6%** at R/sigma=0.10, and **+49.3%** at R/sigma=0.20 --
entirely ordinary for a LEO conjunction a few days out. It always errs high.
`pc2d_1d` has no such approximation and costs one 200-node quadrature.

### Validated against CARA's own code on the published Alfano cases

`nasa/CARA_Analysis_Tools` is **public** under NOSA (GSC-18593-1), and it ships
the whole Pc SDK plus the Alfano (2009) test cases *with their published
expected values*. It clones and runs under GNU Octave 9.4.0, so the reference is
CARA's actual `Pc2D_Foster.m`, not a re-implementation:

```
git clone https://github.com/nasa/CARA_Analysis_Tools.git
sudo apt-get install -y --no-install-recommends octave octave-io
python3 scripts/extract_alfano_cases.py   # .xls -> /tmp/alfano_cases.csv
octave --no-gui --quiet scripts/run_cara.m # CARA's code -> /tmp/cara_reference.csv
python3 scripts/validate_against_cara.py   # ours vs CARA
```

CARA's own code reproduces its published values to **1e-12**, confirming the
environment is CARA. Against that reference:

| our method | worst rel err | median | CARA's own unit-test tolerance |
|---|---|---|---|
| `pc2d_1d` (Eq. N-2) | **8.2e-07** | 6.7e-08 | 1e-3 |
| `pc2d_numeric` (Eq. N-1, Foster & Estes) | **6.9e-06** | 4.1e-07 | 1e-3 |
| `cara_monte_carlo` (2M samples) | worst \|z\| = **3.10** | — | 4 sigma |
| `pc2d` (small-HBR closed form) | **1.21e+01** | 1.00 | — |

**On CARA's published cases the closed form is not merely imprecise.** Its
zero-miss limit `R^2/(2*sqrt(det C))` already exceeds 1 for cases 4, 5, 9 and
10, and at non-zero miss case 4 gives **1.4e-25 against CARA's 4.93e-02** -- a
24-order-of-magnitude error, because the disk centre sits ~10 sigma out along
the minor axis of a highly elongated covariance while the disk still straddles
the ridge. Appendix N's own limit `R_lim = 4*sigma_x - min(R)` shows up in the
published cases rather than being derived.

**Conclusion: use Eq. N-2. It is exact, fast, and matches CARA to 1e-7.** Two
further confirmations remain in place: ESA's operational `risk` field is
reproduced to **+0.0012 dex** over 199,082 CDMs, and CARA's own Fig. N-7
benchmark on 63,603 real conjunctions agrees to 98.9%.

## Layer 2 -- the evaluation core

`src/calibration.py`, `scripts/calibration.py`, `scripts/threshold_fpfn_199k.py`,
`scripts/attribution_199k.py`, `notes/layer2_evaluation.md`.

### The headline table: threshold-explicit FP/FN across all 199,082 CDMs

The competition scores only its 2,167-event test split at a single 1e-6
threshold. Scoring every CDM in the raw release against its own event's final
risk makes the threshold dependence visible at full sample size.
(199,082 CDMs / 15,321 events / 183,761 non-final CDMs.)

CDM level:

| threshold | n_pos | **FP rate** | **FN rate** | F2 |
|---|---|---|---|---|
| 1e-7 | 8,691 | **0.10838** | **0.12427** | 0.6203 |
| 3e-7 (SpaceX) | 6,469 | **0.08341** | **0.15520** | 0.5924 |
| 1e-6 (Kelvins) | 4,457 | 0.05698 | 0.18667 | 0.5723 |
| 1e-5 (ESA lower) | 1,448 | **0.02038** | **0.28246** | 0.4926 |
| 1e-4 (ESA/DLR/CARA) | 375 | **0.00468** | **0.52800** | 0.3491 |
| 1e-3 (JAXA) | 79 | **0.00035** | **0.60759** | 0.3771 |

Moving the reaction threshold from 1e-3 to 1e-7 cuts the false-negative rate
from **61% to 12%** and raises the false-positive rate from **0.035% to 10.8%**
-- a 300x swing in FP for a 5x cut in FN. No setting is good on both. That is
the entire content of the table and it is invisible to any single-threshold
evaluation. At the event level the 1e-3 positive class is **7 events**, so the
JAXA row is reported for completeness of the sweep, not as a measurement.

### Calibration: does Pc mean anything? Not below ~1e-6.

Reliability diagrams in `results/calibration_reliability.png`. At the SpaceX
threshold the observed frequency of "ends up above 3e-7" is **flat at ~1e-3**
across predicted Pc from 1e-22 to 1e-11, rising only in the top bin. Sixteen
decades of predicted probability carry essentially no information.

Calibration regression (test split; slope 1 / intercept 0 = calibrated):

| threshold | LRP slope | LRP log10loss | CRP log10loss | LRP+Platt log10loss | PR-AUC |
|---|---|---|---|---|---|
| 3e-7 | **0.34** | 0.5047 | 0.3876 | **0.0759** | 0.6674 |
| 1e-6 | **0.34** | 0.4468 | 0.3461 | **0.0789** | 0.6598 |
| 1e-5 | **0.28** | 0.1621 | 0.1292 | **0.0505** | 0.5238 |
| 1e-4 | **0.32** | 0.0354 | 0.0392 | **0.0128** | 0.3430 |

1. **LRP is over-confident** -- slope 0.2-0.8 on both splits, at every threshold.
2. **A constant predictor beats it on log loss** at every threshold with enough
   positives to tell.
3. **A 2-parameter Platt map fixes most of it**: fit on train, applied to test,
   log loss drops **6.6x** at 3e-7 with PR-AUC unchanged (0.6674 -> 0.6651).
   LRP's *ranking* is fine; its *probabilities* are not.

The Brier score is useless here -- identical to three decimals for LRP and CRP
in every cell, because the probabilities are tiny relative to the base rate.

### Attribution: where the error comes from

| axis | finding |
|---|---|
| **CDM age (dominant)** | FP rate rises **30x** from <0.5 d lead (0.50%) to 5-7 d (15.1%) at 3e-7. Early CDMs cry wolf. |
| **HBR choice** | flat 5 m instead of the catalogue HBR moves the **FN rate ~35%** at every threshold, with zero change in skill. RCS uncertainty at +/-30% moves it only 3-5%. |
| **geometry (v_rel)** | FP rate varies only 0.080 -> 0.122 across a 34x range of relative speed, with no trend. **Hypothesis rejected** -- slow encounters are a recall problem, not a precision problem. |

## Repository layout

```
kelvins_pc/
├── src/
│   ├── metrics.py        F_beta, MSE_HR, the competition loss L, risk clipping
│   ├── thresholds.py     Layer 0: configurable thresholds, sweep, disagreement
│   ├── calibration.py    Layer 2: reliability, calibration regression, PR
│   ├── foster2d.py       Layer 1: Foster 2-D, CARA-style MC, HBR variants
│   ├── kelvins_data.py   dataset loader + "latest CDM per event" selector
│   ├── orbital.py        Keplerian elements, TLE parsing, SGP4, J2 rates
│   ├── cdm_parser.py     CCSDS KVL CDM parser + Kelvins tabular adapter
│   ├── covariance.py     RTN basis, encounter-plane projection, Mahalanobis
│   └── pc2d.py           2-D Pc: closed form, quadrature, Monte Carlo
├── scripts/
│   ├── reproduce_lrp.py      <-- the gate
│   ├── threshold_sweep.py    Layer 0 sweep + cross-provider disagreement
│   ├── calibration.py        Layer 2 calibration + PR curves + plots
│   ├── threshold_fpfn_199k.py  headline FP/FN table over all 199k CDMs
│   ├── attribution_199k.py    FP/FN by lead time, v_rel, HBR variant
│   ├── validate_foster_mc.py  Foster 2-D vs Chan vs CARA-style MC
│   ├── run_cara.m            GNU Octave driver: runs CARA's Pc2D_Foster.m
│   ├── extract_alfano_cases.py  Alfano .xls -> CSV for the Octave driver
│   ├── validate_against_cara.py  ours vs CARA on the published cases
│   ├── fetch_data.py          re-acquire the dataset after a reset
│   ├── restore.sh             one-command full environment restore
│   ├── validate_pipeline.py  parser + Pc validation harness
│   └── demo_pipeline.py      end-to-end walkthrough
├── tests/
│   └── test_orbital.py     regression tests for the orbital layer
├── notes/
│   ├── learning_path.md     Kepler/TLE/covariance/2-D Pc study notes
│   └── layer0_thresholds.md why a single F2 is undefined
└── results/                  metrics and per-event predictions
```

## Restoring the environment (after any reset)

Two things are lost between turns, and **neither is a workspace problem**:

1. **The 221 MB dataset** -- it exceeds the workspace snapshot size cap. It is
   re-fetched from Zenodo (public, DOI 10.5281/zenodo.4463683).
2. **Installed pip packages** -- these never persist; only files under
   `/home/user` are snapshotted.

The project itself is **1.2 MB / 42 files**, roughly 1% of the 128 MB snapshot
cap, so code, notes and `results/` persist normally.

One command restores both and re-verifies everything:

```bash
bash scripts/restore.sh          # full restore + all 8 checks
bash scripts/restore.sh --check  # verify only
```

`requirements.txt` pins the exact versions the published results were produced
with. Cloud storage was considered and rejected -- see
`notes/storage_and_persistence.md` for the reasoning.

## Reproducing

```bash
# data: unzip the Zenodo release, or point KELVINS_DATA_ROOT at it
export KELVINS_DATA_ROOT=/path/to/"Collision Avoidance Challenge - Dataset"

python3 scripts/reproduce_lrp.py      # the gate
python3 scripts/validate_pipeline.py  # parser + 2-D Pc validation
python3 tests/test_orbital.py         # orbital regression tests
python3 scripts/demo_pipeline.py      # orbital mechanics walkthrough
```

Dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `sgp4`.

## Dataset facts (verified, not assumed)

| Check | Value |
|---|---|
| test CDMs | 24,484 |
| test events | 2,167 |
| high-risk test events (r ≥ −6) | 150 — matches the paper exactly |
| min `time_to_tca` in the inputs | 2.000213 d (rule iii: ≥ 2) |
| max `time_to_tca` in the ground truth | 0.977200 d (rule ii: < 1) |
| CDMs per event | 1 / 11.30 / 17 (min / mean / max) |
| raw database | 199,082 CDMs, 15,321 events |

`test_data_private.csv` is the **ground truth**: one row per test event, the
removed last CDM, with its risk renamed `true_risk`. Without it the baseline
cannot be scored — the `risk` column of `test_data.csv` is an *input feature*,
not the label.

## Validation of the 2-D Pc layer

Beyond the baseline gate, the covariance and Pc code is checked against the
dataset itself (`scripts/validate_pipeline.py`):

| Check | Result |
|---|---|
| det(C_target) vs published determinant | ratio 1.000000 (min and max, all rows) |
| det(C_chaser) vs published determinant | ratio 1.000000 (min and max, all rows) |
| `max_risk_scaling` vs χ²/2 | median ratio 1.0000, corr 0.988 |
| `max_risk_estimate` vs R²/(e χ² σ₁σ₂) | median error +0.0012 dex |
| closed-form Pc vs `risk`, χ² < 1 | 96% within 0.25 dex, median error 0.021 dex |
| isotropic zero-miss Pc vs Rayleigh CDF | relative error 1e-9 … 1e-11 |

The `max_risk_*` fields are an *independent* confirmation: they pin the
encounter plane (perpendicular to the relative velocity), the combined
covariance and the hard-body radius without relying on the `risk` column.

A documented residual remains for far misses (χ² ≥ 10), where ESA's reported
risk falls below the closed form; the published columns carry only the
position block of each covariance. See `notes/learning_path.md` §4.5.

## References

* T. Uriot, D. Izzo, R. Abay, L. F. Simões, N. Einecke, S. Rebhan,
  J. Martinez-Heras, F. Letizia, J. Siminski, K. Merz,
  *Spacecraft Collision Avoidance Challenge: design and results of a machine
  learning competition*, Astrodynamics 5:1–17, 2021. arXiv:2008.03069
* NASA, *Spacecraft Conjunction Assessment and Collision Avoidance Best
  Practices Handbook* (2023) and *Appendix N — Pc Calculation Approaches* (2024)
* D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., 2013
* F. R. Hoots, R. L. Roehrich, *SPACETRACK Report #3*, 1980
* CelesTrak TLE and CDM format documentation
* ESA Kelvins dataset, Zenodo 10.5281/zenodo.4463683
