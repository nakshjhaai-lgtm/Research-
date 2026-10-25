# Roadmap — the five layers

Canonical plan of record. Update the status column as things land; don't rewrite
the architecture mid-flight.

```
LAYER 0 — DEFINITIONS ("you can't optimise an undefined quantity")
  • Configurable FP/FN definitions per threshold:
    10⁻³ (JAXA) · 10⁻⁴ (ESA/DLR/18SPCS) · 10⁻⁵ · 3×10⁻⁷ (SpaceX autonomous)
  • Every result reported as a THRESHOLD SWEEP, never one number.
  • Cross-provider disagreement as a first-class metric.

LAYER 1 — Pc METHODS (reference implementations, cited)
  • Foster/Estes 2D (Foster & Estes 1992)
  • Monte Carlo ground-truth estimator (CARA-style; Appendix N methodology)
  • HBR variants: flat 5 m (Space-Track EVR behaviour), catalogue HBR,
    RCS-uncertainty HBR
  • (Optional, credit) Chan/3D variants

LAYER 2 — THE EVALUATION CORE   ← the actual contribution
  • Calibration / reliability curves (does Pc mean anything?)
  • F₂ + precision-recall at every threshold (Kelvins-standard metrics;
    beat or contextualise LRP F₂ = 0.739)
  • Attribution: how much FP comes from HBR choice vs covariance staleness
    (CDM age) vs method error vs geometry (low-relative-velocity encounters)
  • Interpretability scorecard (per CARA's "physical reasoning" requirement):
    does the method explain WHY?

LAYER 3 — YOUR TRIAGE METHOD (the candidate that must pass your own harness)
  • Interpretable features: Pc persistence/decay across the CDM sequence,
    volatility near threshold, time-to-TCA, CDM age, HBR provenance,
    geometry flags
  • Calibrated, transparent model (gradient-boosted trees or logistic —
    monotonic constraints), explicit F₂ reporting
  • Baselines to beat/contextualise: LRP (0.739), Kessler-style sequence model,
    plain Foster thresholding

LAYER 4 — LIVE REPLAY (what keeps it alive 2026→2028)
  • Rolling "state of alert trustworthiness" report (gpsjam.org but for alarm
    quality). New fragmentation events automatically enter the stream.

LAYER 5 — REPORTS (the artifact people read)
  • One command → markdown/PDF report with tables, calibration plots,
    threshold sweeps, honest limitations
```

---

## Status

| Layer | Status | Where |
|---|---|---|
| 0 — Definitions | **DONE** | `src/thresholds.py`, `scripts/threshold_sweep.py`, `notes/layer0_thresholds.md` |
| 1 — Pc methods | **DONE** — validated against CARA's own code | `src/pc2d.py`, `src/foster2d.py`, `src/covariance.py`, `scripts/validate_against_cara.py`, `notes/layer1_pc_methods.md` |
| 2 — Evaluation core | **DONE** — all four attribution axes + interpretability scorecard | `src/calibration.py`, `src/cdm_covariance.py`, `scripts/calibration.py`, `scripts/attribution_method.py`, `scripts/interpretability_scorecard.py`, `notes/layer2_evaluation.md` |
| 3 — ML forecaster | NOT STARTED | — |
| 4 — Live replay | NOT STARTED | — |
| 5 — Reports | NOT STARTED | — |

---

## Facts verified for this plan (2026-09-27)

**USA-32 fragmentation — confirmed.** S4S notice reproduced by TraCSS:
*"U.S. Space Forces-Space (S4S) confirms the fragmentation of USA 32 (#19460) in
LEO on 13 Sep 2026 (JDay 256) at approximately 2113 UTC. All tracked debris are
being incorporated into routine conjunction assessment screenings to support
spaceflight safety."* Near-circular 773 × 780 km, inclination 84.98°, 100.4 min
period; cause unknown, no fragment count released as of 24 Sep 2026. Sources:
[tracss.gov](https://tracss.gov/), [SpaceNexus](https://spacenexus.us/blog/usa-32-farrah-spy-satellite-breakup-debris),
[Indian Defence Review](https://indiandefencereview.com/classified-us-spy-satellite-breaks-apart-leaves-debris-space/).

**SOCRATES is tri-daily, not weekly.** CelesTrak runs the screen **three times
each day**, each run covering the **next seven days**, against the full
unclassified GP (TLE) catalog. SOCRATES Plus no longer screens intra-fleet
conjunctions between fully-operational satellites of the same fleet — that cut
the total from 143,997 to 56,343 pairs on 2023-03-02. Format docs:
[celestrak.org/SOCRATES/](https://celestrak.org/SOCRATES/),
[socrates-format.php](https://celestrak.org/SOCRATES/socrates-format.php).

**The Kelvins CDMs carry what Layer 2's attribution needs.** Confirmed present
in the 103 flattened columns: `relative_speed` + `relative_velocity_r/t/n`
(geometry), `time_to_tca` (CDM age), `t_span`/`c_span` (HBR provenance),
`t_recommended_od_span` and `t_residuals_accepted` (covariance staleness),
`t_rcs_estimate` (RCS-uncertainty input), `miss_distance`,
`max_risk_scaling`, `mahalanobis_distance`. So the low-relative-velocity and
covariance-staleness attributions are feasible on the existing data — no new
ingest required.

---

## Corrections to the plan, so the layers don't drift

### Layer 1: the HBR in this dataset is not 5 m
The Kelvins CDMs use `HBR = (t_span + c_span)/2`, which reproduces ESA's `risk`
to a median error of −0.021 dex (validated on 61 near-zero-χ² rows). A flat 5 m
is the Space-Track/18-SPCS EVR convention — **a different assumption, not a
variant of the same one.** Implement flat-5 m, catalogue-HBR and
RCS-uncertainty-HBR as *variants that move the HBR input*, and report how far
each one moves Pc. That is an attribution result in its own right, and it is
exactly the kind of thing Layer 2 should quantify rather than assume.

### Layer 1: "Foster/Estes 2D" is not what is currently implemented
`src/pc2d.py` holds the **Alfriend–Akella / Chan closed form** for the integral
of a 2-D Gaussian over a disk, plus polar quadrature and Monte Carlo. Foster &
Estes 1992 established the 2-D conjunction-plane formulation; the closed-form
expression is conventionally credited to Alfriend & Akella and to Chan. Same
plane definition, different derivation. Label the code accordingly, and cite the
derivation actually implemented — do not claim a Foster & Estes implementation.

### Layer 1: the Monte Carlo is a benchmark, not a CARA ground truth
`pc2d_monte_carlo` is a 2-D Gaussian sampler used as a cross-check (5σ Poisson
band, ~6 expected hits at Pc ≈ 1.5e-5). A CARA-style estimator draws from the
**full 6-D** state with the SGP4 propagator and the two-body mapping, at 10⁶–10⁸
samples, and handles the position–velocity covariance block. That block is
**absent from the flattened Kelvins CSVs** (present in
`raw_data/raw_data_2015-2019.gz`). It is also the likely cause of the
unexplained residual for χ² ≥ 10 documented in `notes/learning_path.md` §4.5 —
worth revisiting when Layer 1 is built out.

**Superseded in part:** `src/foster2d.py::cara_monte_carlo` is now the CARA-style
full-6-D estimator (with CARA's `CovRemEigValClip` eigenvalue clipping), and
`scripts/validate_against_cara.py` runs it against CARA's own `Pc2D_Foster` on
the published Alfano cases: worst |z| = 3.10 at 2M samples. The χ² ≥ 10 residual
on ESA's `risk` is still open.

### Layer 4: SOCRATES has no covariance, so Layer 2 does not transfer directly
SOCRATES emits TLE/GP data only. There is **no covariance matrix, no
`t_span`/`c_span`, no CDM history** — you must supply a covariance-from-TLE
model yourself. Pc-from-TLE therefore has a fundamentally different error
profile from Pc-from-CDM, and Layer 2's calibration curves **cannot** be assumed
to carry over. Layer 4 needs its own validation, most likely a retrospective
one: replay archived TLEs, propagate, and compare against known CDMs.

### Layer 5: PDF needs a converter
Markdown and HTML render in the preview natively. A PDF requires a converter
(pandoc/weasyprint) that is **not installed**; matplotlib is available for the
calibration plots. Deliver markdown + HTML first, PDF as a separate step.

---

## Suggested order

1. **Layer 2 calibration** — highest value, zero new data, and it directly
   attacks the "does Pc mean anything?" question using the 2,167-event truth
   set. Report the reliability curve *per threshold* (Layer 0 makes this
   mandatory, not optional).
2. **Layer 1 HBR variants** — cheap, and it produces the first real
   attribution numbers (how much of the error is an assumption about object
   size?).
3. **Layer 2 attribution** — now that the features are confirmed present.
4. **Layer 3** — the triage model, evaluated as a sweep.
5. **Layer 4** — the retrospective TLE replay, before any live ingest.
6. **Layer 5** — one command, and only once there is something worth reporting.
