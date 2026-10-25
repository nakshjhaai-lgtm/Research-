# HANDOFF — Kelvins Collision-Avoidance Study

**Purpose of this file:** a brand-new session must be able to read it and
*continue the work* without re-deriving anything, re-proving earlier results, or
guessing. Every claim here is either (a) backed by a named artifact and a
passing check, or (b) explicitly marked OPEN / NOT STARTED. Nothing is asserted
on memory alone.

Last updated: 2026-09-27. Status: **Layers 0, 1 and 2 complete. Layer 3+ not
started.**

---

## 0. START HERE — read this first

1. **Run the one-command restore.** It re-establishes the dataset, the Python
   packages, the CARA clone and GNU Octave, then runs all 11 checks:

   ```bash
   cd /home/user/kelvins_pc && bash scripts/restore.sh
   ```

   Expected tail: 11 × `PASS` then `RESTORE COMPLETE -- all checks pass`.

2. **The environment is hostile.** The dataset and installed packages have been
   destroyed **five separate times** in one session, including mid-turn. Any
   `FileNotFoundError` on `test_data.csv`, or a missing `octave`/`sgp4`, is an
   environment reset, not a bug in the code. `scripts/restore.sh` is the only
   supported fix. Do not "fix" this by relaxing a check.

3. **`results/` is regenerated, never edited.** Every number quoted in this file
   comes from a script that exits 0. If a number here disagrees with a fresh
   run, the fresh run wins — report the discrepancy, do not reconcile it by
   editing the notes.

4. **The one-command sanity check for any claim in this document:**
   ```bash
   cd /home/user/kelvins_pc && bash scripts/restore.sh --check
   ```

---

## 1. The task list, verbatim, and its status

### The layers

| Layer | Requirement | Status | Evidence |
|---|---|---|---|
| **L0** | Configurable FP/FN per threshold: 10⁻³ (JAXA), 10⁻⁴ (ESA/DLR/18SPCS), 10⁻⁵, 3×10⁻⁷ (SpaceX) | **DONE** | `src/thresholds.py`, `scripts/threshold_sweep.py`, `results/threshold_sweep*.csv` |
| **L0** | Every result a THRESHOLD SWEEP, never one number | **DONE** | every result table carries a `log10_threshold` column; `sweep()` clips internally |
| **L0** | Cross-provider disagreement as a first-class metric | **DONE** | `results/threshold_definition_sensitivity.csv`; truth 7.52%, predictions 10.84% |
| **L1** | Foster/Estes 2-D (Foster & Estes 1992) | **DONE** | `pc2d_numeric` in `src/pc2d.py` = Eq. N-1 |
| **L1** | MC ground-truth estimator (CARA-style, Appendix N) | **DONE** | `cara_monte_carlo` in `src/foster2d.py`; full 6-D, CARA eigenvalue clipping |
| **L1** | HBR variants: flat 5 m, catalogue, RCS-uncertainty | **DONE** | `hbr_flat_5m`, `hbr_catalogue`, `hbr_rcs_uncertainty` in `src/foster2d.py` |
| **L1** | (Optional, credit) Chan / 3-D variants | **PARTIAL** | Chan closed form = `pc2d`. **3-D is NOT implemented.** Optional per spec; see §7 OPEN. |
| **L1** | Validation against CARA tools on known cases | **DONE** | `scripts/validate_against_cara.py` → `results/cara_alfano_validation.csv` |
| **L2** | Calibration / reliability curves | **DONE** | `results/calibration_reliability.png` + `_reliability.csv` |
| **L2** | F₂ + PR at every threshold | **DONE** | `results/threshold_sweep.csv`, `results/calibration_pr_curves.png` |
| **L2** | Attribution: HBR choice vs covariance staleness vs method error vs geometry | **DONE (all four)** | see §4.4 — this took two new scripts this session |
| **L2** | Interpretability scorecard | **DONE** | `scripts/interpretability_scorecard.py` → `results/interpretability_scorecard.csv` |

### The learning item

> Learn: Keplerian orbits → TLE/SGP4 → covariance basics → 2D Pc math.
> Sources: CARA Best Practices Handbook + Appendix N, Vallado, CelesTrak docs.

`notes/learning_path.md`. **CARA Appendix N was read in full (13 chunks) and
CelesTrak's TLE + CDM format docs were fetched. Vallado was NOT read** — it is
cited by chapter (ch. 2 two-body, ch. 9 SGP4, ch. 11 frames/covariance
transforms, eq. 9-35) from knowledge of the standard text, with no copy in the
workspace. If that citation must be verified, obtain the book. This is the only
task item with no external check.

### The gates

| Gate | Requirement | Status |
|---|---|---|
| G-LRP | Reproduce LRP within ±0.02 of F₂ = 0.739 | **PASSED** — F₂ = 0.7390745, error 7×10⁻⁵ |
| G-RESULT | ≥2 reproducible result tables + calibration plots | **PASSED** — 16 CSVs + 2 PNGs |
| G-CARA | Foster-vs-MC agreement on CARA test cases | **PASSED** — see §4.3 |

### The headline deliverable

> "threshold-explicit FP/FN rates across 10⁻³…10⁻⁷ on 199k CDMs"

**DONE** — `results/fpfn_thresholds_199k.csv` (36 rows: CDM-level and
event-level). Table in §4.1.

### Not started

- **Layer 3** — ML forecaster predicting final risk from CDMs visible ≥2 d before TCA.
- **Layer 4** — live replay / SOCRATES ingest.
- **Layer 5** — one-command report.
- **3-D Pc variant** (optional/credit in Layer 1).

---

## 2. Repository map

```
/home/user/kelvins_pc/
  src/
    thresholds.py         L0: RiskThreshold, PROVIDER_THRESHOLDS, sweep(),
                          disagreement(), decision_instability(),
                          operational_cost(), format_sweep()
    pc2d.py               pc2d (Chan small-HBR closed form), pc2d_max,
                          pc2d_numeric (Foster & Estes 1992, Eq. N-1),
                          pc2d_monte_carlo, pc2d_from_rtn
    foster2d.py           pc2d_1d (Eq. N-2, Alfano/Elrod) + foster2d alias,
                          foster2d_pc_max, encounter_plane_covariance,
                          cara_monte_carlo, hbr_* variants, foster_vs_mc
    cdm_covariance.py     NEW: assembles the full 6x6 RTN covariance from the
                          raw CDM element columns; plane_basis, project,
                          clip_psd
    covariance.py         encounter_plane_basis, project_to_plane, mahalanobis
    kelvins_data.py       load_raw / load_train / load_test / load_test_truth /
                          latest_cdm_per_event / build_lrp_frame
    cdm_parser.py         CCSDS KVL parser
    orbital.py            SGP4 wrappers (needs sgp4)
    calibration.py        L2-penalised sklearn logistic
  scripts_helpers.py      NEW: shared vectorised pc2d kernels (exact + closed).
                          Single source so the corpus scripts cannot drift.
  scripts/
    restore.sh            ONE COMMAND. Re-establishes everything, runs 11 checks.
    fetch_data.py         Zenodo re-acquisition, --check mode
    reproduce_lrp.py      GATE G-LRP
    validate_pipeline.py  5 checks incl. ESA risk reproduction
    validate_foster_mc.py Foster vs MC on synthetic cases
    validate_against_cara.py  NEW: ours vs CARA's own Pc2D_Foster
    run_cara.m            NEW: GNU Octave driver, runs CARA's code
    extract_alfano_cases.py   NEW: Alfano .xls -> CSV
    threshold_sweep.py    L0 sweep
    calibration.py        L2 calibration
    threshold_fpfn_199k.py    HEADLINE
    attribution_199k.py   L2 axes 1-3 (geometry, HBR, frontier)
    attribution_method.py NEW: L2 axis 4 (method error)
    interpretability_scorecard.py  NEW: L2 scorecard
    demo_pipeline.py
  tests/test_orbital.py   10 regression tests
  notes/                  ROADMAP.md, learning_path.md, layer0_thresholds.md,
                          layer1_pc_methods.md, layer2_evaluation.md,
                          storage_and_persistence.md, ROADMAP_VERBATIM.md
  results/                16 CSVs + 2 PNGs (all regenerated by the scripts)
```

`/home/user/CARA_Analysis_Tools/` — the CARA clone (re-created by `restore.sh`).

---

## 3. How to reproduce everything from zero

```bash
# 0. workspace is /home/user/kelvins_pc
cd /home/user/kelvins_pc

# 1. one command: dataset + packages + CARA clone + Octave + all 11 checks
bash scripts/restore.sh

# 2. the CARA comparison, step by step (also chained inside
#    validate_against_cara.py, which runs any missing step itself)
python3 scripts/extract_alfano_cases.py     # .xls -> /tmp/alfano_cases.csv
octave --no-gui --quiet scripts/run_cara.m  # CARA's code -> /tmp/cara_reference.csv
python3 scripts/validate_against_cara.py    # ours vs CARA
```

Manual prerequisites if `restore.sh` cannot be used:
```bash
pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.1 scikit-learn==1.6.1 \
            matplotlib==3.10.9 sgp4==2.27
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends octave octave-io
git clone --depth 1 --filter=blob:none --no-checkout \
    https://github.com/nasa/CARA_Analysis_Tools.git /home/user/CARA_Analysis_Tools
cd /home/user/CARA_Analysis_Tools
git sparse-checkout set DistributedMatlab/ProbabilityOfCollision \
    DistributedMatlab/Utils DataFiles
git checkout
```

The dataset itself: Zenodo record 4463683, CC-BY, ~221 MB. `scripts/fetch_data.py`
re-acquires it and unpacks it to the path `src/kelvins_data.py` expects. Note the
zip unpacks one directory level shallower than expected; `fetch_data.py` inserts
the `kelvins/` level. **The Zenodo release already IS durable cloud storage** —
re-uploading it elsewhere is strictly worse (see `notes/storage_and_persistence.md`).

---

## 4. Key results

### 4.1 HEADLINE — threshold-explicit FP/FN, 199,082 raw CDMs / 15,321 events

`results/fpfn_thresholds_199k.csv`. 183,761 non-final CDMs scored.

**CDM level:**

| threshold | n_pos | FP | FN | FP rate | FN rate | F₂ |
|---|---|---|---|---|---|---|
| 10⁻³ | 79 | 64 | 48 | 0.00035 | 0.608 | 0.377 |
| 10⁻⁴ | 375 | 858 | 198 | 0.00468 | 0.528 | 0.349 |
| 10⁻⁵ | 1,448 | 3,715 | 409 | 0.02038 | 0.282 | 0.493 |
| 10⁻⁶ | 4,457 | 10,217 | 832 | 0.05698 | 0.187 | 0.572 |
| 10⁻⁷ | 8,691 | 18,974 | 1,080 | 0.10838 | 0.124 | 0.620 |

**Event level (last non-final CDM):** 10⁻³ → 7 positives, FP 0.00060 / FN 0.714;
10⁻⁷ → 1,133 positives, FP 0.18377 / FN 0.19947.

**The trade-off:** moving 10⁻³ → 10⁻⁷ cuts FN 61% → 12% and raises FP 0.035% →
10.8%. A 300× FP swing for a 5× FN cut. No setting is good on both.

### 4.2 LRP baseline gate

`results/lrp_metrics.csv` — **F₂ = 0.7390745**, MSE_HR = 0.5128891, L = 0.6939613.
Target 0.739 ± 0.02. Test set: 2,167 events, 24,484 CDMs, 150 high-risk.
CRP cross-check L = 2.5041 vs the paper's 2.5.

### 4.3 CARA validation — the reference is CARA's own code

`results/cara_alfano_validation.csv`. CARA's `Pc2D_Foster.m` (from
`nasa/CARA_Analysis_Tools`, public under NOSA GSC-18593-1) executed under GNU
Octave 9.4.0 on the 12 Alfano (2009) cases whose published expected values ship
with the SDK.

**CARA's own code reproduces its published values to 9.5×10⁻¹²** — this is the
check that proves the environment is CARA, not a re-implementation.

| our method | worst rel err vs CARA | median |
|---|---|---|
| `pc2d_1d` (Eq. N-2) | **8.17×10⁻⁷** | 6.74×10⁻⁸ |
| `pc2d_numeric` (Eq. N-1) | **6.89×10⁻⁶** | 4.11×10⁻⁷ |
| `cara_monte_carlo` (2M) | worst \|z\| = **3.10** | all within 4σ |
| `pc2d` (closed form) | **1.21×10⁺¹** | 1.00 |

CARA's own unit test asserts `RelTol = 1e-3`; our 1-D form is ~1200× tighter.

**On CARA's published cases the small-HBR closed form is structurally invalid.**
Its zero-miss limit `R²/(2√det C)` already exceeds 1 for cases 4 (3.51), 5 (1.01),
9 and 10 (4.86) — impossible for a probability. Case 4 gives **1.4×10⁻²⁵ against
CARA's 4.93×10⁻²**: a 24-order-of-magnitude error, because the disk centre sits
~10σ out along the minor axis while the disk still straddles the ridge of a
highly elongated covariance. Appendix N's own limit `R_lim = 4σ_x − min(R)`
shows up in the published cases rather than being derived.

**Use Eq. N-2. It is exact, fast, and matches CARA to 10⁻⁷.**

Case 12 is degenerate: `|r_rel| = 0` and `|v_rel| = 0` ("LEO, identical orbits"),
so CARA's encounter frame `y = v/|v|` is 0/0 undefined — which is why CARA's own
published CSV has NaN there. Skipped explicitly, not silently. Cases 9 and 10 sit
at `|v_rel| = 2.08 mm/s` and Eq. N-2 still matches CARA to 8.2×10⁻⁷.

### 4.4 Layer 2 attribution — all four axes

| Axis | File | Finding |
|---|---|---|
| Geometry (relative speed) | `attribution_199k.csv` | **Not an aggregate FP driver.** FP varies only 0.080 → 0.122 across a 34× speed range. Only 2,495 of 183,761 CDMs (1.4%) have v_rel < 500 m/s. Do **not** over-read this: CARA documents 2-D Pc errors up to **330×** for ~3 m/s conjunctions. Correct wording: v_rel is not an aggregate driver *in this corpus*. |
| HBR choice | `attribution_199k.csv` | **Large systematic.** Flat 5 m vs catalogue moves FN ~35% at every threshold (10⁻⁴: 0.528 → 0.723) with zero change in skill. RCS ±30% moves it only 3–5%. Catalogue HBR median 5.55 m, range 1.05–18.35 m — **not** 5 m. |
| Covariance staleness (CDM age) | `fpfn_by_leadtime_199k.csv` | **Dominates.** FP rises 30× from <0.5 d lead (0.00502) to 5–7 d (0.15063) at 3×10⁻⁷. FN far flatter (3.4% → 18.3%). |
| **Method error** | `attribution_method_199k.csv`, `attribution_method_validated.csv` | **NEW.** Smallest of the four. On the validated subset (35,429 CDMs where our covariance reproduces ESA's risk within 0.25 dex), switching closed form → exact changes FP by +5 (of 9,135) at 3×10⁻⁷, +8 (of 2,659) at 10⁻⁵, +3 (of 682) at 10⁻⁴, −1 (of 53) at 10⁻³. Median method error 1×10⁻⁴ dex, but **11.4% of CDMs off by >0.1 dex and 5.9% by >1 dex.** |

**Why method error is small here but catastrophic on CARA's cases:** throughout
this corpus `R << σ` (median R = 5.55 m, median σ_major = 2,019 m; **0.00% of
CDMs exceed `R_lim`**). On CARA's Alfano cases `R > σ`. Both facts are true;
they are different regimes. Do not generalise either.

**Covariance reconstruction (new, `src/cdm_covariance.py`):** the raw CDMs carry
the complete 6×6 RTN covariance element-by-element (`t_ct_r`, `t_crdot_r`,
`t_ctdot_rdot`, … = 15 unique elements). Assembling it reproduces the published
miss vector to **4.4×10⁻⁵ relative**, and ESA's own functional form to
**−0.009 dex median on near conjunctions** (χ² ≈ 3.6). On far conjunctions
(χ² ≈ 22.5) ESA's `risk` is not reproduced — the published block is evidently
not the block behind `risk` there. 5.6% of rows have missing covariance
elements; 95% are complete. The 2-D convention beats the 3-D one by ~100×, so
the element mapping is confirmed correct.

**Do not use the raw `mahalanobis_distance` column as the χ² in ESA's formula** —
it is a different quantity (~120× smaller than the 3-D relative-position χ²).
Compute the 2-D χ² from the miss vector and the encounter-plane covariance.

### 4.5 Layer 2 interpretability scorecard

`results/interpretability_scorecard.csv`. Five dimensions, each with an explicit
definition in `scripts/interpretability_scorecard.py`:

| method | D1 decomp | D2 monot | D3 valid | D4 trace | D5 audit | MEAN |
|---|---|---|---|---|---|---|
| closed_form | 1.000 | 1.000 | 0.857 | 0.725 | 0.000 | 0.716 |
| **exact_N2** | 0.500 | 1.000 | 1.000 | 0.722 | 1.000 | **0.844** |
| quadrature_N1 | 0.500 | 1.000 | 1.000 | 0.683 | 1.000 | 0.837 |
| monte_carlo | 0.000 | 1.000 | 1.000 | 0.857 | 1.000 | 0.771 |

Fitted partial sensitivities of log₁₀ Pc (theory: **+2.000, −1.000, −0.217**):

- closed_form: **+2.000, −1.000, −0.2171** — matches theory exactly.
- exact_N2: +2.166, −1.016, −0.2167.
- monte_carlo: +1.178, −0.573, −0.0932 — attenuated by sampling noise (300 CDMs
  × 100k samples; low-Pc CDMs yield few hits).

The closed form wins D1 (it literally factors into named terms) and loses D5
(it fails CARA validation). Eq. N-2 wins overall. This is the scorecard's actual
message: **analytic transparency and correctness are in tension, and the
correct-but-less-transparent method is the one to use.**

**D2 uses PARTIAL sensitivities, not marginal correlations.** Marginal Spearman
of `log(R/σ)` against Pc is **−0.65** — the wrong sign — because R and χ² covary
across CDMs. The unconfounded test is the fitted multivariate coefficient. If you
"simplify" D2 to a marginal correlation it will silently invert.

### 4.6 Layer 2 calibration

`results/calibration_*.csv`, 2 PNGs. Test log loss, LRP vs LRP+Platt:
3×10⁻⁷ → 0.5047 / **0.0759**; 10⁻⁶ → 0.4468 / **0.0789**; 10⁻⁵ → 0.1621 /
**0.0505**; 10⁻⁴ → 0.0354 / **0.0128**. LRP slope 0.20–0.80 (ideal 1.0) —
over-confident. Platt cuts log loss **6.6×** with PR-AUC unchanged
(0.6674 → 0.6651): ranking is fine, the probability scale is wrong. **Brier is
useless here** (identical to 4 dp for LRP and CRP, since Pc ≪ base rate).
Observed frequency is flat at ~10⁻³ across predicted Pc 10⁻²²…10⁻¹¹, rising only
in the top bin.

### 4.7 ESA `risk` reproduction

`validate_pipeline.py` check 3: ESA's operational `risk` reproduced from
published covariances to a median error of **+0.0012 dex**, 55.9% of rows within
0.05 dex. Functional form:
`risk = log10(R²/(2√det C₂)) − χ²/(2 ln 10)`, `R = (t_span + c_span)/2`.
41.4% of raw rows sit exactly on the −30 floor (harmless; every threshold used is
far above it).

### 4.8 Cross-provider disagreement (L0)

Truth 7.52%, predictions 10.84% — a floor no model can beat. True-HR count varies
**34×** (168 at 3×10⁻⁷ → 5 at 10⁻³) across thresholds.

### 4.9 LRP sweep (L0)

True-HR / F₂: SpaceX 3×10⁻⁷ → 168 / 0.7811; Kelvins 10⁻⁶ → 150 / 0.7391; ESA
10⁻⁵ → 56 / 0.6734; ESA/DLR 10⁻⁴ → 17 / 0.4945; JAXA 10⁻³ → 5 / 0.4545. At
10⁻⁶: CRP 0.2711, MAX 0.5915, MEAN 0.6688.

---

## 5. Environment facts and gotchas

**The environment destroys itself.** Dataset gone 5×, installed packages gone
2×, the CARA clone (including `.git`) gone 2×, Octave gone 2× — all in one
session, some mid-turn. `restore.sh` is verified working every time. Budget for
it: ~2.5 min.

**Never let a check silently skip.** An earlier version of `restore.sh` reported
`SKIP` when Octave was missing, which is how a result rots unnoticed. It now
re-clones CARA and installs Octave, and `run_cara.m` hard-errors if CARA fails
to reproduce its own published values.

**Octave specifics:**
- No `readtable`. Use `csvread(CSV, 1, 0)` for the numeric block.
- `xlsread` needs `pkg load io;` — and the `io` package **cannot read legacy
  `.xls`**. Extract with Python (`xlrd`) and feed Octave a CSV.
- Script-local functions are NOT hoisted. Inline them.
- `__import__("importlib.util")` returns the top-level package, not the
  submodule. Use a real `import importlib.util`.

**Column-index bugs are the recurring failure mode.** `run_cara.m` originally
hard-coded column indices; widening the CSV from 3×3 to 6×6 covariance silently
pointed them at position-*velocity* terms and fed CARA garbage. It now locates
columns **by header name**. Any script that indexes a CSV by position should be
converted.

**`edit_file` drops parameters** for text containing `> 1e-30` / `1e-100`. Use a
bash heredoc with `Path.read_text()` / `replace()` / `write_text()`.

**`sed` intermittently fails** with `shell_error`, `exit_code=null`, ~60 ms, no
stderr — often a malformed `cwd` (e.g. `home/user` missing the leading slash).

**`lru_cache` + `usecols` list → `TypeError: unhashable type: 'list'`.** Do not
re-decorate the public loaders in `kelvins_data.py`.

**`latest_cdm_per_event()` must stay `head(1)`**, not `tail(1)`. Rows are sorted
ascending by `time_to_tca`, so the *first* row is the CDM closest to TCA.
`tail(1)` silently returns the earliest CDM. This bug cost the LRP gate once.

**Memory:** ~1 GB RAM / 2 CPUs. The batch Pc kernels are chunked at 8,192 rows
for this reason. Do not raise the chunk size.

**Logistic calibration must stay L2-penalised.** Unpenalised MLE diverges under
perfect separation (slopes ~10¹⁰).

**Constant predictor collapses quantile binning** → `ECE = nan` for CRP. Fixed by
expanding degenerate edges; `pr_curve` early-returns `degenerate: True` when
`ptp(rp) == 0`.

**Kelvins CSVs vs raw:** the flattened competition CSVs have **no covariance**;
`raw_data_2015-2019.gz` does, but as individual elements. See `src/cdm_covariance.py`.

---

## 6. Attribution and equation-number corrections (do not regress these)

- The 1-D error-function reduction is **Alfano (2005a), refined by Elrod (2019)
  = Appendix N Eq. N-2**, NOT "Foster & Estes 1992". Foster & Estes (1992) is the
  **2-D quadrature = Eq. N-1** = `pc2d_numeric`.
- Appendix N has equations **N-1 … N-10 only**. There is no N-17, N-18 or N-23.
  Do not cite Appendix N equation numbers without opening the PDF.
- `foster2d` is a backwards-compatibility alias for `pc2d_1d`. Do not describe
  `pc2d_1d` as "Foster-style".
- CARA's multi-encounter bounds are Eqs. N-7/N-8:
  `max(Nc,k) ≤ Pc ≤ 1−Π(1−Nc,k)`, `Nc = Σ Nc,k`.

---

## 7. OPEN items

1. **Layer 3** — ML forecaster: predict final (last-CDM) risk from CDMs visible
   ≥2 days before TCA. Not started. The calibrated LRP+Platt model in
   `src/calibration.py` is the natural starting point.
2. **Layer 4** — live replay / SOCRATES ingest. Not started. **SOCRATES emits
   TLE/GP only — no covariance, no CDM history** — so the Layer 2 metrics do not
   transfer directly. Plan around this before starting.
3. **Layer 5** — one-command report. Not started.
4. **3-D Pc variant** (Layer 1, optional/credit). Not implemented. CARA's
   `Pc3D_Hall.m` is in the clone and the published Alfano output CSV carries a
   `CARA_Pc3D` column, so a reference is available.
5. **χ² ≥ 10 ESA residual.** For χ² ≥ 10, ESA reports `risk` far below the
   closed-form 2-D Pc: median |err| 1.336 dex vs 0.021 dex for χ² < 1. The
   position–velocity covariance block was the suspected cause; the assembled 6×6
   in `src/cdm_covariance.py` now makes this testable directly. **This is the
   highest-value next investigation** — it is the one unexplained result left in
   the completed layers.

---

## 8. Sources actually consulted

- **NASA CARA Best Practices Handbook, Appendix N** (rev. 1, update March 2024) —
  fetched and read in full, 13 chunks. Eq. N-1 (Foster & Estes 1992 2-D
  quadrature), Eq. N-2 (Alfano 2005a / Elrod 2019 1-D erf), Eqs. N-3/N-4
  (circumscribing square; USSPACECOM ASW CDMs often report P_sq — the Kelvins
  `risk` is the disk form), `R_lim = 4σ_x − min(R)`, the four conjunction-plane
  assumptions (statistical independence, rectilinear motion, Gaussian positions
  with negligible velocity uncertainty, temporal isolation), the ~330× worked
  example, Fig. N-7 (63,603 real CARA conjunctions 2017-05-01…2019-08-15; 62,913
  agree at p>10⁻³, 254 at p≤10⁻³, 436 at p≤10⁻⁶ → ~1.1% violate the assumptions;
  2-D/3-D Nc match MC far better), Eqs. N-7/N-8 (multi-encounter), Eq. N-10
  (decorrelated joint covariance, Casali et al. 2018), defective-covariance
  classes (null, default 10 Earth radii² diagonal, non-PSD → eigenvalue
  clipping).
- **`github.com/nasa/CARA_Analysis_Tools`** — public, NOSA GSC-18593-1 /
  18848-1 / 19027-1 / 19374-1, last commit 2026-09-08. The full Pc SDK plus the
  Alfano test cases with published expected values.
- **CelesTrak** — CDM format docs (KVL parser field set) and TLE format docs
  (80-column offsets, implied decimal point in eccentricity, implied exponent
  sign in B*/ndot/nddot, two-digit epoch year rule).
- **Uriot et al. 2021**, arXiv 2008.03069 — the Kelvins challenge paper. Metric
  L(r̂) = (1/F₂)·MSE_HR; F_β = (1+β²)pq/((β²p)+q), β=2; high-risk r ≥ −6.
  Eligibility: ≥2 CDMs, first CDM ≥2 d before TCA, last CDM <1 d; 150 high-risk
  test / 66 train. LRP (Eq. 6): r̂_i = r₋₂,ᵢ if ≥ −6 else −6.001. CRP: r̂ = −5 →
  L = 2.5. Table 3: LRP test F₂ 0.739 / L 0.694 / leaderboard 0.718; top teams
  TU Delft 0.104, Andes 0.152, GEOSTAT 0.185. Table 2: 15,321 events, 199,082
  CDMs. ESA thresholds 10⁻⁵…10⁻⁴. 97 teams, 12 beat LRP.
- **Zenodo 10.5281/zenodo.4463683** — public mirror of the dataset, CC-BY.
- **JAXA JMR-016(E)** §6.2.2 — Risk Level 1 Pc ≥ 10⁻⁵, Level 2 ≥ 10⁻⁴, Level 3
  ≥ 10⁻³.
- **ESA SDC8 (2021), paper 64** — "ESA set a threshold of PoC>10⁻⁴ for HIEs, as
  does DLR and CSA, while JAXA uses a threshold of 10⁻³."
- **SpaceX Starlink** autonomous avoidance policy — manoeuvre above ~3×10⁻⁷,
  ~300× more conservative than 10⁻⁴; post-manoeuvre residual ~10⁻⁶.
- **CelesTrak SOCRATES** — runs 3× daily, screens the next 7 days against the
  full unclassified GP/TLE catalogue. TLE/GP only.
- **D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.
  (2013)** — cited by chapter (2, 9, 11; eq. 9-35). **Not read; no copy in the
  workspace.** See §1.

---

## 9. Standing instruction from the user

> "Strictly perform every task with fully 100% accuracy everywhere."

Every "DONE" above means *a named artifact plus a passing check*. Where an item
has neither, it is listed in §7 OPEN. The learning item (§1) is the one place
where "done" rests on self-attestation rather than an external check, and the
Vallado citation is unverified.
