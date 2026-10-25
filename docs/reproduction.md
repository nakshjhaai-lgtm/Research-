# Running it

Two levels of effort are possible here. The baseline gate needs Python and the
dataset and takes a few minutes. The CARA validation needs GNU Octave and a clone
of NASA's toolkit, takes longer, and is only worth doing if you want to check the
Pc claims rather than take them on trust.

Nothing in this repository contains the dataset. It is 221 MB and public.

---

## 1. Bare minimum: reproduce the baseline

```bash
git clone https://github.com/nakshjhaai-lgtm/Research-.git
cd Research-

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the published numbers were produced with
(numpy 2.3.5, pandas 2.2.3, scipy 1.17.1, scikit-learn 1.6.1, matplotlib 3.10.9,
sgp4 2.27). Newer versions will almost certainly work; if a number comes out
slightly different, that is the first thing to suspect.

Then fetch the data:

```bash
python scripts/fetch_data.py          # download, unpack, place
python scripts/fetch_data.py --check  # verify what is already there
```

The script pulls Zenodo record 4463683, unpacks it to
`/home/user/data/kelvins/Collision Avoidance Challenge - Dataset/`, and extracts
`train_data.zip` along the way. That path is a leftover from where the work was
done, so on any other machine use the environment variable instead:

```bash
export KELVINS_DATA_ROOT=/path/to/Collision\ Avoidance\ Challenge\ -\ Dataset
```

`src/kelvins_data.py` reads `KELVINS_DATA_ROOT` first and only falls back to the
hard-coded path. Everything else in `scripts/` assumes you run it from the
repository root, because each script inserts the parent directory into
`sys.path` before importing `src`.

Run the gate:

```bash
python scripts/reproduce_lrp.py
```

Expected tail:

```
F2 = 0.7391   MSE_HR = 0.5129   L = 0.6940
GATE: PASS  (target F2 = 0.739 +/- 0.02)
```

The per-event predictions land in `results/lrp_predictions.csv` and the summary
row in `results/lrp_metrics.csv`.

Run the tests:

```bash
python -m pytest tests/ -q       # or: python tests/test_orbital.py
```

Ten tests covering Keplerian element conversion, TLE parsing, SGP4 wrappers and
the J2 rates. They are the reason three of the four documented bugs were caught.

---

## 2. The other result tables

Each script writes one or more files into `results/`. They are independent, apart
from needing the dataset:

```bash
python scripts/validate_pipeline.py        # parser and 2-D Pc checks against ESA's own columns
python scripts/validate_foster_mc.py       # Eq. N-2 vs Eq. N-1 vs Monte Carlo, synthetic cases
python scripts/threshold_sweep.py          # layer 0: five providers, one set of predictions
python scripts/threshold_fpfn_199k.py      # headline FP/FN tables over 183,761 non-final CDMs
python scripts/calibration.py              # reliability, calibration regression, PR curves, plots
python scripts/attribution_199k.py         # FP/FN by lead time, relative speed, HBR variant
python scripts/attribution_method.py       # method error as an attribution axis
python scripts/interpretability_scorecard.py
python scripts/demo_pipeline.py            # a walkthrough of the orbital layer, no dataset needed
```

`demo_pipeline.py` is the one to run first if you want to see the maths work
before touching the dataset. It takes a pair of element sets, builds states, projects the covariance
into the encounter plane and prints a Pc, with each step visible.

---

## 3. Full restore, including CARA

`scripts/restore.sh` exists because the working environment reset repeatedly
during the project: the dataset was deleted five times, and pip packages do not
persist between sessions at all. The script reinstalls packages, re-fetches the
data, re-clones CARA, installs Octave if needed, and then runs every script as a
check.

```bash
bash scripts/restore.sh            # full restore, then 11 checks
bash scripts/restore.sh --check    # verify only, download nothing
```

Expected tail is eleven `PASS` lines followed by
`RESTORE COMPLETE -- all checks pass`.

If you would rather do the CARA part by hand, the sequence is:

```bash
git clone --depth 1 https://github.com/nasa/CARA_Analysis_Tools.git \
    /home/user/CARA_Analysis_Tools
sudo apt-get install -y --no-install-recommends octave octave-io

python scripts/extract_alfano_cases.py      # .xls -> /tmp/alfano_cases.csv
octave --no-gui --quiet scripts/run_cara.m  # CARA's code -> /tmp/cara_reference.csv
python scripts/validate_against_cara.py     # ours vs CARA
```

`run_cara.m` hard-errors if CARA fails to reproduce its own published values, on
purpose. A check that quietly skips when a dependency is missing is how a result
rots without anybody noticing.

---

## 4. Regenerating the figures in this documentation

```bash
pip install matplotlib pillow
python tools/make_figures.py
```

Every plot in `assets/` is drawn from a CSV in `results/`, so the pictures cannot
drift away from the numbers. The two animations are sequences of matplotlib
frames assembled into GIFs with Pillow. This is the only part of the repository
that needs Pillow, and none of the analysis imports it.

---

## 5. Things that will bite you

**Column indices.** The recurring bug in this codebase was positional CSV
indexing. `run_cara.m` used to hard-code column numbers and silently fed CARA
position–velocity terms when the covariance block widened from 3×3 to 6×6. Every
script now locates columns by header name. If you add a column, do not shift an
index.

**`head(1)`, not `tail(1)`.** The rows are sorted by `time_to_tca` ascending, so
the first row per event is the CDM closest to TCA, which is the latest one. This
one word was worth 0.22 of F₂.

**Memory.** The batch Pc kernels are chunked at 8,192 rows for a reason: the
intended machine had about 1 GB of RAM and two cores. Raising the chunk size will
swap.

**Octave quirks,** if you touch `run_cara.m`: there is no `readtable`, use
`csvread(CSV, 1, 0)`; the `io` package cannot read legacy `.xls`, which is why
`extract_alfano_cases.py` exists and does it in Python; and script-local
functions are not hoisted, so they have to be inlined.

**Calibration must stay L2-penalised.** Unpenalised logistic regression diverges
under perfect separation, with slopes around 10¹⁰.

**The `risk` column in `test_data.csv` is a feature, not a label.** The ground
truth is `test_data_private.csv`, one row per event, with the removed final CDM
renamed `true_risk`. Scoring against the wrong file produces a plausible and
completely meaningless number.

**`mahalanobis_distance` is not the χ² in ESA's risk formula.** It is a different
quantity, roughly 120× smaller than the three-dimensional relative-position χ².
Compute the 2-D χ² from the miss vector and the encounter-plane covariance.

---

## 6. What a clean run looks like

| Check | Script | Expected |
|---|---|---|
| Baseline gate | `reproduce_lrp.py` | F₂ = 0.7391, target 0.739 ± 0.02 |
| Orbital regression | `tests/test_orbital.py` | 10 tests pass |
| Parser and Pc | `validate_pipeline.py` | determinant ratios 1.000000, ESA risk median +0.0012 dex |
| Pc methods | `validate_foster_mc.py` | Eq. N-2 within 1e-7 of the Rayleigh anchor |
| Our Pc vs CARA | `validate_against_cara.py` | Eq. N-2 worst rel err 8.2e-7 |
| Everything else | the sweep, calibration and attribution scripts | exit 0, files refreshed in `results/` |
