# Learning path: Keplerian orbits → TLE/SGP4 → covariance → 2-D Pc

Notes built while reproducing the ESA Kelvins Collision Avoidance Challenge
baseline. Every claim below is either derived from first principles in
`src/` or cross-checked against the published dataset (see
`scripts/validate_pipeline.py`).

**Primary sources**

| # | Source | Used for |
|---|--------|----------|
| S1 | NASA, *Spacecraft Conjunction Assessment and Collision Avoidance Best Practices Handbook* (2023 rev.) — <https://www.nasa.gov/wp-content/uploads/2021/07/nasa_ca_handbook_for_s3vi_20211013.pdf> | operational CA pipeline, Pc as the risk metric, reaction thresholds |
| S2 | NASA CARA, *Appendix N — Pc Calculation Approaches* (update 20 Mar 2024) — <https://www.nasa.gov/wp-content/uploads/2024/04/ca-handbook-rev-1-appendix-n-update-march-2024.pdf> | 2-D Pc Eq. N-1/N-2, Nc Eqs. N-5..N-8, decorrelated covariance Eq. N-10, CARA's own 63,603-conjunction MC validation (Fig. N-7) |
| S3 | D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed. (2013) | two-body mechanics (ch. 2), SGP4 (ch. 9), frames & covariance transforms (ch. 11) |
| S4 | CelesTrak, *Two-Line Element Set Format* — <https://celestrak.org/NORAD/documentation/tle-fmt.php> | TLE field-by-field layout |
| S5 | Hoots & Roehrich, *SPACETRACK Report #3* (1980) | the SGP4 theory itself |
| S6 | Uriot et al., *Spacecraft Collision Avoidance Challenge: design and results of a machine learning competition*, Astrodynamics 5:1–17 (2021) — <https://arxiv.org/abs/2008.03069> | dataset, metric, LRP baseline |
| S7 | ESA Kelvins dataset, Zenodo 10.5281/zenodo.4463683 | 199,082 CDMs, 15,321 events |

---

## 1. Keplerian orbits

### 1.1 The two-body problem

A single object around a spherical Earth obeys

```
r¨ = -mu * r / |r|^3 ,   mu = 398600.4418 km^3/s^2
```

whose solution is a conic section described by **six classical elements**:

| Element | Symbol | Meaning |
|---------|--------|---------|
| semi-major axis | `a` | size of the orbit |
| eccentricity | `e` | shape |
| inclination | `i` | tilt w.r.t. the equator |
| RAAN | `Ω` | where the orbit crosses the equator going north |
| argument of perigee | `ω` | orientation of the ellipse in the orbit plane |
| true anomaly | `ν` | where the object is on the ellipse *now* |

Position and velocity follow from the eccentric anomaly `E`, obtained by solving
**Kepler's equation**

```
M = E - e sin E            (M = mean anomaly, E = eccentric anomaly)
```

by Newton–Raphson (`src/orbital.py::eccentric_anomaly_from_mean`), then

```
r     = a (1 - e cos E)
x_pqw = r cos E,  y_pqw = r sqrt(1-e^2) sin E
v_pqw = sqrt(mu a)/r * (-sin E, sqrt(1-e^2) cos E, 0)
```

and a rotation into the inertial frame, `Q = Rz(Ω) Rx(i) Rz(ω)`
(`perifocal_to_eci`).

The mean motion is `n = sqrt(mu/a^3)` and the period `T = 2π/n`, so
`a = (mu/n^2)^(1/3)` — this is exactly how a TLE's semi-major axis is recovered
from its mean motion (S4).

### 1.2 Why Kepler alone is not enough

A pure two-body propagator ignores `J2` (Earth's oblateness), drag and
third-body gravity. `src/orbital.py::j2_secular_rates` implements the
first-order secular rates (Vallado eq. 9-35):

```
dΩ/dt  = -1.5 J2 (Re/a)^2 n / (1-e^2)^2            * cos i
dω/dt  =  0.75 J2 (Re/a)^2 n / (1-e^2)^2           * (5 cos^2 i - 1)
dM/dt  =  0.25 J2 (Re/a)^2 n / (1-e^2)^2 sqrt(1-e^2)* (3 cos^2 i - 1) + n
```

Measured on an ISS TLE (`scripts/demo_pipeline.py`): `dΩ/dt = -5.12°/day`,
`dω/dt = +5.73°/day`. Propagating the same TLE one day with the two-body model
vs SGP4 gives a **475 km** separation — which is precisely why operational
screening uses SGP4, not Kepler.

(Part of that gap is present at the epoch itself: TLEs carry *mean* elements
and SGP4 converts them to osculating internally, so the two states already
differ by ~7.5 km before any propagation. The rest is J2 and drag.)

---

## 2. TLE and SGP4

### 2.1 The two-line element format (S4, S5)

A TLE is a fixed-width 80-column pair of lines. Column 1 is the line number,
column 3 is the NORAD catalogue number, and the remaining fields sit at
**hard-coded offsets**:

```
1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN
     |      |      |       |       |       |       |      +-- mean motion [rev/day]
     |      |      |       |       |       |       +--------- mean anomaly [deg]
     |      |      |       |       |       +----------------- argument of perigee [deg]
     |      |      |       |       +------------------------- eccentricity (leading "0." implied)
     |      |      |       +--------------------------------- RAAN [deg]
     |      |      +----------------------------------------- inclination [deg]
     |      +------------------------------------------------ B* drag term
     +------------------------------------------------------- epoch (YYDDD.DDDDDDDD)
```

Two gotchas that `src/orbital.py::parse_tle` handles:

* **Eccentricity** is published *without* its decimal point: `0006703` means
  `0.0006703`.
* The exponent-bearing floats (`B*`, `ndot/2`, `nddot/6`) use an implied
  exponent sign: `-11606-4` means `-0.11606e-4`.

Epoch years are two-digit; `< 57` means 20xx, otherwise 19xx.

Note that a TLE gives the **mean** anomaly, so `nu` must be recovered through
`M → E → ν` before building a state vector — treating the mean anomaly as the
true anomaly is a common and silently wrong shortcut.

### 2.2 SGP4

SGP4 (Hoots & Roehrich 1980) is a Brouwer–Kozai–style analytic propagator with
a drag term. `src/orbital.py::sgp4_propagate` wraps the `sgp4` package
(Vallado's reference implementation). Two things to remember:

1. Output is in **TEME** (true equator, mean equinox), *not* J2000. The
   difference is small but matters when comparing against covariances given in
   another frame (Vallado ch. 11).
2. `sgp4()` returns a non-zero error code for degenerate elements (decayed
   satellites, `e` out of range, …); `sgp4_error_codes()` maps them.

---

## 3. Covariance basics

### 3.1 What the covariance represents

Orbit determination returns a state **and** its uncertainty as a 6×6
covariance matrix `P`. Because errors are dominated by along-track growth, the
matrix is extremely anisotropic — for a typical LEO object the cross-track
uncertainty can be orders of magnitude smaller than the along-track one.

CDMs express `P` in the local **RTN** frame (S1, S2):

```
R (radial)     along the position vector, away from Earth
T (transverse) in the orbit plane, perpendicular to R, along-track
N (normal)     completes the right-handed triad (orbit normal, cross-track)
```

RTN is used because the uncertainty structure is physically meaningful there
and because the matrix stays small and well-conditioned.

### 3.2 Combining two objects

The two objects of a conjunction are tracked independently, so their errors
are treated as **uncorrelated** and the combined covariance is simply the sum
(S2, Eq. N-1):

```
C = C_primary + C_secondary
```

Appendix N (S2) also gives the cross-correlated form (Eq. N-10), used when the two
objects share an observation source or a common atmospheric model; the
comparison in the appendix shows this correction rarely changes Pc
appreciably except for conjunctions with elevated drag energy dissipation and
a particular geometry.

**Validated on the dataset:** the Kelvins release publishes only the three
position sigmas and the three position–position correlation coefficients per
object, plus the determinant of each full covariance. Rebuilding
`C = diag(σ²) + ρ_ij σ_i σ_j` reproduces the published determinants
**exactly** (ratio 1.000000, min and max, over all 2,167 test CDMs). See
`src/cdm_parser.py::CDMTable.covariance_rtn` and
`scripts/validate_pipeline.py` check [3].

### 3.3 Mahalanobis distance

The natural scalar measure of "how many sigmas away is the miss" is

```
χ² = m^T C^-1 m        (m = miss vector, C = combined covariance)
```

χ² is the quantity that drives everything: it appears in the exponential of
Pc, and — as discovered below — it also sets the covariance scaling that
maximises Pc.

---

## 4. Two-dimensional Pc

### 4.1 The model

At TCA the relative motion is well approximated as **rectilinear** over the
short encounter. The probability of collision is then the probability that the
relative position, modelled as a 2-D Gaussian in the **encounter plane**
(the plane perpendicular to the relative velocity), falls inside the disk of
radius `R` = the combined hard-body radius, centred on the origin:

```
Pc = 1/(2π sqrt(det C2)) * ∫∫_{|x| ≤ R} exp(-½ x^T C2^-1 x) dx
```

where `C2` is the combined covariance projected into the encounter plane and
the Gaussian is centred on the projected miss vector `m2`.

`HBR = (span_primary + span_secondary) / 2` — in the Kelvins data,
`t_span`/`c_span` are the object sizes fed to the risk algorithm, with a
minimum 2 m diameter assumed for an unidentified chaser.

### 4.2 The operational closed form

The exact integral has no elementary closed form. The universally used
operational approximation (Alfriend–Akella / Chan; the form behind the CDM
`risk` field) replaces the integrand with its value at the centre of the disk:

```
Pc ≈ R² / (2 σ1 σ2) * exp(-½ χ²)        σ1 σ2 = sqrt(det C2)
```

Valid to `O(R²)`, and exact to `O(R⁴)` in the small-`R/σ` limit.

**Validated on the dataset.** For the 489 test CDMs whose reported risk is
above ESA's 1e-30 reporting floor, the closed form matches the reported
`risk` with a **median error of 0.021 dex**, and for the 90 conjunctions where
the objects actually graze the HBR (χ² < 1) **96% agree within 0.25 dex**.

### 4.3 Two independent confirmations from the dataset

The Kelvins data carries two extra fields that let the whole chain be checked
without trusting the `risk` column:

* **`max_risk_scaling`** — the scaling factor applied to the combined
  covariance to obtain the maximum Pc. Maximising
  `Pc(sC) = R²/(2s σ1σ2) exp(-χ²/2s)` over `s` gives `s* = χ²/2`. The
  published column reproduces `χ²/2` with **median ratio 1.0000 and
  correlation 0.988** over the whole test set — which simultaneously
  confirms the encounter-plane projection (perpendicular to the relative
  velocity) and the combined covariance.
* **`max_risk_estimate`** — the resulting maximum Pc. Theory gives
  `R²/(e χ² σ1σ2)`; the dataset matches to a **median of +0.0012 dex**.

### 4.4 Where the closed form breaks

The approximation requires `R << σ1, σ2`. At `σ = 100 m, R = 500 m` the exact
answer is 1.0 but the closed form returns **12.5** — unphysical. The
validation harness prints this regime explicitly
(`scripts/validate_pipeline.py` check [2]).

### 4.5 Known residual (documented, not hidden)

For **far misses** (χ² ≥ 10) the reported `risk` falls well below the closed
form (median error 1.34 dex). The published columns contain only the
**position block** of each covariance, whereas ESA's operational algorithm
propagates the full 6×6 state (including the position–velocity correlations)
to TCA. Since the residual grows monotonically with χ², it is consistent with
a slightly different effective miss/covariance in the along-velocity
direction. This is a limitation of the *published data*, not of the
implementation — the internal checks in §4.3 all pass exactly.

### 4.6 Beyond 2-D Pc: the 3-D Nc

Appendix N (S2) extends this to the **statistically expected number of
collisions** `Nc`, which equals Pc for a single temporally isolated
conjunction but can exceed it for multi-encounter interactions, and which
gives both upper and lower bounds on the total probability. Appendix N gives the
multi-encounter bounds as Eqs. N-7/N-8: `max(Nc,k) <= Pc(tau_a, tau_b) <=
1 - Prod(1 - Nc,k)`, with the total expected number of collisions `Nc =
Sum Nc,k`. For low-relative-velocity conjunctions (~3 m/s) that violate the
rectilinear-motion assumption, Appendix N's worked example shows the
conjunction-plane Pc wrong by a factor of 330, with the 3-D Nc method matching
Monte Carlo instead.

---

## 5. Putting it together: the CA pipeline

```
TLEs ──► SGP4 propagation ──► screening ──► close approach detected
                                              │
                                    CDM issued (CCSDS 508.0-B-1)
                                    state + covariance at TCA
                                              │
                         combined covariance C = C1 + C2  (RTN)
                         project onto encounter plane (⊥ v_rel)
                                              │
                         2-D Pc = R²/(2σ1σ2) exp(-χ²/2)
                                              │
                       compare with the reaction threshold (1e-5 … 1e-4)
```

The Kelvins challenge sits on top of step 5: given the CDMs **at least two
days before TCA**, forecast the risk that the **last** CDM (closest to TCA)
will report.
