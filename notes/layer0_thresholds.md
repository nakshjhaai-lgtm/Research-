# Layer 0 — Definitions: "you can't optimise an undefined quantity"

> *Configurable FP/FN definitions per threshold. Every result reported as a
> threshold sweep, never one number. Cross-provider disagreement as a
> first-class metric.*

Code: `src/thresholds.py`, `scripts/threshold_sweep.py`.

---

## 1. Why "high risk" is a policy, not a measurement

There is no physical boundary at which a conjunction becomes dangerous. What
exists is a **policy choice** per operator, and that choice *defines the
positive class* — and therefore defines what a false positive and a false
negative even are. Optimising F2 without naming the threshold is optimising an
undefined quantity.

The thresholds used here, with primary sources:

| Operator | Pc threshold | Source |
|---|---|---|
| SpaceX Starlink (autonomous) | **3 × 10⁻⁷** | SpaceX published autonomous-avoidance policy; manoeuvre when Pc exceeds ~3 in 10 million |
| Kelvins competition (notification) | 10⁻⁶ | Uriot et al. 2021 |
| ESA (lower bound of mission range) | 10⁻⁵ | "ESA missions in LEO adopt reaction thresholds ranging between 1e-5 and 1e-4" |
| ESA / DLR / CSA / 18 SPCS / NASA CARA | **10⁻⁴** | ESA SDC8 (2021): "ESA set a threshold of PoC>10⁻⁴ for HIEs, as does DLR and CSA"; CARA handbook |
| JAXA | **10⁻³** | JAXA-JMR-016(E) §6.2.2, risk level 3 ("urgent") |

That is a spread of **3.5 orders of magnitude** in the definition of the thing
being predicted.

---

## 2. The threshold set is configurable

```python
from src.thresholds import RiskThreshold, PROVIDER_THRESHOLDS, threshold_from_log10

# use the defaults, or define your own
mine = threshold_from_log10([-3.0, -4.0, -5.0, -6.522878745280337])
mine = [RiskThreshold("mine", "My operator", -4.0, "internal policy", role="reaction")]
```

The `RiskThreshold` dataclass carries the provider, the log10 threshold, the
role (notification / reaction / autonomous manoeuvre) and a citation, so a
sweep table is self-documenting.

---

## 3. Every result is a sweep

`sweep(r_true, r_pred)` returns one row **per threshold**:

```
provider                                      log10T   nHR  flag   TP   FP   FN   prec    rec      F2
SpaceX Starlink (autonomous)                  -6.523   168   237  142   95   26 0.5992 0.8452  0.7811
Kelvins competition                           -6.000   150   178  115   63   35 0.6461 0.7667  0.7391
ESA (lower bound of mission range)            -5.000    56    73   40   33   16 0.5479 0.7143  0.6734
ESA / DLR / CSA / 18 SPCS / NASA CARA         -4.000    17    23    9   14    8 0.3913 0.5294  0.4945
JAXA                                          -3.000     5     2    2    0    3 1.0000 0.4000  0.4545
```

**The LRP baseline's F2 ranges from 0.781 to 0.455 purely because of the
definition of "high risk".** Quoting "F2 = 0.739" without the threshold is
quoting a number that has no meaning.

### 3.1 The clip floor is threshold-relative

A subtlety that materially changes the sweep. The Kelvins paper notes that once
an event is predicted low-risk, the optimal prediction is `T − ε`. That floor is
**relative to the threshold**, so it moves with the sweep:

```
r̂ = r      if r >= T
  = T − ε   otherwise        (ε = 0.001)
```

Re-using the single fixed floor −6.001 at the SpaceX threshold (−6.523) makes
*every* low-risk event look high-risk — LRP would flag all 2,167 events with
recall 1.000 and precision 0.078. Applying the floor per threshold instead
gives the correct 237 flagged / precision 0.599. `sweep()` applies the clip
internally for exactly this reason; pass it **raw** model output.

---

## 4. Definition sensitivity: the problem itself changes

| Threshold | true high-risk events | share of test set |
|---|---|---|
| 3 × 10⁻⁷ (SpaceX) | 168 | 7.75 % |
| 10⁻⁶ (Kelvins) | 150 | 6.92 % |
| 10⁻⁵ (ESA lower) | 56 | 2.58 % |
| 10⁻⁴ (ESA/DLR) | 17 | 0.78 % |
| 10⁻³ (JAXA) | **5** | 0.23 % |

The positive class varies by a factor of **34**. At the JAXA threshold the
reported F2 = 0.4545 rests on **five** positive examples — two significant
figures on a sample of five is numerology, not measurement. Any model
comparison at that threshold should be reported with the count attached.

---

## 5. Cross-provider disagreement (first-class)

`disagreement(values)` measures how much the *decision* depends on whose rules
you apply. Because the thresholds are ordered, an event's decision vector is a
staircase, and two providers disagree exactly on events whose risk falls
between their two thresholds.

**On the LRP predictions:**

```
overall disagreement rate     : 10.84 %
all providers act             :  0.09 %   (2 events)
no provider acts              : 89.06 %   (1930 events)
max pairwise disagreement     : 10.84 %   (spacex_auto vs jaxa)
mean providers acting / event : 0.237 of 5
histogram of #providers acting: {0: 1930, 1: 59, 2: 105, 3: 50, 4: 21, 5: 2}
```

Pairwise matrix (fraction of events where the two providers disagree):

| | spacex | kelvins | esa_lower | esa_dlr | jaxa |
|---|---|---|---|---|---|
| **spacex** | 0 | 2.72 % | 7.57 % | 9.88 % | **10.84 %** |
| **kelvins** | | 0 | 4.85 % | 7.15 % | 8.12 % |
| **esa_lower** | | | 0 | 2.31 % | 3.28 % |
| **esa_dlr** | | | | 0 | 0.97 % |
| **jaxa** | | | | | 0 |

**On the true final risk** the disagreement is 7.52 % — i.e. even with perfect
knowledge, roughly one event in thirteen is classified differently depending on
which operator's rulebook you use. That is a floor on cross-provider
disagreement that no model can remove.

The widest gap is always SpaceX-vs-JAXA (3 × 10⁻⁷ vs 10⁻³, 3.5 dex apart):
**10.8 % of events get opposite decisions.**

---

## 6. The FP/FN trade-off is a knob, and it is yours to set

Two independent knobs, both explicit in the code so neither can hide in a
default:

**(a) The cost ratio.** `operational_cost(df, cost_fp, cost_fn)` prices each FP
as a wasted manoeuvre and each FN as a missed conjunction:

| cost(FN)/cost(FP) | cheapest threshold | dearest threshold |
|---|---|---|
| 1 | JAXA (0.014/event) | Kelvins 1e-6 (0.191) |
| 10 | JAXA (0.014) | Kelvins 1e-6 (0.191) |
| 100 | JAXA (0.138) | Kelvins 1e-6 (1.644) |
| 1000 | JAXA (1.384) | Kelvins 1e-6 (16.18) |

The ordering of thresholds by cost is stable here, but the *gap* grows by three
orders of magnitude as the ratio moves — and with a different predictor the
ordering itself can invert. The ratio is a policy input, never a measurement.

**(b) The margin.** `sweep(..., margin=δ)` shifts the *prediction* threshold
while the provider's definition stays fixed, trading FP for FN at fixed model
output:

| threshold | margin | FP | FN | precision | recall | F2 |
|---|---|---|---|---|---|---|
| 3 × 10⁻⁷ | −0.5 | 1999 | 0 | 0.078 | 1.000 | 0.296 |
| 3 × 10⁻⁷ | 0.0 | 95 | 26 | 0.599 | 0.845 | **0.781** |
| 3 × 10⁻⁷ | +0.5 | 63 | 50 | 0.652 | 0.702 | 0.692 |
| 10⁻⁴ | −0.5 | 2150 | 0 | 0.008 | 1.000 | 0.038 |
| 10⁻⁴ | 0.0 | 14 | 8 | 0.391 | 0.529 | **0.495** |
| 10⁻⁴ | +0.5 | 4 | 13 | 0.500 | 0.235 | 0.263 |

At 10⁻⁴ the F2-optimal margin is 0.0; at 3 × 10⁻⁷ it is also 0.0, but the
sensitivity to the margin differs by an order of magnitude. A model tuned to
maximise F2 at one threshold with one margin is not transferable.

---

## 7. What this means for the layers above

1. **Never report a single F2.** Report the sweep, with `nHR` (the positive
   count) in the same table.
2. **State the threshold with every metric**, and cite whose it is.
3. **Report disagreement alongside the metric.** A model can be excellent at
   one threshold and vacuous at another; the disagreement table is what makes
   that visible.
4. **Small positive classes are a warning, not a result.** At 10⁻³ the test set
   holds five high-risk events.
5. **The clip floor moves with the threshold** — see §3.1.
