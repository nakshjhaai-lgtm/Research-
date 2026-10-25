# What this study is about

A satellite operator does not ask "will these two objects collide?". Nobody can
answer that. The operator asks "is the probability of collision above the number
my organisation treats as worth manoeuvring for?", and then decides whether to
spend fuel. Everything in this repository sits inside that second question.

The work started as a reproduction exercise. ESA ran a machine-learning
competition in 2019 where teams predicted, from a sequence of Conjunction Data
Messages, whether a conjunction would end up high risk. The competition paper
publishes a deliberately dumb baseline called Latest Risk Prediction: take the
risk number from the most recent CDM and use it as the forecast. Reproducing
that baseline to the published score is the first gate, because a study of
uncertainty that cannot reproduce a known result is opinion.

It became something more interesting once the baseline was in place and the
question turned to definitions. Here is the thing the competition scores do not
tell you: at the threshold every operator actually uses, and with the Pc method
every operator actually runs, the number of high-risk events in the same test
set changes by a factor of thirty-four. That is not a modelling choice. It is
two organisations disagreeing about what "high risk" means, and both writing it
down in policy documents.

---

## 1. The setting

Two objects in low Earth orbit. One is a satellite, the other usually debris. A
conjunction is flagged, CDMs start arriving, and the gap between the predicted
positions at closest approach, the time to conjunction or TCA, is compared
against the combined size of the two objects.

Conjunction Data Messages in this dataset were produced by the US 18th Space
Control Squadron (now the 18th Space Defense Squadron) and obtained through
Space-Track; ESA's Space Debris Office curated and anonymised the four-year
collection. Each message carries, for both objects, a state vector, a covariance
matrix in the radial–along-track–cross-track frame, the hard-body radius, and a
probability of collision computed by the provider from all of the above. The
Kelvins release is 199,082 CDMs covering 15,321 conjunctions between 2015 and
2019.

Three numbers matter throughout:

- **Pc**, the probability of collision. Usually written as log₁₀, so a value of
  −6 means one in a million.
- **The reaction threshold**, the Pc above which an operator will consider a
  manoeuvre. This is a policy number, not a physical one.
- **F₂**, the competition's score. It is an F-score with β = 2, which weights
  recall twice as heavily as precision, because a missed conjunction is worse
  than a wasted manoeuvre.

Two objects, two covariances, one threshold. The rest is bookkeeping that has to
be exactly right, which is where most of the effort went.

---

## 2. Layer 0: the threshold is the experiment

There is a table in [`notes/layer0_thresholds.md`](../notes/layer0_thresholds.md)
sourced from policy documents rather than from convenience:

| Operator | Reaction threshold | Where it comes from |
|---|---|---|
| SpaceX Starlink, autonomous | ≈ 3 × 10⁻⁷ | published autonomous-avoidance policy |
| Kelvins competition | 10⁻⁶ | competition paper |
| ESA, lower end of mission range | 10⁻⁵ | ESA mission practice |
| ESA, DLR, CSA, 18 SPCS, NASA CARA | 10⁻⁴ | ESA SDC8 (2021), CARA handbook |
| JAXA | 10⁻³ | JAXA-JMR-016(E) §6.2.2 |

Every table in this study therefore reports a sweep over those thresholds and
never a single number. The same code, the same predictions, five different
definitions of the positive class:

| Threshold | events that count as high risk | F₂ |
|---|---|---|
| 3 × 10⁻⁷ | 168 | 0.781 |
| 10⁻⁶ | 150 | 0.739 |
| 10⁻⁵ | 56 | 0.673 |
| 10⁻⁴ | 17 | 0.495 |
| 10⁻³ | 5 | 0.455 |

At 10⁻³ the whole metric rests on five events. Two significant figures on a
sample of five is numerology. The row is in the table for completeness of the
sweep, and it should not be read as a measurement.

There is a second finding hiding in the same place. Even if the forecast were
perfect, the classification of an event depends on whose rulebook you use.
Comparing all five policies pairwise on the true final risk, 7.52% of events are
classified differently by at least one operator pair. On realistic predictions
the disagreement is 10.84%. That is a floor. No model removes it, because it is
built into the definitions rather than into the data.

---

## 3. Layer 1: computing Pc, and checking it against NASA

Four independent implementations were written, so they can police each other:

1. the Chan (2008) small-hard-body closed form everybody quotes;
2. Appendix N Eq. N-2, the 1-D error-function reduction credited to Alfano
   (2005a) and refined by Elrod (2019);
3. Appendix N Eq. N-1, Foster and Estes (1992), 2-D quadrature;
4. a six-dimensional Monte Carlo that draws the state, propagates to TCA and
   counts hits, following CARA's own methodology.

The naming in the literature is a trap worth recording. The 1-D error-function
form is *not* Foster and Estes. Appendix N says plainly that Foster and Estes
applied a two-dimensional quadrature. An earlier version of the notes had this
wrong, and the corrections are documented in
[`notes/layer1_pc_methods.md`](../notes/layer1_pc_methods.md).

NASA publishes the CARA Analysis Tools, including the Pc SDK and the Alfano
(2009) test cases with their expected values. Running `Pc2D_Foster.m` under GNU
Octave on those cases reproduces the published expectations to 1 × 10⁻¹², which
establishes that the environment is genuinely CARA and not a re-implementation.
Against that reference:

| Our method | Worst relative error | Median |
|---|---|---|
| Eq. N-2, 1-D erf | 8.2 × 10⁻⁷ | 6.7 × 10⁻⁸ |
| Eq. N-1, 2-D quadrature | 6.9 × 10⁻⁶ | 4.1 × 10⁻⁷ |
| 6-D Monte Carlo, 2M samples | worst \|z\| = 3.10 | within 4σ |
| Small-HBR closed form | 1.2 × 10¹ | 1.00 |

CARA's own unit test allows 10⁻³. Eq. N-2 is about a thousand times tighter than
that, and costs one quadrature.

The closed form fails structurally on these cases rather than merely being
imprecise. At zero miss distance its limiting value `R² / (2 √det C)` already
exceeds one for several of the published cases, which is impossible for a
probability. Case 4 gives 1.4 × 10⁻²⁵ where CARA gives 4.9 × 10⁻², because the
disk centre sits about ten sigma out along the minor axis of a highly elongated
covariance while the disk still overlaps the long ridge of that distribution. The
closed form assumes a locally flat probability density, and this is exactly the
geometry where that assumption has nothing to stand on.

There is a real reason the closed form survives in operational software: in
routine LEO screening the hard-body radius is much smaller than the position
uncertainty (median disk 5.55 m against a median major axis of about 2 km in this
corpus, and no CDM in it exceeds Appendix N's validity limit). Under those
conditions the error is a fraction of a percent. Both statements are true at
once, and they describe different regimes. Generalising either one is a mistake.

---

## 4. Layer 2: what happens when you score everything

The competition scores 2,167 test events once, at 10⁻⁶. Scoring every CDM in the
raw release against its own event's final risk is a bigger and less flattering
experiment: 183,761 non-final CDMs, six thresholds, false positives and false
negatives counted separately.

| Threshold | positives | false-positive rate | false-negative rate |
|---|---|---|---|
| 10⁻⁷ | 8,691 | 10.8% | 12.4% |
| 3 × 10⁻⁷ | 6,469 | 8.3% | 15.5% |
| 10⁻⁶ | 4,457 | 5.7% | 18.7% |
| 10⁻⁵ | 1,448 | 2.0% | 28.2% |
| 10⁻⁴ | 375 | 0.47% | 52.8% |
| 10⁻³ | 79 | 0.035% | 60.8% |

Moving the line from 10⁻³ to 10⁻⁷ divides false alarms by 300 and cuts missed
events by a factor of five. There is no setting that is good on both axes, and
that trade-off is invisible to any single-threshold evaluation.

### Calibration

Ranking and calibration are different questions, and the forecasts in this
dataset only answer the first one well. The reliability curves in
[`results/calibration_reliability.png`](../results/calibration_reliability.png)
show observed frequency flat at about 10⁻³ while predicted Pc runs from 10⁻²² to
10⁻¹¹. Sixteen decades of predicted probability carry essentially no
information. The calibration slope of LRP is 0.2 to 0.8 where an honest forecast
would give 1.0, and a constant predictor beats it on log loss at every threshold
with enough positives to measure.

Fix the scale, though, and the picture changes. A two-parameter Platt map fitted
on the training split, applied unchanged to the test split, cuts log loss by
6.6× at 3 × 10⁻⁷, with the PR-AUC unmoved (0.6674 to 0.6651). The ranking was
always fine. The probability attached to it was not.

The Brier score is useless here, which is worth saying because it is the first
score most people reach for. It comes out identical to four decimal places for
the real forecast and the constant one, because Pc is tiny compared with the base
rate.

### Attribution: where the errors come from

Four candidate explanations were tested, and one of them was rejected.

**CDM age dominates.** At 3 × 10⁻⁷, the false-positive rate rises from 0.50% for
CDMs issued less than half a day before closest approach to 15.1% for CDMs issued
five to seven days out. False negatives move much less, from 3.4% to 18.3%. The
early messages are not wrong about the geometry; they are uncertain about it, and
an uncertain covariance produces a wide, flat probability distribution that
happens to straddle the threshold.

**Hard-body radius is a large systematic.** Swapping the catalogue value for a
flat 5 m moves the false-negative rate by about 35% at every threshold while
leaving the ranking untouched. Catalogue values run from 1.05 m to 18.35 m with a
median of 5.55 m, so the flat default is wrong for most objects. Perturbing the
radius for radar-cross-section uncertainty instead moves things by 3 to 5%, which
is a different order of problem.

**Method error is the smallest of the four.** On the subset where our covariance
reproduces ESA's reported risk to within 0.25 dex, switching from the closed form
to the exact integral changes the false-positive count by single digits out of
thousands. The median disagreement is 10⁻⁴ dex, though 11.4% of CDMs are off by
more than 0.1 dex and 5.9% by more than 1 dex. Small in aggregate, not small
case by case.

**Geometry was a hypothesis that did not survive.** The plausible story was that
slow encounters (relative speed below a few hundred m/s) would be the ones where
the flat-density assumption breaks, producing garbage Pc and therefore false
alarms. Across a 34-fold range of relative speed the false-positive rate moved
from 8.0% to 12.2% with no trend. Only 1.4% of CDMs in this corpus fall below
500 m/s, so the test has little power, and CARA documents 2-D Pc errors up to
330× for conjunctions at a few m/s. The honest statement is that relative speed
is not an aggregate driver *in this corpus*, not that it never matters.

### The covariance reconstruction

The raw release carries the complete 6×6 RTN covariance element by element, as
fifteen unique terms. Assembling them reproduces the published miss vector to
4.4 × 10⁻⁵ relative and ESA's own risk formula to −0.009 dex median on near
conjunctions. That also settles a trap: the dataset's `mahalanobis_distance`
column is not the χ² in ESA's formula. It is a different quantity, about 120
times smaller than the three-dimensional relative-position χ², and using it in
place of the right one produces numbers that look plausible and are wrong.

One residual remains open. For far misses (χ² ≥ 10) ESA's reported risk sits
well below the closed-form value, with a median error of 1.34 dex against 0.021
dex for near conjunctions. The likely explanation is that the published
covariance block is not the block behind the `risk` column in those rows. This is
the highest-value loose end in the finished layers, and
[`docs/handoff.md`](handoff.md) says so.

### Interpretability

CARA's handbook asks for physical reasoning as well as a number, so the four Pc
methods were scored on five dimensions: decomposability, monotonicity, validity,
traceability and auditability. Eq. N-2 wins at 0.844, the closed form is last at
0.716 despite being the most transparent of the four, and the Monte Carlo sits in
between. The scorecard's real message is a trade-off: the method that is easiest
to explain is the one that gives wrong answers.

One methodological warning is filed with it. Monotonicity had to be measured
with fitted partial sensitivities, not marginal correlations. The marginal
Spearman correlation of `log(R/σ)` against Pc is −0.65, the wrong sign, because
radius and χ² covary across CDMs. Simplifying that test in the obvious way
inverts the result.

---

## 5. What this work does not establish

- The forecasting model of layer 3 does not exist. Nothing here predicts an
  event better than the published baseline.
- The live replay of layer 4 does not exist. SOCRATES publishes TLEs without
  covariance, so the metrics built here do not transfer to it unchanged.
- Only one dataset is examined, and it is anonymised. Cross-dataset claims would
  need a second corpus.
- Vallado's textbook is cited by chapter for standard results, but no copy was
  available during the work and those citations are unverified. This is called
  out in the handoff document rather than buried.
- The figures are statistical. A study of 15,321 conjunctions says something
  about how alarms behave, not about how often objects collide.

---

## 6. If you only remember one thing

The interesting quantity in conjunction assessment is not the collision
probability. It is the distance between the collision probability and the
threshold somebody chose, measured in units of how uncertain that probability is.
Everything in the layers above is that idea applied consistently: sweep the
threshold instead of picking one, validate the Pc method against somebody else's
code instead of trusting the formula, and attribute every false alarm to a cause
you can name.
