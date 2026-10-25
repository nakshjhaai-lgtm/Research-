# Glossary

Plain definitions for the terms that carry the argument. Where a symbol is
standard, both the symbol and the word are given.

**CDM, Conjunction Data Message.** A standardised message about one pair of
objects that will pass close to each other. It carries positions, velocities,
covariances, hard-body radii and a probability of collision. The Kelvins dataset
is 199,082 of them.

**Conjunction.** A close approach between two tracked objects, close enough to be
screened. Not a collision.

**TCA, time of closest approach.** The instant the two objects are nearest each
other. Everything in a CDM is either measured at TCA or measured relative to it.

**Pc, probability of collision.** Probability that the two objects occupy the
same space at TCA, given the stated covariances. Reported here as log₁₀, so
Pc = −6 means one in a million. Values in this dataset run from −30 (a floor,
meaning "negligible") to about −1.4.

**High risk.** Whatever the operator says it is. In this study, `log10 Pc ≥ −6`
for the competition, and four other thresholds drawn from published policy.

**Reaction (or manoeuvre) threshold.** The Pc above which an operator will
consider spending fuel. A policy number, and the thing layer 0 holds constant
while everything else varies.

**HBR, hard-body radius.** The combined size of the two objects, treated as a
disk in the encounter plane. It is the radius the collision probability is
integrated over, so it matters linearly. Catalogue values in this corpus run
1.05 m to 18.35 m, median 5.55 m. Space-Track's default of 5 m is a guess.

**RTN frame.** Radial, along-track, cross-track. The natural frame for a
covariance in orbit: the along-track direction is the one you know least well,
usually by a factor of ten or more. All CDM covariances arrive in RTN.

**Encounter plane (conjunction plane).** The plane perpendicular to the relative
velocity at TCA, containing the relative position. The 2-D Pc methods project
both covariances into it, which turns a 3-D problem into an ellipse overlapping a
disk.

**Combined covariance.** The two objects' covariances added together, after both
have been projected into the encounter plane. The relative-position uncertainty
is the sum, since the errors are treated as independent.

**Miss distance.** Distance between the two predicted positions at TCA. Compare
it with the HBR: if it is smaller than the HBR, the disks overlap.

**χ² (in this context).** The squared Mahalanobis distance in the encounter
plane, `rᵀ C⁻¹ r`. It says how many standard deviations the miss distance is. Note
that the dataset also ships a column called `mahalanobis_distance`, and it is not
this quantity.

**σ_x, σ_y.** Major and minor semi-axes of the combined covariance ellipse in the
encounter plane. `R_lim = 4σ_x − min(R)` is Appendix N's rough validity limit for
the small-HBR approximation.

**Eq. N-1.** The exact 2-D integral of a Gaussian density over a disk, from
Foster and Estes (1992), evaluated here by quadrature.

**Eq. N-2.** A 1-D reduction of the same integral to a difference of error
functions, from Alfano (2005a) and refined by Elrod (2019). Fast and, as far as
the CARA test cases can tell, exact to about 10⁻⁷ relative.

**Small-HBR closed form.** The Chan (2008) formula, `R² / (2 √det C) · exp(−χ²/2)`,
which assumes the probability density is flat across the disk. Accurate when R is
much smaller than the uncertainties, catastrophic when it is not.

**Monte Carlo Pc.** Draw both states from their covariances, propagate to TCA,
count how many draws collide. The ground truth the other methods are checked
against, expensive and noisy at small Pc.

**LRP, Latest Risk Prediction.** The challenge's published baseline. Use the risk
value from the newest CDM still visible to the competitor, that is, the last one
sent at least two days before TCA, and clip low values down to −6.001 so they sit
below the classification line.

**CRP.** Essentially a constant predictor; in the challenge it means predicting a
fixed low risk. Its published loss is 2.5, and it beats LRP on log loss, which is
one of the more uncomfortable findings here.

**F₂.** F-score with β = 2, `(1+β²)·p·r / (β²·p + r)`. Recall is weighted twice
as heavily as precision because a missed conjunction is worse than a wasted
manoeuvre. This is the competition's primary score.

**MSE_HR.** Mean squared error of the prediction, computed only over high-risk
events. Note the oddity: the score rewards getting the hard cases right and does
not care about the low-risk ones.

**L, the challenge loss.** `MSE_HR / F₂`. Reported for comparison with the
published 0.694.

**FP/FN rate.** False positives divided by the number of true negatives, and
false negatives divided by the number of true positives. Deliberately reported
separately, because F₂ hides the trade-off between them.

**Calibration.** Whether a predicted probability of 0.3 happens about 30% of the
time. Rank-based scores such as PR-AUC say nothing about it, which is why a
forecast can rank well and be badly calibrated. The slope test here fits the
outcome against the prediction and asks whether the slope is 1.

**Platt scaling.** Fitting `σ(a·x + b)` to a score so the output behaves like a
probability. Two parameters, fitted on the training split only, and enough to cut
log loss by 6.6× here without changing the ordering.

**Reliability diagram.** Predicted probability against observed frequency, per
bin. A flat line across many decades of prediction means the predictions carry no
information in that range, which is exactly what happens here below 10⁻⁶.

**ECE, expected calibration error.** A single number summarising a reliability
diagram. Misleading for constant predictors, where only one bin is populated.

**Brier score.** Mean squared error of a probability forecast. Useless in this
problem, because Pc is tiny compared with the base rate, so every reasonable
forecast scores the same to four decimal places.

**Lead time / CDM age.** How long before TCA a CDM was issued. The dominant
driver of false alarms in this corpus.

**Cross-provider disagreement.** The share of events that at least two of the
five operator policies classify differently. 7.52% on the truth and 10.84% on
predictions: a floor no model can remove, because it comes from the definitions.

**SGP4.** The standard propagator for TLEs, from Hoots and Roehrich (1980). Takes
mean elements plus a drag term and predicts the state at a requested time.

**TLE.** Two-line element set, the 80-column format used for unclassified
catalogues. Implied decimal points and implied exponent signs make hand-parsing
it a reliable source of bugs.

**KVL.** Key–value language, the text format of a standard CDM. `src/cdm_parser.py`
reads it; the Kelvins release instead ships flattened tables, which the same
module adapts.

**SOCRATES.** CelesTrak's public conjunction screening service, three runs a day
over the whole unclassified catalogue. TLEs only, so no covariance, which is why
layer 4 cannot reuse the metrics built here as they stand.

**CARA.** NASA's Conjunction Assessment Risk Analysis group. Publishes the
analysis toolkit used as the reference implementation, and the handbook plus
Appendix N that define the Pc methodology.
