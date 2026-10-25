# Roadmap — VERBATIM SPEC (stored as received, unaltered)

> Reproduced exactly as given, including the copy-paste artefacts in the source
> text (the `&amp;amp;amp;` entity mangling and the nested gpsjam.org markdown
> link). Nothing here has been corrected, reworded or tidied.
> Annotated, corrected and status-tagged version: `notes/ROADMAP.md`.

- ┌────────────────────────────────────────────────────────────┐

│ LAYER 0 — DEFINITIONS (the "you can't optimise an          │

│ undefined quantity" module)                                │

│ • Configurable FP/FN definitions per threshold:            │

│   10⁻³ (JAXA) · 10⁻⁴ (ESA/DLR/18SPCS) · 10⁻⁵ ·           │

│   3×10⁻⁷ (SpaceX autonomous)                               │

│ • Every result reported as a THRESHOLD SWEEP, never one    │

│ number. Cross-provider disagreement as a first-class      │

│ metric.                                                   │

├────────────────────────────────────────────────────────────┤

│ LAYER 1 — Pc METHODS (reference implementations, cited)    │

│ • Foster/Estes 2D (Foster &amp;amp;amp; Estes 1992)                     │

│ • Monte Carlo ground-truth estimator (CARA-style;           │

│ Appendix N methodology)                                   │

│ • HBR variants: flat 5 m (Space-Track EVR behaviour),       │

│ catalogue HBR, RCS-uncertainty HBR                        │

│ • (Optional, credit) Chan/3D variants                        │

├────────────────────────────────────────────────────────────┤

│ LAYER 2 — THE EVALUATION CORE  ← your actual contribution  │

│ • Calibration / reliability curves (does Pc mean anything?) │

│ • F₂ + precision-recall at every threshold (Kelvins-        │

│ standard metrics; beat or contextualise LRP F₂=0.739)     │

│ • Attribution: how much FP comes from HBR choice vs         │

│ covariance staleness (CDM age) vs method error vs         │

│ geometry (low-relative-velocity encounters)               │

│ • Interpretability scorecard (per CARA's "physical          │

│ reasoning" requirement): does the method explain WHY?      │

├────────────────────────────────────────────────────────────┤

│ LAYER 3 — YOUR TRIAGE METHOD (the candidate that must       │

│ pass your own harness)                                      │

│ • Interpretable features: Pc persistence/decay across       │

│ CDM sequence, volatility near threshold, time-to-TCA,     │

│ CDM age, HBR provenance, geometry flags                   │

│ • Calibrated, transparent model (gradient-boosted trees or  │

│ logistic — monotonic constraints), explicit F₂ reporting   │

│ • Baselines you must beat/contextualise: LRP (0.739),       │

│ Kessler-style sequence model, plain Foster thresholding    │

├────────────────────────────────────────────────────────────┤

│ LAYER 4 — LIVE REPLAY (what keeps it alive 2026→2028)       │

│ • Weekly SOCRATES ingest; rolling "state of alert            │

│ trustworthiness" report (like [[gpsjam.org](http://gpsjam.org)]([http://gpsjam.org](http://gpsjam.org)) but for alarm     │

│ quality). New fragmentation events (e.g. USA-32,          │

│ Sep 2026) automatically enter the stream.                  │

├────────────────────────────────────────────────────────────┤

│ LAYER 5 — REPORTS (the artifact people read)                │

│ • One command → markdown/PDF report with tables,            │

│ calibration plots, threshold sweeps, honest limitations    │

└──────────────  └─────────────────────────────────────────────┘
