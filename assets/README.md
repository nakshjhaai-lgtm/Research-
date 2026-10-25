# Figures

Everything in this directory is generated. `python tools/make_figures.py`
redraws all of it from the CSVs in `results/`, so no picture here can disagree
with the data it illustrates.

| File | Shows | Source |
|---|---|---|
| `figures/banner.png` | title card | drawn, no data |
| `figures/lrp-gate.png` | the reproduced baseline against the published one, and four selectors that do worse | `results/lrp_metrics.csv`, `docs/technical-report.md` |
| `figures/threshold-tradeoff.gif` | false alarms against missed events as the threshold slides from 10⁻³ to 10⁻⁷, 18 frames | `results/fpfn_thresholds_199k.csv` |
| `figures/lead-time.gif` | the same FP/FN split by how early the CDM was issued, at four policies, 4 frames | `results/fpfn_by_leadtime_199k.csv` |
| `figures/pc-method-error.png` | our Pc against NASA CARA's on the published Alfano cases, and the relative error of each method | `results/cara_alfano_validation.csv` |
| `figures/provider-disagreement.png` | how many of the same 2,167 events each policy calls high risk | `results/threshold_definition_sensitivity.csv` |
| `diagrams/pipeline.png` | the five stages and the check behind each | drawn |

The two GIFs animate a sweep rather than presenting a static result. They are
looped and unprompted on purpose: the point of the study is that a single frame
of this data is misleading, so the animation shows the whole range.

If you would rather have static images, every GIF frame is reproducible from the
same script in a couple of lines by saving the figure instead of appending it to
the frame list.
