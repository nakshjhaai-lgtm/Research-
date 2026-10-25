# Licences, data and third-party material

Four different things are in this repository and they do not all share the same
terms. None of this is legal advice; it is a description of what is here and
where it came from, so you can make your own decision.

---

## 1. The dataset

ESA Kelvins *Spacecraft Collision Avoidance Challenge*, DOI
[10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683), released under
CC-BY. It is not committed here. `scripts/fetch_data.py` downloads it from Zenodo,
which is the recommended way to get it anyway: the archive is 221 MB and Zenodo
is a stable, citable mirror.

If you publish anything based on it, cite the dataset and Uriot et al. (2021),
listed in `CITATION.cff`.

---

## 2. NASA CARA Analysis Tools

`third_party/cara_analysis_tools/` is an unmodified copy of the top level of
[nasa/CARA_Analysis_Tools](https://github.com/nasa/CARA_Analysis_Tools), taken
during the project. It contains four NASA Open Source Agreement documents
(GSC-18593-1, GSC-18848-1, GSC-19027-1, GSC-19374-1) and NASA's own README, both
of which travel with it.

That material stays under its own terms. In particular:

- It is used here as an **external reference implementation**, to check our
  numbers against NASA's. No code in `src/` or `scripts/` is copied from it, and
  the agreement between the two is a numerical result rather than a shared
  implementation.
- Only the files listed above are included, not the toolkit. The MATLAB source,
  the test data and the data files that `run_cara.m` consumes are fetched from
  GitHub by `scripts/restore.sh`, on the assumption that anyone who wants to run
  CARA can clone it from the source. That is why the validation script does not
  work out of the box on a machine with no network.

If you re-distribute this repository, keep the NOSA PDFs with it, or drop
`third_party/` entirely. Nothing in the analysis depends on those files being
present, only on a clone of the toolkit existing at the path `run_cara.m` expects.

---

## 3. The research code and the write-up

No licence has been attached to the author's own work in `src/`, `scripts/`,
`notes/`, `docs/` and `assets/`. That is a deliberate omission rather than an
oversight: choosing a licence is the author's call, and inventing one on their
behalf would be worse than leaving the question visible.

Until a licence is added, default copyright applies, which means people can read
and cite the work but should ask before reusing it. If you are the author and want
to settle it, adding a `LICENSE` file at the root is the whole job. MIT or
Apache-2.0 are the usual choices for research code; CC-BY-4.0 is often used for
the prose in `docs/`.

---

## 4. Quoted material

The notes and documentation quote short passages from published sources (ESA
conference papers, JAXA and NASA policy documents, the Kelvins competition paper)
for identification and commentary. Those quotations remain the property of their
authors. Full references are in `docs/technical-report.md` and in the source list
at the end of `docs/handoff.md`.

---

## 5. Figures and images

Everything in `assets/` was drawn for this repository by
`tools/make_figures.py`, using nothing but the CSVs in `results/` and matplotlib.
There is no stock imagery, no traced diagram and no third-party artwork. The
fonts are whatever matplotlib ships with (DejaVu Sans by default).

That means the figures can be regenerated, corrected and re-licensed freely along
with the rest of the author's work.
