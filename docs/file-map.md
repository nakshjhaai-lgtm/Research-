# Where everything went

This repository is the archive of a working session, reorganised for reading. It
seemed more useful to say exactly what moved than to pretend the layout was
always like this.

Three rules were followed.

1. **No file content was changed.** Not one line in `src/`, `scripts/`,
   `tests/`, `notes/` or `results/`. If you diff this repository against the
   original upload, every one of those files is identical byte for byte.
2. **Only paths and working titles changed.** `kelvins_pc/` was a project folder
   inside a larger workspace. At repository root it would be a pointless extra
   level, so its contents were lifted up. Two documents whose file names
   were working labels got names describing their contents.
3. **Nothing was discarded except one empty marker file** that the sandbox wrote
   into the home directory, described at the bottom.

---

## The mapping

| Original path | Path here | Note |
|---|---|---|
| `kelvins_pc/src/*` | `src/*` | unchanged |
| `kelvins_pc/scripts/*` | `scripts/*` | unchanged |
| `kelvins_pc/tests/*` | `tests/*` | unchanged |
| `kelvins_pc/notes/*` | `notes/*` | unchanged |
| `kelvins_pc/results/*` | `results/*` | unchanged |
| `kelvins_pc/requirements.txt` | `requirements.txt` | unchanged |
| `kelvins_pc/scripts_helpers.py` | `scripts_helpers.py` | unchanged |
| `kelvins_pc/README.md` | `docs/technical-report.md` | renamed: it is the full technical write-up, not a front page |
| `kelvins_pc/HANDOFF.md` | `docs/handoff.md` | renamed: lowercase, and it now sits with the other reading material |
| `CARA_Analysis_Tools/` | `third_party/cara_analysis_tools/` | renamed to make the authorship obvious; contents and NOSA terms untouched |
| `.sudo_as_admin_successful` | not carried over | zero-byte sandbox artefact |

New in this repository, written for the upload: `README.md`, everything in
`docs/` except the two renamed files above, `assets/`, `tools/make_figures.py`,
`.gitignore` and `CITATION.cff`.

---

## References that still mention the old layout

Because the files were not edited, a few of them refer to paths that no longer
exist. They are all harmless, and listed here so nobody hunts for them:

- `docs/handoff.md` and `notes/storage_and_persistence.md` refer to
  `/home/user/kelvins_pc/` and to file counts from the working session. Read
  `kelvins_pc/` as "the repository root".
- `scripts/restore.sh`, `scripts/attribution_method.py`,
  `scripts/interpretability_scorecard.py`, `scripts/extract_alfano_cases.py` and
  `src/kelvins_data.py` contain absolute paths under `/home/user`. The dataset
  path can be overridden with `KELVINS_DATA_ROOT`; the CARA path expects a clone
  at `/home/user/CARA_Analysis_Tools`, which is what `restore.sh` creates.
  [`reproduction.md`](reproduction.md) covers both.
- `docs/technical-report.md` refers to a `kelvins_pc/` directory tree and a
  "1.2 MB / 42 files" budget from the original environment. The tree it draws is
  still accurate; only the outer folder name changed.
- `notes/ROADMAP_VERBATIM.md` contains the original task text exactly as it was
  received, including its copy-paste artefacts. That is the point of the file: it
  is the record of the brief, not a tidied version of it.

---

## The upload itself

The original archive is `workspace-01a0e1f0-067c-759f-9a89-9e953ce5d31c.zip`,
1.5 MB, which unpacked to 65 files across two top-level folders. It was uploaded
to the `main` branch of this repository at commit `27dbd3d` ("Add files via
upload"). The zip has been removed from the working tree because a compressed
copy of a tree that is already committed serves no purpose; it remains in the
history and can be recovered with `git show 27dbd3d:workspace-01a0e1f0-067c-759f-9a89-9e953ce5d31c.zip > original.zip`.

If you would rather keep the zip at the repository root as the archival record,
say so and it can go back. Nothing else about it changes.
