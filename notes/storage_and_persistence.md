# Storage and persistence — why cloud storage is the wrong fix

Written after the dataset vanished from the workspace mid-project, which cost a
re-download and revealed a second, quieter problem.

---

## What actually happened

Two separate losses, from two different causes:

| Lost | Cause | Recovered by |
|---|---|---|
| `/home/user/data` (221 MB dataset) | exceeded the workspace snapshot size cap | `scripts/fetch_data.py` (Zenodo) |
| the `sgp4` pip package | installed packages are not snapshotted at all | `requirements.txt` |

The second one is the important one. It was not a one-off: `sgp4` was installed,
the test suite passed, and the package was gone again in a later turn. **Every
turn starts from the base image plus whatever is under `/home/user`.** That is a
recurring cost, not a single incident, and it is why `scripts/restore.sh` exists.

---

## Is cloud storage the answer? No — and here is the arithmetic

### The project is not the problem

```
kelvins_pc/          1.2 MB    42 files
  results/           876 KB    <- every published table and plot
  src/               100 KB
  scripts/            92 KB
  notes/              64 KB
  tests/             8.0 KB
```

The snapshot cap is **128 MB / 10,000 files**. The project uses **1.2 MB / 42
files** — about 1% of the allowance. Code, notes and results persist normally
and need no backup mechanism. The only thing that was dropped was the dataset.

### The dataset is already in the cloud, and better than anything I could upload it to

| | Zenodo (current) | anonymous host |
|---|---|---|
| Size | 221 MB | 512 MB cap (0x0.st) |
| Retention | permanent, DOI `10.5281/zenodo.4463683` | 14 days (transfer.sh), 30–365 d size-based (0x0.st), 60 d (pixeldrain) |
| Account | no | no |
| Provenance / citable | yes | no |
| Cost | free | free |
| Restore | one command, ~4 min | one command, ~4 min |

Uploading the dataset to an anonymous host would be **strictly worse on every
axis**: shorter retention, a hard size cap, no DOI, no citation, and it would
make a copy of a public dataset depend on a hobby service. The research is blunt
about the reliability of these hosts — 0x0.st is "operated as a hobby service,
which makes it a fine fit for informal sharing and **a bad fit for anything
production-critical**".

### The platform's own durable-storage primitive is not available here

E2B does have *Volumes* that "persist independently of the sandbox lifecycle"
and survive sandbox destruction — but the feature is **in private beta**, and
from inside the sandbox there is no API to provision one. Pause/resume and
snapshots preserve filesystem + memory, but only for a sandbox that is still
alive; they do not survive a teardown. So the platform answer exists in
principle and is not reachable from this environment.

---

## What was done instead

1. **`scripts/fetch_data.py`** — re-fetches the dataset from Zenodo, unpacks it,
   extracts the inner `train_data.zip`, and inserts the extra `kelvins/` path
   level the loaders expect. Verifies the byte count against the server's
   `content-length` (221,128,642).
2. **`requirements.txt`** — pins numpy 2.3.5, pandas 2.2.3, scipy 1.17.1,
   scikit-learn 1.6.1, matplotlib 3.10.9, sgp4 2.27 — the exact versions the
   published results were produced with.
3. **`scripts/restore.sh`** — one command: install packages, fetch data, then
   re-run all eight checks and report PASS/FAIL. Exits non-zero on failure.

Total restore cost: ~10 s for packages, ~4 min for the dataset, ~4 min for
verification. That is cheaper than maintaining a cloud mirror, and it keeps the
reproducibility story entirely inside the repo.

---

## If you still want a cloud backup

The results bundle is 876 KB and would fit on any host, so it is *technically*
trivial to add:

```bash
tar czf results.tar.gz results/ && curl -F 'file=@results.tar.gz' https://0x0.st
```

But: it makes the tables publicly downloadable at an unguessable URL with a
30–365 day lifetime, it adds a moving part to the reproducibility chain, and the
same bytes are already safe in the workspace snapshot. The one scenario where it
earns its keep is handing the results to someone who does not have workspace
access — and for that, a DOI'd Zenodo deposit of the results would be the right
answer, not an anonymous host.

Not implemented. If you want it, say so and it is a ten-line script.
