#!/usr/bin/env bash
# One-command environment restore after a reset.
#
# Two things are lost between turns, and neither is a workspace problem:
#   1. the 221 MB dataset      -> re-fetched from Zenodo (public, DOI 10.5281/zenodo.4463683)
#   2. installed pip packages  -> reinstalled from requirements.txt
#
# The project itself (code, notes, results/) is ~1.2 MB / 42 files and is well
# inside the 128 MB / 10,000-file snapshot cap, so it persists untouched.
#
# Usage:  bash scripts/restore.sh            (full restore)
#         bash scripts/restore.sh --check    (verify only, download nothing)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

echo "=============================================================="
echo "Kelvins pipeline -- environment restore"
echo "=============================================================="

# ---------------------------------------------------------------- packages
echo
echo "[1/2] Python packages"
if python3 -c "import numpy, pandas, scipy, sklearn, matplotlib, sgp4" 2>/dev/null; then
    echo "  all imports OK -- nothing to install"
else
    missing=$(python3 - <<'PY'
import importlib.util
mods = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "sklearn": "scikit-learn", "matplotlib": "matplotlib", "sgp4": "sgp4"}
missing = [pkg for mod, pkg in mods.items()
           if importlib.util.find_spec(mod) is None]
print(" ".join(missing))
PY
)
    if [ -n "${missing// /}" ]; then
        echo "  installing: $missing"
        pip install --quiet $missing
    else
        echo "  all present"
    fi
fi
python3 -c "import numpy, pandas, scipy, sklearn, matplotlib, sgp4; print('  imports verified')"

# ---------------------------------------------------------------- dataset
echo
echo "[2/2] Dataset"
CHECK_ONLY=0
case "${1:-}" in --check) CHECK_ONLY=1 ;; esac

if python3 scripts/fetch_data.py --check >/dev/null 2>&1; then
    echo "  dataset present"
else
    echo "  fetching from Zenodo (221 MB, ~4 min)..."
    python3 scripts/fetch_data.py
fi

# ---------------------------------------------------------------- verify
echo
echo "=============================================================="
echo "Verification"
echo "=============================================================="
status=0
for s in reproduce_lrp validate_pipeline validate_foster_mc threshold_sweep \
         calibration threshold_fpfn_199k attribution_199k \
         attribution_method interpretability_scorecard; do
    printf "  %-24s " "$s"
    if python3 "scripts/$s.py" >/dev/null 2>&1; then echo "PASS"; else echo "FAIL"; status=1; fi
done
printf "  %-24s " "test_orbital"
if python3 tests/test_orbital.py >/dev/null 2>&1; then echo "PASS"; else echo "FAIL"; status=1; fi

# Layer 1: ours vs CARA's own Pc2D_Foster on the published Alfano cases.
# This needs the CARA clone and GNU Octave, neither of which is a Python
# dependency, so establish them here. They have been wiped more than once, and a
# silently SKIPped check is how a result rots without anyone noticing.
echo
echo "  Establishing the CARA validation environment"

if [ "$CHECK_ONLY" = "1" ]; then
    if [ ! -f /home/user/CARA_Analysis_Tools/DistributedMatlab/ProbabilityOfCollision/Pc2D_Foster.m ]; then
        echo "    SKIP (CARA repo not present)"
    elif ! command -v octave >/dev/null 2>&1; then
        echo "    SKIP (octave not installed)"
    fi
else
    if [ ! -f /home/user/CARA_Analysis_Tools/DistributedMatlab/ProbabilityOfCollision/Pc2D_Foster.m ]; then
        echo "    cloning nasa/CARA_Analysis_Tools ..."
        rm -rf /home/user/CARA_Analysis_Tools
        git clone --depth 1 --filter=blob:none --no-checkout \
            https://github.com/nasa/CARA_Analysis_Tools.git \
            /home/user/CARA_Analysis_Tools 2>&1 | tail -1
        ( cd /home/user/CARA_Analysis_Tools && \
          git sparse-checkout set DistributedMatlab/ProbabilityOfCollision \
              DistributedMatlab/Utils DataFiles 2>&1 | tail -1 && \
          git checkout 2>&1 | tail -1 )
    else
        echo "    CARA repo already present"
    fi
    if ! command -v octave >/dev/null 2>&1; then
        echo "    installing GNU Octave ..."
        sudo apt-get update -qq >/dev/null 2>&1
        sudo apt-get install -y --no-install-recommends octave octave-io \
            >/dev/null 2>&1 || echo "    WARNING: octave install failed"
    else
        echo "    octave already installed"
    fi
fi

printf "  %-24s " "validate_against_cara"
if [ ! -f /home/user/CARA_Analysis_Tools/DistributedMatlab/ProbabilityOfCollision/Pc2D_Foster.m ]; then
    echo "SKIP (CARA repo not present)"
elif ! command -v octave >/dev/null 2>&1; then
    echo "SKIP (octave not installed)"
elif python3 scripts/validate_against_cara.py >/dev/null 2>&1; then
    echo "PASS"
else
    echo "FAIL"; status=1
fi

echo
if [ "$status" -eq 0 ]; then
    echo "RESTORE COMPLETE -- all checks pass"
else
    echo "RESTORE INCOMPLETE -- see failures above"
fi
exit "$status"
