#!/usr/bin/env bash
# Every experiment of the paper (three seeds each), then all figures and tables.
#   bash scripts/run_all.sh [DATA_DIR] [RUNS_DIR] [PAPER_DIR]
# Finished seeds are skipped, so the script can be restarted after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
data=${1:-data}; runs=${2:-runs}; paper=${3:-paper}
run() { python scripts/train_eval.py --data "$data" --out "$runs" "$@"; }
# 1. baselines: spectral resolution x evaluation protocol (FCN + SVM), spectral-only MLP, learning curve
for sc in IP PU SA; do
  for sp in full s2 rgb; do
    run --scene "$sc" --spectral "$sp" --split random
    run --scene "$sc" --spectral "$sp" --split block
  done
  run --scene "$sc" --model mlp --no-svm
  for n in 10 25 100; do run --scene "$sc" --per-class "$n"; done
done
# 2. AnySensor: one model for every sensor, against one FCN per sensor, and its ablations
for sc in IP PU SA; do
  for split in random block; do
    run --scene "$sc" --model anysensor --split "$split"
    for sp in landsat8 rgbn; do run --scene "$sc" --spectral "$sp" --split "$split"; done
  done
  run --scene "$sc" --model anysensor --no-rss
  run --scene "$sc" --model anysensor --no-fwhm
done
python scripts/paper_figures.py --data "$data" --runs "$runs" --out "$paper"
