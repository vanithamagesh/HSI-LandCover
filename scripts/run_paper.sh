#!/usr/bin/env bash
# The experiments behind every number in the paper (FCN, SVM and MLP, three seeds each), then all
# figures and tables. It is run_all.sh without the AnySensor runs: 42 settings.
#   bash scripts/run_paper.sh [DATA_DIR] [RUNS_DIR] [PAPER_DIR]
# Finished seeds are skipped, so the script can be restarted after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
data=${1:-data}; runs=${2:-runs}; paper=${3:-paper}
run() { python scripts/train_eval.py --data "$data" --out "$runs" "$@"; }
for sc in IP PU SA; do
  for sp in full s2 landsat8 rgbn rgb; do
    run --scene "$sc" --spectral "$sp" --split random
    run --scene "$sc" --spectral "$sp" --split block
  done
  run --scene "$sc" --model mlp --no-svm
  for n in 10 25 100; do run --scene "$sc" --per-class "$n"; done
done
python scripts/paper_figures.py --data "$data" --runs "$runs" --out "$paper"
