#!/usr/bin/env bash
# Download the Indian Pines, Pavia University and Salinas scenes (Grupo de Inteligencia Computacional, UPV/EHU)
# into data/ (or the folder given as first argument). A GitHub mirror is used when the original server cannot be
# reached. load_scene() checks the size of each cube and the number of labelled pixels; the SHA-256 sums of the
# files used for the reported results are compared here as well.
set -euo pipefail
out=${1:-data}; here=$(cd "$(dirname "$0")" && pwd); mkdir -p "$out"
base=https://www.ehu.eus/ccwintco/uploads
mirror=https://raw.githubusercontent.com/gokriznastic/HybridSN/master/data
get() {  # file name, path on the original server
  if [ -s "$out/$1" ]; then echo "ok      $1 (already downloaded)"; return 0; fi
  if curl -fsSL --retry 3 -o "$out/$1.part" "$base/$2" 2>/dev/null; then src="original server"
  elif curl -fsSL --retry 3 -o "$out/$1.part" "$mirror/$1"; then src="mirror"
  else echo "FAILED  $1"; return 1; fi
  mv "$out/$1.part" "$out/$1"; echo "ok      $1 ($src)"
}
get Indian_pines_corrected.mat 6/67/Indian_pines_corrected.mat
get Indian_pines_gt.mat c/c4/Indian_pines_gt.mat
get PaviaU.mat e/ee/PaviaU.mat
get PaviaU_gt.mat 5/50/PaviaU_gt.mat
get Salinas_corrected.mat a/a3/Salinas_corrected.mat
get Salinas_gt.mat f/fa/Salinas_gt.mat
(cd "$out" && sha256sum -c --quiet "$here/SHA256SUMS.txt") || echo "note: files differ from those used for the reported results"
