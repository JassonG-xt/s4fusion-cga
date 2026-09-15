#!/usr/bin/env bash
# F3 cross-scale drift: B0 vs S on M3FD (native, untiled), then verdict.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p results/abl
MAN=../dataset/manifests/m3fd_test.csv
LIM=${1:-120}
echo "[f3] B0 baseline (limit=$LIM)"
$PY run_f3.py --manifest "$MAN" --checkpoint checkpoints/abl_B0_s42.pt \
  --boundary-mode none  --out-csv results/abl/B0_f3.csv --limit "$LIM"
echo "[f3] S state"
$PY run_f3.py --manifest "$MAN" --checkpoint checkpoints/abl_S_s42.pt \
  --boundary-mode state --out-csv results/abl/S_f3.csv --limit "$LIM"
echo "[f3] analyze"
$PY analyze_f3.py --b0-csv results/abl/B0_f3.csv --s-csv results/abl/S_f3.csv
