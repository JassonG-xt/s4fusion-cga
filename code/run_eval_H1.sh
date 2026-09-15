#!/usr/bin/env bash
# Step 1 H1 evaluation: run AFTER S finishes (do not run concurrently with
# training on the 4 GB GPU). Evaluates B0 + S on M3FD test under identical tiling,
# then runs the paired-significance H1 analysis.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p results/abl
MAN=../dataset/manifests/m3fd_test.csv
TILE="--tile-size 507 --tile-overlap 64"

[ -f checkpoints/abl_S_s42.pt ] || { echo "ERROR: checkpoints/abl_S_s42.pt missing — S not finished"; exit 1; }

echo "[eval] B0 baseline (boundary none)"
$PY evaluate_brss.py --data-root ../dataset --manifest "$MAN" \
  --checkpoint checkpoints/abl_B0_s42.pt \
  --boundary-mode none --fusion-scales none --gate-type scalar \
  --output-dir results/abl/B0_m3fd --metrics-csv results/abl/B0_m3fd.csv \
  $TILE --amp --no-save-rgb

echo "[eval] S state (boundary state)"
$PY evaluate_brss.py --data-root ../dataset --manifest "$MAN" \
  --checkpoint checkpoints/abl_S_s42.pt \
  --boundary-mode state --fusion-scales none --gate-type scalar \
  --output-dir results/abl/S_m3fd --metrics-csv results/abl/S_m3fd.csv \
  $TILE --amp --no-save-rgb

echo "[analyze] H1 gate"
$PY analyze_h1.py --b0-csv results/abl/B0_m3fd.csv --s-csv results/abl/S_m3fd.csv \
  --b0-ckpt checkpoints/abl_B0_s42.pt --s-ckpt checkpoints/abl_S_s42.pt
