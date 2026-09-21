#!/usr/bin/env bash
# H1 control (the "analogous control on its reliability map" promised in §3.4):
# inference-only, no training. Waits for the main evaluation chain to release the
# GPU, then re-evaluates the seed-42 H1 arm with the reliability map forced to a
# content-free value (uniform) and to a spatially permuted copy (shuffle).
set -uo pipefail
cd /mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code
PY=.venv-brss/bin/python
MAN=../dataset/manifests/m3fd_test.csv
TILE="--tile-size 507 --tile-overlap 64"
mkdir -p results/abl logs

echo "[$(date '+%F %T')] waiting for the main H1 evaluation to finish ..."
while true; do
  if [ -f results/abl/S_m3fd_s3407.csv ] && ! pgrep -f "evaluate_brss.py" >/dev/null; then
    break
  fi
  sleep 60
done
sleep 20
echo "[$(date '+%F %T')] GPU free; running the H1 reliability-map control"

for MODE in uniform shuffle; do
  echo "[$(date '+%F %T')] H1 control: reliability-override=${MODE}"
  $PY evaluate_brss.py --data-root ../dataset --manifest "$MAN" \
    --checkpoint checkpoints/abl_S_s42.pt \
    --boundary-mode state --fusion-scales none --gate-type scalar \
    --reliability-override "${MODE}" \
    --output-dir "results/abl/S_${MODE}_m3fd" \
    --metrics-csv "results/abl/S_${MODE}_m3fd.csv" \
    $TILE --amp --no-save-rgb > "logs/H1_control_${MODE}.log" 2>&1
  echo "[$(date '+%F %T')] done ${MODE}"
done

echo "[$(date '+%F %T')] H1_CONTROL_ALL_DONE"
