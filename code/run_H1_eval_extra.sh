#!/usr/bin/env bash
# H1 gate evaluation for the extra seeds (123, 3407).
# Run ONLY after run_H1_seeds_extra.sh has finished (single 4 GB GPU).
# Same protocol as the seed-42 H1 evaluation: tiled 507/overlap 64, M3FD test
# manifest, paired per-image analysis by analyze_h1.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p results/abl logs
MAN=../dataset/manifests/m3fd_test.csv
TILE="--tile-size 507 --tile-overlap 64"

for S in 123 3407; do
  if [ ! -f "checkpoints/abl_S_s${S}.pt" ]; then
    echo "[skip] s${S}: checkpoints/abl_S_s${S}.pt missing (training not finished)"
    continue
  fi
  echo "[$(date '+%F %T')] eval s${S}: B0"
  $PY evaluate_brss.py --data-root ../dataset --manifest "$MAN" \
    --checkpoint checkpoints/abl_B0_s${S}.pt \
    --boundary-mode none --fusion-scales none --gate-type scalar \
    --output-dir results/abl/B0_m3fd_s${S} --metrics-csv results/abl/B0_m3fd_s${S}.csv \
    $TILE --amp --no-save-rgb

  echo "[$(date '+%F %T')] eval s${S}: S"
  $PY evaluate_brss.py --data-root ../dataset --manifest "$MAN" \
    --checkpoint checkpoints/abl_S_s${S}.pt \
    --boundary-mode state --fusion-scales none --gate-type scalar \
    --output-dir results/abl/S_m3fd_s${S} --metrics-csv results/abl/S_m3fd_s${S}.csv \
    $TILE --amp --no-save-rgb

  echo "[$(date '+%F %T')] analyze s${S}"
  $PY analyze_h1.py --b0-csv results/abl/B0_m3fd_s${S}.csv \
    --s-csv results/abl/S_m3fd_s${S}.csv \
    --b0-ckpt checkpoints/abl_B0_s${S}.pt --s-ckpt checkpoints/abl_S_s${S}.pt \
    > logs/H1_analysis_s${S}.log 2>&1
  cat logs/H1_analysis_s${S}.log
done

echo "[$(date '+%F %T')] H1 extra-seed evaluation: DONE"
