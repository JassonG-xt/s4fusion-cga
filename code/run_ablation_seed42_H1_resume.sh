#!/usr/bin/env bash
# Resume Step 1 (H1) after a mid-run shutdown.
# State at resume: B0 completed epoch 13 (best checkpoint clean); S not started.
# Checkpoint saves are now atomic (write-tmp-then-rename), so another interruption
# cannot truncate a good checkpoint. If this is itself interrupted, re-run: B0's
# --resume picks up abl_B0_s42.pt, and re-add --resume for S if its _last exists.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p logs checkpoints
COMMON="--data-root ../dataset \
  --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 20 --lr 2e-5 --seed 42 --amp \
  --num-workers 2 --max-val-steps 200"

echo "[$(date '+%F %T')] RESUME B0 from epoch-13 checkpoint"
$PY train_brss.py $COMMON --boundary-mode none --fusion-scales none --gate-type scalar \
  --resume checkpoints/abl_B0_s42.pt \
  --save-path checkpoints/abl_B0_s42.pt >> logs/abl_B0_s42.log 2>&1
echo "[$(date '+%F %T')] DONE B0"

echo "[$(date '+%F %T')] START S (fresh, boundary-mode state)"
$PY train_brss.py $COMMON --boundary-mode state --fusion-scales none --gate-type scalar \
  --save-path checkpoints/abl_S_s42.pt > logs/abl_S_s42.log 2>&1
echo "[$(date '+%F %T')] DONE S — B0 + S seed=42 complete"
