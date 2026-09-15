#!/usr/bin/env bash
# Fair D1 test: retrain B0 + S at crop 256 (4x the scan sequence length of the
# 128 run) so the mechanism is actually stressed by long sequences in training.
# Resume-aware (atomic checkpoints) and safe to re-run after a shutdown.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p logs checkpoints
COMMON="--data-root ../dataset \
  --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --crop-size 256 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 15 --lr 2e-5 --seed 42 --amp \
  --num-workers 2 --max-val-steps 200"

B0R=""; [ -f checkpoints/abl_B0_c256_s42_last.pt ] && B0R="--resume checkpoints/abl_B0_c256_s42_last.pt"
echo "[$(date '+%F %T')] B0 crop256 ${B0R:-fresh}"
$PY train_brss.py $COMMON --boundary-mode none --fusion-scales none --gate-type scalar $B0R \
  --save-path checkpoints/abl_B0_c256_s42.pt >> logs/abl_B0_c256_s42.log 2>&1
echo "[$(date '+%F %T')] DONE B0"

SR=""; [ -f checkpoints/abl_S_c256_s42_last.pt ] && SR="--resume checkpoints/abl_S_c256_s42_last.pt"
echo "[$(date '+%F %T')] S crop256 ${SR:-fresh}"
$PY train_brss.py $COMMON --boundary-mode state --fusion-scales none --gate-type scalar $SR \
  --save-path checkpoints/abl_S_c256_s42.pt >> logs/abl_S_c256_s42.log 2>&1
echo "[$(date '+%F %T')] DONE S — crop256 B0+S complete"
