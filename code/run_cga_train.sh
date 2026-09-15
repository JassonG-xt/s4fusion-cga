#!/usr/bin/env bash
# Train CGA (Conflict-Gated Arbitration) fine-tuned from the S4Fusion checkpoint.
# crop128, 20 epochs, seed 42. Resume-aware + atomic checkpoints (shutdown-safe).
# Compare against the same-regime baseline abl_B0_s42 (crop128/20ep).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p logs checkpoints
R=""; [ -f checkpoints/abl_CGA_s42_last.pt ] && R="--resume checkpoints/abl_CGA_s42_last.pt"
echo "[$(date '+%F %T')] CGA train ${R:-fresh}"
$PY train_brss.py \
  --data-root ../dataset --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --boundary-mode none --fusion-scales none --gate-type scalar \
  --use-cga --cga-hidden 16 --w-cga-conflict 0.1 \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 20 --lr 2e-5 --seed 42 --amp \
  --num-workers 2 --max-val-steps 200 $R \
  --save-path checkpoints/abl_CGA_s42.pt >> logs/abl_CGA_s42.log 2>&1
echo "[$(date '+%F %T')] DONE CGA"
