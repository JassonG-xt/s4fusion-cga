#!/usr/bin/env bash
# Step 1 (H1 gate): mechanism ablation, seed=42, cells B0 (baseline) + S (state).
# Sequential to fit the 4 GB GPU. Full protocol budget (20 epochs x 2048 steps).
# In-training val capped at 200 for checkpoint selection; final metrics come from
# evaluate_brss.py on the test manifest, not from this val number.
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

echo "[$(date '+%F %T')] START B0 (boundary-mode none, fusion-scales none)"
$PY train_brss.py $COMMON --boundary-mode none  --fusion-scales none --gate-type scalar \
  --save-path checkpoints/abl_B0_s42.pt > logs/abl_B0_s42.log 2>&1
echo "[$(date '+%F %T')] DONE B0"

echo "[$(date '+%F %T')] START S (boundary-mode state, fusion-scales none)"
$PY train_brss.py $COMMON --boundary-mode state --fusion-scales none --gate-type scalar \
  --save-path checkpoints/abl_S_s42.pt   > logs/abl_S_s42.log 2>&1
echo "[$(date '+%F %T')] DONE S"

echo "[$(date '+%F %T')] ALL DONE — B0 + S seed=42 trained"
