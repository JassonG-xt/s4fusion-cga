#!/bin/bash
# Smoke test: single-step train + confirm GPU path, then measure per-step time.
set -e
cd /mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code
PY=.venv-brss/bin/python
export PYTHONUNBUFFERED=1

echo "=== smoke: build + 1-step train (boundary-mode=state) ==="
$PY train_brss.py \
  --data-root ../dataset \
  --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --boundary-mode state --fusion-scales none \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 1 --lr 2e-5 --seed 42 --amp \
  --max-train-steps 2 --max-val-steps 2 \
  --save-path checkpoints/_smoke_state.pt

echo "=== smoke done ==="
