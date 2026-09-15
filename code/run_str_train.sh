#!/usr/bin/env bash
# Strengthened CGA training per pre-registered success criteria (2026-08-24):
#   C1 high-conflict recall >= +0.02 with p<0.05 (Holm, single primary endpoint)
#   C2 mAP50 >= +0.01 on the full official M3FD-300 test split
#   C3 SF/AG structural metrics non-inferior to the same-regime baseline
# Design: build on the only variant with a positive signal (abl_CGA_gatelr_s42 =
# w_cga_commit=10 + gate_lr=0.01), extend 20 -> 32 epochs, keep every other knob
# identical for same-regime comparability against abl_B0_s42 (crop128, seed 42).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p logs checkpoints
R=""; [ -f checkpoints/abl_CGA_str_s42_last.pt ] && R="--resume checkpoints/abl_CGA_str_s42_last.pt"
echo "[$(date '+%F %T')] STR-CGA train ${R:-fresh}"
$PY train_brss.py \
  --data-root ../dataset --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --boundary-mode none --fusion-scales none --gate-type scalar \
  --use-cga --cga-hidden 16 --w-cga-conflict 0.1 --w-cga-commit 10.0 \
  --gate-lr 0.01 \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 32 --lr 2e-5 --seed 42 --amp \
  --num-workers 2 --max-val-steps 200 $R \
  --save-path checkpoints/abl_CGA_str_s42.pt >> logs/abl_CGA_str_s42.log 2>&1
echo "[$(date '+%F %T')] DONE STR-CGA"
