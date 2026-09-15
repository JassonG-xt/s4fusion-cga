#!/bin/bash
# B1 seed queue: CGA_str s123/s3407 (main method first) + B0 s123/s3407.
# Same regime as s42 checkpoints (crop128, 20ep B0 / 32ep CGA_str). No param tuning.
set -e
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
MANIFEST=../dataset/manifests/train_all.csv
VAL=../dataset/manifests/val_all.csv
CKPT=../../S4Fusion-main/model/model.pkl

echo "[queue] $(date) start"
for SEED in 123 3407; do
  if [ ! -f "checkpoints/abl_CGA_str_s${SEED}.pt" ]; then
    echo "[queue] $(date) CGA_str seed ${SEED} train"
    $PY train_brss.py --data-root ../dataset --train-manifest $MANIFEST --val-manifest $VAL \
      --checkpoint $CKPT --use-cga --cga-hidden 16 \
      --w-cga-conflict 0.1 --w-cga-commit 10.0 --gate-lr 0.01 \
      --crop-size 128 --batch-size 1 --accumulation-steps 8 \
      --samples-per-epoch 2048 --epochs 32 --lr 2e-5 --seed ${SEED} --amp \
      --boundary-mode none --fusion-scales none \
      --save-path checkpoints/abl_CGA_str_s${SEED}.pt \
      --resume checkpoints/abl_CGA_str_s${SEED}_last.pt 2>/dev/null || \
    $PY train_brss.py --data-root ../dataset --train-manifest $MANIFEST --val-manifest $VAL \
      --checkpoint $CKPT --use-cga --cga-hidden 16 \
      --w-cga-conflict 0.1 --w-cga-commit 10.0 --gate-lr 0.01 \
      --crop-size 128 --batch-size 1 --accumulation-steps 8 \
      --samples-per-epoch 2048 --epochs 32 --lr 2e-5 --seed ${SEED} --amp \
      --boundary-mode none --fusion-scales none \
      --save-path checkpoints/abl_CGA_str_s${SEED}.pt
  fi
done
for SEED in 123 3407; do
  if [ ! -f "checkpoints/abl_B0_s${SEED}.pt" ]; then
    echo "[queue] $(date) B0 seed ${SEED} train"
    $PY train_brss.py --data-root ../dataset --train-manifest $MANIFEST --val-manifest $VAL \
      --checkpoint $CKPT --crop-size 128 --batch-size 1 --accumulation-steps 8 \
      --samples-per-epoch 2048 --epochs 20 --lr 2e-5 --seed ${SEED} --amp \
      --boundary-mode none --fusion-scales none --gate-type scalar \
      --save-path checkpoints/abl_B0_s${SEED}.pt \
      --resume checkpoints/abl_B0_s${SEED}_last.pt 2>/dev/null || \
    $PY train_brss.py --data-root ../dataset --train-manifest $MANIFEST --val-manifest $VAL \
      --checkpoint $CKPT --crop-size 128 --batch-size 1 --accumulation-steps 8 \
      --samples-per-epoch 2048 --epochs 20 --lr 2e-5 --seed ${SEED} --amp \
      --boundary-mode none --fusion-scales none --gate-type scalar \
      --save-path checkpoints/abl_B0_s${SEED}.pt
  fi
done
echo "[queue] $(date) all seeds done"
