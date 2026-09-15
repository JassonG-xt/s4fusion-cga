#!/bin/bash
# B2: distill the full CGA_str teacher into a compact student.
# Teacher: abl_CGA_str_s42 (depths 1,2,1, trained with --use-cga).
# Student: shallower depths 1,1,1 + CGA, distilled via train_brss.py's
# teacher branch (output-L1 + feature-SmoothL1 + boundary-KL distillation).
# Same data regime as the s42 ablations so the efficiency comparison is fair.
set -e
cd "$(dirname "$0")"
PY=.venv-brss/bin/python

echo "[distill] $(date) start"
$PY train_brss.py --data-root ../dataset \
  --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --teacher-checkpoint checkpoints/abl_CGA_str_s42.pt \
  --teacher-depths 1,2,1 \
  --depths 1,1,1 \
  --use-cga --cga-hidden 16 \
  --w-cga-conflict 0.1 --w-cga-commit 10.0 --gate-lr 0.01 \
  --w-distill-output 1.0 --w-distill-feature 0.2 --w-distill-boundary 0.5 \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 20 --lr 2e-5 --seed 42 --amp \
  --boundary-mode none --fusion-scales none \
  --save-path checkpoints/distill_student_s42.pt \
  --resume checkpoints/distill_student_s42_last.pt 2>/dev/null || \
$PY train_brss.py --data-root ../dataset \
  --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --teacher-checkpoint checkpoints/abl_CGA_str_s42.pt \
  --teacher-depths 1,2,1 \
  --depths 1,1,1 \
  --use-cga --cga-hidden 16 \
  --w-cga-conflict 0.1 --w-cga-commit 10.0 --gate-lr 0.01 \
  --w-distill-output 1.0 --w-distill-feature 0.2 --w-distill-boundary 0.5 \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 20 --lr 2e-5 --seed 42 --amp \
  --boundary-mode none --fusion-scales none \
  --save-path checkpoints/distill_student_s42.pt
echo "[distill] $(date) done"
