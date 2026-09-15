#!/bin/bash
# v2 seed queue (2026-09-10). Marker-file based completion guard 鈥?replaces the
# checkpoint-existence guard of run_seeds_queue.sh, which silently SKIPS a seed
# whose best checkpoint already exists after an interruption (best .pt is
# written from epoch 1 onward, before training completes).
#
# Markers: logs/seed_done_<cell>_s<seed> written only after the train command
# exits 0. Resume still handled by train_brss.py --resume <cell>_last.pt.
# Appends to the same logs/seeds_queue.log so run_after_seeds_v2.sh's
# "all seeds done" trigger keeps working.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
MANIFEST=../dataset/manifests/train_all.csv
VAL=../dataset/manifests/val_all.csv
CKPT=../../S4Fusion-main/model/model.pkl

echo "[queue] $(date) start (v2 marker-based)"
for SEED in 123 3407; do
  MARK=logs/seed_done_CGA_str_s${SEED}
  if [ ! -f "$MARK" ]; then
    echo "[queue] $(date) CGA_str seed ${SEED} train"
    # M5: resume WITHOUT a silent from-scratch fallback. The previous shape
    # (`--resume X 2>/dev/null || <from scratch>`) meant a crash at epoch 30
    # silently restarted from epoch 1 and OVERWROTE _last.pt, turning a 16 h run
    # into 22 h+ with no signal. Now the resume is attempted only when a _last
    # checkpoint exists, and its non-zero exit is allowed to propagate.
    CGA_ARGS=(--data-root ../dataset --train-manifest $MANIFEST --val-manifest $VAL \
      --checkpoint $CKPT --use-cga --cga-hidden 16 \
      --w-cga-conflict 0.1 --w-cga-commit 10.0 --gate-lr 0.01 \
      --crop-size 128 --batch-size 1 --accumulation-steps 8 \
      --samples-per-epoch 2048 --epochs 32 --lr 2e-5 --seed ${SEED} --amp \
      --boundary-mode none --fusion-scales none \
      --save-path checkpoints/abl_CGA_str_s${SEED}.pt)
    if [ -f "checkpoints/abl_CGA_str_s${SEED}_last.pt" ]; then
      $PY train_brss.py "${CGA_ARGS[@]}" --resume checkpoints/abl_CGA_str_s${SEED}_last.pt
      RC=$?
    else
      echo "[queue] $(date) CGA_str seed ${SEED}: no _last checkpoint, training from scratch"
      $PY train_brss.py "${CGA_ARGS[@]}"
      RC=$?
    fi
    if [ $RC -eq 0 ]; then touch "$MARK"; echo "[queue] $(date) CGA_str s${SEED} done (marker)"; else echo "[queue] $(date) CGA_str s${SEED} FAILED rc=$RC"; fi
  else
    echo "[queue] CGA_str s${SEED}: marker exists, skip"
  fi
done
# B0 arm. v3 (2026-09-14): budget-matched 32-epoch control added for the
# decision point. Seeds are now 42/123/3407 (s42 was previously trained
# out-of-band, but it is the ONLY seed with a budget asymmetry 鈥?CGA best ep26
# vs a 20-epoch B0 鈥?so it must be retrained at 32 epochs too). Marker and
# save-path are renamed *_e32_* so the frozen 20-epoch abl_B0_s*.pt stay intact.
for SEED in 42 123 3407; do
  MARK=logs/seed_done_B0e32_s${SEED}
  if [ ! -f "$MARK" ]; then
    echo "[queue] $(date) B0_e32 seed ${SEED} train"
    # M5: same explicit resume/scratch decision as the CGA loop above -- never
    # silently restart from epoch 1 after a late crash.
    B0_ARGS=(--data-root ../dataset --train-manifest $MANIFEST --val-manifest $VAL \
      --checkpoint $CKPT --crop-size 128 --batch-size 1 --accumulation-steps 8 \
      --samples-per-epoch 2048 --epochs 32 --lr 2e-5 --seed ${SEED} --amp \
      --boundary-mode none --fusion-scales none --gate-type scalar \
      --save-path checkpoints/abl_B0_e32_s${SEED}.pt)
    if [ -f "checkpoints/abl_B0_e32_s${SEED}_last.pt" ]; then
      $PY train_brss.py "${B0_ARGS[@]}" --resume checkpoints/abl_B0_e32_s${SEED}_last.pt
      RC=$?
    else
      echo "[queue] $(date) B0_e32 seed ${SEED}: no _last checkpoint, training from scratch"
      $PY train_brss.py "${B0_ARGS[@]}"
      RC=$?
    fi
    if [ $RC -eq 0 ]; then touch "$MARK"; echo "[queue] $(date) B0_e32 s${SEED} done (marker)"; else echo "[queue] $(date) B0_e32 s${SEED} FAILED rc=$RC"; fi
  else
    echo "[queue] B0_e32 s${SEED}: marker exists, skip"
  fi
done
echo "[queue] $(date) all seeds done"
