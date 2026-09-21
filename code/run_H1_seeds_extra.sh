#!/usr/bin/env bash
# H1 gate, extra seeds: S arm (boundary-mode state) for seeds 123 and 3407.
# Only the S arm is trained: abl_B0_s{42,123,3407}.pt already exist.
# Sequential, so one 4 GB GPU is not oversubscribed. Marker-file driven and
# idempotent: rerun after an interruption and finished seeds are skipped.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p logs checkpoints

COMMON="--data-root ../dataset \
  --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 20 --lr 2e-5 --amp \
  --num-workers 2 --max-val-steps 200"

echo "[$(date '+%F %T')] H1 extra seeds: start"

for S in 123 3407; do
  MARK="checkpoints/abl_S_s${S}.done"
  if [ -f "$MARK" ]; then
    echo "[$(date '+%F %T')] skip s${S} (marker present)"
    continue
  fi
  LAST="checkpoints/abl_S_s${S}_last.pt"
  EXTRA=""
  if [ -f "$LAST" ]; then
    EXTRA="--resume checkpoints/abl_S_s${S}_last.pt"
    echo "[$(date '+%F %T')] resume s${S} from $LAST"
  else
    echo "[$(date '+%F %T')] start s${S}"
  fi
  $PY train_brss.py $COMMON --seed ${S} \
    --boundary-mode state --fusion-scales none --gate-type scalar $EXTRA \
    --save-path checkpoints/abl_S_s${S}.pt > logs/abl_S_s${S}.log 2>&1
  echo "[$(date '+%F %T')] done s${S}"
  touch "$MARK"
done

echo "[$(date '+%F %T')] H1 extra seeds: ALL DONE"
