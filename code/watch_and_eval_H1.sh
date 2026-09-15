#!/usr/bin/env bash
# Best-effort watcher: wait until S training exits, then (only if S reached
# epoch 20 with a checkpoint) run the H1 evaluation. If a shutdown kills this,
# just run ./run_eval_H1.sh by hand once S is done.
cd "$(dirname "$0")"
while pgrep -f "train_brss.py.*abl_S_s42.pt" >/dev/null 2>&1; do
  sleep 120
done
if grep -qa "epoch=20 " logs/abl_S_s42.log 2>/dev/null && [ -f checkpoints/abl_S_s42.pt ]; then
  echo "[$(date '+%F %T')] S complete -> running H1 evaluation" > logs/eval_H1.log
  ./run_eval_H1.sh >> logs/eval_H1.log 2>&1
  echo "[$(date '+%F %T')] eval done" >> logs/eval_H1.log
else
  echo "[$(date '+%F %T')] S did NOT reach epoch 20 cleanly; skipping eval" > logs/eval_H1.log
fi
