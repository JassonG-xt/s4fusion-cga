#!/usr/bin/env bash
# Auto-run the CGA A/B eval when training finishes (epoch 20 + checkpoint).
cd "$(dirname "$0")"
while pgrep -f "run_cga_train.sh" >/dev/null 2>&1; do sleep 180; done
if grep -qa "epoch=20 " logs/abl_CGA_s42.log 2>/dev/null && [ -f checkpoints/abl_CGA_s42.pt ]; then
  echo "[$(date '+%F %T')] CGA training done -> eval" > logs/eval_cga.log
  chmod +x run_eval_cga.sh
  ./run_eval_cga.sh >> logs/eval_cga.log 2>&1
  echo "[$(date '+%F %T')] done" >> logs/eval_cga.log
else
  echo "[$(date '+%F %T')] CGA training NOT complete (shutdown?); re-run run_cga_train.sh then run_eval_cga.sh" > logs/eval_cga.log
fi
