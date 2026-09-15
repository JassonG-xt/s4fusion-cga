#!/usr/bin/env bash
# Best-effort: when the crop-256 orchestrator exits and S reached epoch 15,
# auto-run the c256 evaluation. If a shutdown kills this, just run
# ./run_eval_c256.sh by hand once S_c256 shows epoch=15.
cd "$(dirname "$0")"
while pgrep -f "run_ablation_c256_H1.sh" >/dev/null 2>&1; do
  sleep 180
done
if grep -qa "epoch=15 " logs/abl_S_c256_s42.log 2>/dev/null && [ -f checkpoints/abl_S_c256_s42.pt ]; then
  echo "[$(date '+%F %T')] crop256 training complete -> evaluating" > logs/eval_c256.log
  chmod +x run_eval_c256.sh
  ./run_eval_c256.sh >> logs/eval_c256.log 2>&1
  echo "[$(date '+%F %T')] eval done" >> logs/eval_c256.log
else
  echo "[$(date '+%F %T')] crop256 training NOT complete (shutdown?); re-run orchestrator then run_eval_c256.sh" > logs/eval_c256.log
fi
