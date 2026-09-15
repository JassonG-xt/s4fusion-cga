#!/usr/bin/env bash
# Background monitor for the strengthened-CGA pipeline (train pid arg $1).
# Appends timestamped snapshots to logs/str_monitor.log every 5 minutes;
# exits after the eval writes its gate verdicts (or on crash detection).
set -u
cd "$(dirname "$0")"
TRAIN_PID="${1:-66762}"
M=logs/str_monitor.log
log() { echo "[$(date '+%F %T')] $*" >> "$M"; }
log "monitor started (train pid=$TRAIN_PID)"
while true; do
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    EP=$(grep -c "^epoch=" logs/abl_CGA_str_s42.log 2>/dev/null || echo 0)
    LAST=$(grep "^epoch=" logs/abl_CGA_str_s42.log 2>/dev/null | tail -1)
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)
    log "train alive | epochs_done=$EP | $LAST | gpu=$GPU"
  else
    log "train process exited"
    if grep -q "DONE STR-CGA" logs/str_pipeline_wrapper.log 2>/dev/null; then
      log "training finished OK; eval phase"
      for i in $(seq 1 60); do
        sleep 120
        T=$(grep -c "\[gate-" logs/eval_str_cga.log 2>/dev/null || echo 0)
        log "eval pending... (gate lines so far: $T)"
        if grep -q "STR-CGA eval done" logs/eval_str_cga.log 2>/dev/null; then
          log "=== FINAL VERDICT ==="
          grep -E "\[gate-|recall ALL|recall HIGH|mAP50:" logs/eval_str_cga.log >> "$M"
          log "=== monitor exit (complete) ==="
          exit 0
        fi
      done
      log "eval did not finish within timeout window"
      exit 1
    else
      log "ALERT: training died without DONE marker; check logs/abl_CGA_str_s42.log"
      tail -5 logs/abl_CGA_str_s42.log >> "$M" 2>/dev/null
      exit 2
    fi
  fi
  sleep 300
done
