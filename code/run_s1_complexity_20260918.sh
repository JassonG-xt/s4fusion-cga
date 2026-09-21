#!/usr/bin/env bash
# S1 (review round 2026-09-18): re-measure the incremental-cost table with variance.
#
# The frozen table (`_temp/complexity_b2.csv`) holds ONE measurement per cell, with
# warmup 5 / iters 20. The review asked for >=10 warmup, >=30 forwards, an explicit
# sync discipline and a reported spread. This runner repeats the timed loop 3x per
# cell (mean +/- std reported) and writes a NEW csv, leaving the frozen artifact
# untouched.
#
# Run inside WSL:  bash code/run_s1_complexity_20260918.sh
set -uo pipefail

CODE=/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code
PY=/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/.venv-brss/bin/python
TEMP=/mnt/e/lunwen/S4Fusion-main/_temp
OUT="$TEMP/complexity_s1_20260918.csv"
LOG="$CODE/logs/s1_complexity_20260918.log"
OFFICIAL=/mnt/e/lunwen/S4Fusion-main/S4Fusion-main/model/model.pkl

mkdir -p "$CODE/logs"
rm -f "$OUT"

cd "$CODE" || exit 1
{
  echo "[s1 $(date +%H:%M:%S)] profiling 3 arms x 2 sizes, warmup=10 iters=30 repeats=3"
  "$PY" profile_brss_complexity.py --tag official \
      --checkpoint "$OFFICIAL" \
      --boundary-mode none --gate-type scalar --fusion-scales none \
      --warmup 10 --iters 30 --repeats 3 --out-csv "$OUT"
  "$PY" profile_brss_complexity.py --tag B0 \
      --checkpoint checkpoints/abl_B0_s42.pt \
      --boundary-mode none --gate-type scalar --fusion-scales none \
      --warmup 10 --iters 30 --repeats 3 --out-csv "$OUT"
  "$PY" profile_brss_complexity.py --tag CGA \
      --checkpoint checkpoints/abl_CGA_str_s42.pt --use-cga --cga-hidden 16 \
      --boundary-mode none --gate-type scalar --fusion-scales none \
      --warmup 10 --iters 30 --repeats 3 --out-csv "$OUT"
  echo "[s1 $(date +%H:%M:%S)] done -> $OUT"
} 2>&1 | tee -a "$LOG"
