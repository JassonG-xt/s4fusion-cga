#!/usr/bin/env bash
# P1 experiment runner (2026-09-14, rev 2 after the 5-expert panel review).
#
# Batch entry point so the whole P1 wave can be launched from a single shell
# invocation instead of one call per step. Every invocation needs explicit
# approval where the sandbox blacklists wsl.exe, so batching keeps that count low.
#
# Usage:  bash run_p1_experiments.sh <stage>
#   stage = b1        content control at BOTH endpoints (fusion + detection)
#   stage = b2        complexity profile (official weights / B0 / CGA)
#   stage = b3train   fire the 32-epoch B0 control queue in the background
#   stage = b3eval    evaluate the 32-epoch B0 arm (run after b3train finishes)
#   stage = all       b2 -> fire b3train -> b1
#
# NOTE on `all`: it fires the training queue FIRST (via tmux, so it returns
# immediately) and only then runs b1. Do not restore the older b2 -> b1 -> b3train
# order: that parks the longest path (16.4 h of training) behind b1.
#
# All scratch output goes to /mnt/e/lunwen/S4Fusion-main/_temp (outside the repo).
#
# ---------------------------------------------------------------------------
# Panel-review fixes baked into this revision:
#   M2  content control must not mix checkpoints across endpoints. The frozen
#       fusion-end control used abl_CGA_gatelr_s42.pt, while the detection-end
#       plan used abl_CGA_str_s42.pt -- two different models, and GATE1_AMEND 搂5
#       forbids mixing them. This script now uses abl_CGA_str_s42.pt (the
#       main-table arm) at BOTH endpoints, and re-runs the fusion-end control
#       with that checkpoint into a NEW directory (results/h5_str) so the frozen
#       results/h5 (gatelr) stays untouched.
#   M3  completeness assertions: generation must produce the expected image count
#       and detection must produce labels, or we abort instead of silently
#       scoring missing predictions as "not detected".
#   M4  b3eval completion is verified by ARTIFACT existence, not by a log
#       substring that is already present from a previous run.
#   M8  b1's generation stage uses the GPU; it must not start while the training
#       session is live unless explicitly allowed.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv-brss/bin/python
OLDPY=/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/.venv/bin/python
TEMP=/mnt/e/lunwen/S4Fusion-main/_temp
LOG="$TEMP/p1_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$TEMP" "$TEMP/panel"

# The single checkpoint used by BOTH content-control endpoints (M2). Seed 42 is
# the seed whose row feeds the main table, and its variant is the un-suffixed
# CGA_str (the other seeds carry an _s<seed> suffix).
CGA_CKPT=checkpoints/abl_CGA_str_s42.pt
LEARNED=CGA_str
MANIFEST=../dataset/manifests/m3fd_test.csv
EXPECT_IMAGES=300

say() { echo "[p1 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
die() { echo "[p1 $(date +%H:%M:%S)] FATAL: $*" | tee -a "$LOG"; exit 1; }

# ---------------------------------------------------------------- stage: b2
stage_b2() {
  say "B-2 profiling: official weights / B0 / CGA  (sizes 512^2 + 1024x768 native)"
  $PY profile_brss_complexity.py --tag official \
      --checkpoint ../../S4Fusion-main/model/model.pkl \
      --boundary-mode none --gate-type scalar --fusion-scales none \
      --out-csv "$TEMP/complexity_b2.csv" 2>&1 | tee -a "$LOG"
  $PY profile_brss_complexity.py --tag B0 \
      --checkpoint checkpoints/abl_B0_s42.pt \
      --boundary-mode none --gate-type scalar --fusion-scales none \
      --out-csv "$TEMP/complexity_b2.csv" 2>&1 | tee -a "$LOG"
  $PY profile_brss_complexity.py --tag CGA \
      --checkpoint "$CGA_CKPT" --use-cga --cga-hidden 16 \
      --boundary-mode none --gate-type scalar --fusion-scales none \
      --out-csv "$TEMP/complexity_b2.csv" 2>&1 | tee -a "$LOG"
  say "B-2 cross-check: params from checkpoints via analyze_h1._param_count"
  $PY - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys; sys.path.insert(0, ".")
from analyze_h1 import _param_count
for tag, p in [("official", "../../S4Fusion-main/model/model.pkl"),
               ("B0", "checkpoints/abl_B0_s42.pt"),
               ("CGA", "checkpoints/abl_CGA_str_s42.pt")]:
    try:
        print(f"  [cross-check] {tag}: {_param_count(p)} params")
    except Exception as e:
        print(f"  [cross-check] {tag}: FAILED {e}")
PYEOF
  say "B-2 done -> $TEMP/complexity_b2.csv"
  say "M8: read peak_mem_mb above to decide whether b1 may run concurrently with training"
}

# ---------------------------------------------------------------- stage: b1
stage_b1() {
  # --- M8: the generation stage uses the GPU, so refuse to start it while the
  # training session is live unless the operator has measured the headroom.
  if tmux has-session -t b3 2>/dev/null; then
    if [ "${ALLOW_B1_CONCURRENT:-0}" != "1" ]; then
      die "tmux session 'b3' is running and B-1's generation stage uses the GPU. \
Run B-2 first and check its peak_mem_mb against the 4 GB capacity, then re-run with \
ALLOW_B1_CONCURRENT=1 if there is real headroom; otherwise wait for training to finish."
    fi
    say "WARNING: running b1 concurrently with the b3 training session (ALLOW_B1_CONCURRENT=1)"
  fi

  # --- M2: fusion-end content control, re-run with the SAME checkpoint as the
  # detection endpoint so the two are comparable. Writes to results/h5_str so the
  # frozen results/h5 (gatelr) is never overwritten.
  say "B-1/M2 fusion-end control with the same checkpoint ($CGA_CKPT) -> results/h5_str"
  for MODE in none uniform shuffle; do
    CSV="results/h5_str/${MODE}.csv"
    if [ ! -f "$CSV" ]; then
      $PY evaluate_brss.py --data-root ../dataset --manifest "$MANIFEST" \
          --checkpoint "$CGA_CKPT" --use-cga --cga-hidden 16 \
          --cga-conflict-override "$MODE" \
          --fusion-scales none --gate-type scalar --boundary-mode none \
          --output-dir "results/h5_str/${MODE}/m3fd" --metrics-csv "$CSV" \
          --tile-size 507 --tile-overlap 64 --amp --save-rgb 2>&1 | tee -a "$LOG" \
        || say "WARNING fusion-end control $MODE returned non-zero"
    else
      say "fusion-end control $MODE exists, skip"
    fi
  done

  # --- detection-end content control -------------------------------------
  say "B-1 detection-end: generate uniform / shuffle fused images from $CGA_CKPT"
  for MODE in uniform shuffle; do
    V="cga_${MODE}"
    if [ ! -d "results/arb/$V/images" ]; then
      $PY gen_cga_fused.py --variant "$V" --checkpoint "$CGA_CKPT" --use-cga \
          --cga-conflict-override "$MODE" --full 2>&1 | tee -a "$LOG" \
        || die "generation failed for $V"
    else
      say "B-1 $V images exist, skip generation"
    fi
    # --- M3: completeness assertion on the generation stage
    N=$(ls "results/arb/$V/images" 2>/dev/null | wc -l)
    [ "$N" -eq "$EXPECT_IMAGES" ] \
      || die "M3: $V produced $N/$EXPECT_IMAGES fused images -- generation is INCOMPLETE. \
Refusing to run stats on a partial arm (missing images would be scored as 'not detected')."
    say "M3 integrity: $V images = $N/$EXPECT_IMAGES OK"
  done

  say "B-1 detection-end: detection on the full 300 (legacy venv, CPU)"
  $OLDPY arb_detect_full300.py cga_uniform cga_shuffle 2>&1 | tee -a "$LOG" \
    || say "WARNING detect stage returned non-zero"

  # --- M3: completeness assertion on the detection stage
  for MODE in uniform shuffle; do
    V="cga_${MODE}"
    L="results/arb/$V/runs_full/det/labels"
    [ -d "$L" ] || die "M3: $V has no detection label directory -- detection did not run"
    N=$(ls "$L" 2>/dev/null | wc -l)
    [ "$N" -gt 0 ] || die "M3: $V produced 0 detection labels"
    say "M3 integrity: $V detection labels = $N"
  done

  say "B-1 detection-end: high-conflict statistics vs the learned map (${LEARNED})"
  for MODE in uniform shuffle; do
    V="cga_${MODE}"
    say "--- $V vs $LEARNED  (read [primary-H5] and [integrity]; [gate-C1] is the stricter gate) ---"
    $PY arb_full300_stats.py "$V" "$LEARNED" 2>&1 | tee -a "$LOG" \
      || say "WARNING stats $V returned non-zero (see [integrity] above)"
  done
  say "B-1 done. Read [primary-H5] for R2: learned>uniform AND learned>shuffle => R2 holds;"
  say "      uniform>=learned => R2 must be rewritten. Record every outcome in GATE1_UNFREEZE."
}

# ---------------------------------------------------------------- stage: b3train
stage_b3train() {
  say "B-3 firing 32-epoch B0 control queue in the background"
  if tmux has-session -t b3 2>/dev/null; then
    say "tmux session b3 already exists; attach with: tmux attach -t b3"
    return 0
  fi
  tmux new-session -d -s b3 "cd $(pwd) && bash run_seeds_queue_v2.sh >> logs/seeds_queue.log 2>&1" \
    || die "tmux new-session failed"
  sleep 2
  tmux ls 2>&1 | tee -a "$LOG" || die "tmux ls failed -- the queue did not start"
  say "B-3 queue started; progress: tail -f logs/seeds_queue.log"
}

# ---------------------------------------------------------------- stage: b3eval
stage_b3eval() {
  # --- M4: completion is verified by ARTIFACTS, not by a log substring. The log
  # already contains "all seeds done" from the previous run, so a substring check
  # would pass with zero evidence.
  say "B-3/M4 verifying training artifacts before evaluating"
  for S in 42 123 3407; do
    [ -f "checkpoints/abl_B0_e32_s${S}.pt" ] \
      || die "M4: checkpoints/abl_B0_e32_s${S}.pt missing -- training is incomplete. Refusing to evaluate."
    [ -f "logs/seed_done_B0e32_s${S}" ] \
      || die "M4: marker logs/seed_done_B0e32_s${S} missing -- that seed did not finish cleanly."
  done
  say "M4 integrity: 3/3 e32 checkpoints and 3/3 markers present"

  say "B-3 evaluating the 32-epoch B0 arm"
  bash run_after_seeds_v2.sh 2>&1 | tee -a "$LOG"

  # --- M4: verify the expected outputs actually landed
  say "B-3/M4 verifying evaluation outputs"
  for S in 42 123 3407; do
    [ -f "results/seeds/B0_e32_s${S}.csv" ] || die "M4: results/seeds/B0_e32_s${S}.csv missing"
  done
  [ -f "results/seeds/seeds_table_e32.md" ] || die "M4: results/seeds/seeds_table_e32.md missing"
  say "M4 integrity: e32 seed table + 3/3 per-seed CSVs present"
  say "B-3 done -> results/seeds/seeds_table_e32.md  (now run arb_full300_stats.py CGA_str_s<seed> B0_e32_s<seed>)"
}

case "${1:-all}" in
  b1)      stage_b1 ;;
  b2)      stage_b2 ;;
  b3train) stage_b3train ;;
  b3eval)  stage_b3eval ;;
  all)     stage_b2; stage_b3train; stage_b1 ;;
  *)       echo "usage: $0 {b1|b2|b3train|b3eval|all}"; exit 2 ;;
esac

say "stage '${1:-all}' finished; log: $LOG"
