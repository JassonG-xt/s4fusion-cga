#!/bin/bash
# v2 post-seed chain (2026-09-10). Replaces run_b2_after_seeds.sh, whose
# "all four checkpoints exist" trigger would fire prematurely (best checkpoints
# are written from epoch 1 onward). Trigger here = the "[queue] ... all seeds
# done" marker in logs/seeds_queue.log, written only after the LAST seed's
# full training completes.
#
# For each new seed (123, 3407):
#   CGA_str  -> full-300 fused images + detection + high-conflict stats
#               + structural non-inferiority + unified metric CSV
#               (results/seeds/CGA_str_s<seed>.csv, eval_metrics.py protocol)
#   B0       -> same fused/metric chain (results/seeds/B0_s<seed>.csv)
# Then analyze_seeds.py -> 3-seed table (42 + 123 + 3407) incl. pooled
# per-image Wilcoxon + Holm.
#
# s42 rows are produced once from the existing CGA_str/B0cmp_full images so the
# 3-seed table is complete in a single pass.
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
OLDPY=/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/.venv/bin/python
QUEUE_LOG=logs/seeds_queue.log
SEEDS="123 3407"

echo "[chain] $(date) waiting for seed queue completion marker"
while ! grep -q "all seeds done" "$QUEUE_LOG" 2>/dev/null; do
  sleep 300
done
echo "[chain] $(date) seed queue complete, starting evals"

# --- unified structural-metric CSV for one cell, naming convention <cell>_s<seed>.csv ---
metric_csv () {  # $1 cell dir name (results/arb/<cell>), $2 out csv
  if [ ! -f "$2" ]; then
    $PY eval_metrics.py --ir-path baselines/inputs/m3fd/ir --vi-path baselines/inputs/m3fd/vi \
      --fused-path "results/arb/$1/images" --use-y --skip-perceptual --out-csv "$2"
  else
    echo "[skip] $2 exists"
  fi
}

# --- s42 rows from existing images (CGA_str s42 + B0cmp_full = B0 s42) ---
metric_csv CGA_str results/seeds/CGA_str_s42.csv
metric_csv B0cmp_full results/seeds/B0_s42.csv

for SEED in $SEEDS; do
  # CGA_str seed eval (GPU gen + CPU detect + stats)
  if [ ! -f "results/seeds/CGA_str_s${SEED}.csv" ]; then
    echo "[chain] $(date) CGA_str s${SEED} eval"
    $PY gen_cga_fused.py --variant "CGA_str_s${SEED}" --checkpoint "checkpoints/abl_CGA_str_s${SEED}.pt" --use-cga --full
    $OLDPY arb_detect_full300.py "CGA_str_s${SEED}" || echo "[warn] detect CGA_str_s${SEED} failed"
    $PY arb_full300_stats.py "CGA_str_s${SEED}" B0cmp_full || echo "[warn] stats CGA_str_s${SEED} failed"
    $PY struct_noninferior.py "CGA_str_s${SEED}" B0cmp_full || echo "[warn] struct CGA_str_s${SEED} failed"
    metric_csv "CGA_str_s${SEED}" "results/seeds/CGA_str_s${SEED}.csv"
  else
    echo "[skip] CGA_str_s${SEED} metrics exist"
  fi
  # B0 seed eval
  if [ ! -f "results/seeds/B0_s${SEED}.csv" ]; then
    echo "[chain] $(date) B0 s${SEED} eval"
    $PY gen_cga_fused.py --variant "B0_s${SEED}" --checkpoint "checkpoints/abl_B0_s${SEED}.pt" --full
    metric_csv "B0_s${SEED}" "results/seeds/B0_s${SEED}.csv"
  else
    echo "[skip] B0_s${SEED} metrics exist"
  fi
done

# ============================================================
# B-3 decision point (2026-09-14): budget-matched 32-epoch B0 control.
# Does NOT touch the frozen 20-epoch path above (B0_s*.csv / seeds_table.md).
# All artifacts carry an _e32_ tag and land in separate files.
# NOTE 1: s42's CGA variant is the un-suffixed "CGA_str"; s123/s3407 are
#         "CGA_str_s<seed>".
# NOTE 2: the frozen protocol compared every seed against the shared
#         B0cmp_full base (s42's 20-epoch B0). Here each seed is compared
#         against its OWN budget-matched B0_e32 base 鈥?deliberately stricter,
#         and therefore not numerically comparable to the frozen deltas.
# ============================================================
for PAIR in 42:CGA_str 123:CGA_str_s123 3407:CGA_str_s3407; do
  SEED=${PAIR%%:*}
  CGA_V=${PAIR##*:}
  CKPT_B0="checkpoints/abl_B0_e32_s${SEED}.pt"
  if [ ! -f "$CKPT_B0" ]; then
    echo "[chain-e32] $(date) MISSING $CKPT_B0 - skip seed ${SEED} (training incomplete?)"
    continue
  fi
  if [ ! -f "results/seeds/B0_e32_s${SEED}.csv" ]; then
    echo "[chain-e32] $(date) B0_e32 s${SEED}: fused images + metrics"
    $PY gen_cga_fused.py --variant "B0_e32_s${SEED}" --checkpoint "$CKPT_B0" --full
    metric_csv "B0_e32_s${SEED}" "results/seeds/B0_e32_s${SEED}.csv"
  else
    echo "[skip] B0_e32_s${SEED} metrics exist"
  fi
  if [ ! -d "results/arb/B0_e32_s${SEED}/runs_full/det/labels" ]; then
    echo "[chain-e32] $(date) B0_e32 s${SEED}: detection labels"
    $OLDPY arb_detect_full300.py "B0_e32_s${SEED}" || echo "[warn] detect B0_e32_s${SEED} failed"
  else
    echo "[skip] B0_e32_s${SEED} detection exists"
  fi
  echo "[chain-e32] $(date) budget-matched H5: ${CGA_V} vs B0_e32_s${SEED}"
  $PY arb_full300_stats.py "${CGA_V}" "B0_e32_s${SEED}" \
    || echo "[warn] stats ${CGA_V} vs B0_e32_s${SEED} failed"
done

# --- decision-point seed table (32-epoch matched budget; frozen table untouched) ---
$PY analyze_seeds.py --glob-cga 'results/seeds/CGA_str_s*.csv' \
  --glob-b0 'results/seeds/B0_e32_s*.csv' --out results/seeds/seeds_table_e32.md \
  && echo "[chain-e32] seeds_table_e32.md written"

# --- 3-seed table ---
$PY analyze_seeds.py --glob-cga 'results/seeds/CGA_str_s*.csv' \
  --glob-b0 'results/seeds/B0_s*.csv' --out results/seeds/seeds_table.md \
  && echo "[chain] seeds_table.md written"

# --- B2 (C2 distillation): teacher = CGA_str s42, student depths 1,1,1 ---
# The distillation arm is a REGISTERED FAILURE: it never produced a usable
# checkpoint (validation NaN from epoch 1, best never saved) and the
# pre-registration forbids re-tuning or re-running it.
# The original guard tested for checkpoints/distill_student_s42.pt, but that best
# file was never written -- only the unusable distill_student_s42_last.pt exists
# -- so `[ ! -f ... ]` evaluated TRUE and every chain invocation would silently
# re-run the registered failure, violating the pre-registration and contaminating
# logs/distill.log. The guard is therefore inverted behind an explicit opt-in:
# the block runs only when RUN_DISTILL=1 is set deliberately.
if [ "${RUN_DISTILL:-0}" != "1" ]; then
  echo "[skip] distillation is a registered failure; not re-run (set RUN_DISTILL=1 to override)"
elif [ ! -f checkpoints/distill_student_s42.pt ]; then
  echo "[chain] $(date) B2 distillation start (explicitly requested via RUN_DISTILL=1)"
  ./run_distill.sh >> logs/distill.log 2>&1 && echo "[chain] $(date) distill done"
else
  echo "[skip] checkpoints/distill_student_s42.pt exists"
fi

echo "[chain] $(date) all done"
