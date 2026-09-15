#!/usr/bin/env bash
# Post-training eval for the strengthened CGA run (abl_CGA_str_s42).
# Full-300 detection A/B vs same-regime baseline, per-object high-conflict stats,
# structural non-inferiority, and the pre-registered gate verdict.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
OLDPY=/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/.venv/bin/python
CK=checkpoints/abl_CGA_str_s42.pt
LOG=logs/eval_str_cga.log
[ -f "$CK" ] || { echo "ERROR: $CK missing (training not done?)"; exit 1; }
[ -d results/arb/B0cmp_full/images ] || { echo "ERROR: B0cmp_full images missing"; exit 1; }

{
echo "[$(date '+%F %T')] STR-CGA eval start"
echo "[gen] CGA_str fused (full 300)"
$PY gen_cga_fused.py --variant CGA_str --checkpoint "$CK" --use-cga --full

echo "[detect] CGA_str + B0cmp_full (old venv, full 300)"
$OLDPY arb_detect_full300.py CGA_str B0cmp_full

echo "[stats] per-object high-conflict endpoint"
$PY arb_full300_stats.py CGA_str B0cmp_full

echo "[struct] SF/AG non-inferiority"
$PY struct_noninferior.py CGA_str B0cmp_full

echo "[gate-C2] mAP50 delta >= +0.01:"
MAP_TEST=$(grep -oP 'CGA_str full-300 P/R/mAP50/mAP5095: \[\K[^\]]+' "$LOG" 2>/dev/null | awk -F', ' '{print $3}')
MAP_BASE=$(grep -oP 'B0cmp_full full-300 P/R/mAP50/mAP5095: \[\K[^\]]+' "$LOG" 2>/dev/null | awk -F', ' '{print $3}')
if [ -n "$MAP_TEST" ] && [ -n "$MAP_BASE" ]; then
  D=$($PY -c "print(f'{$MAP_TEST-$MAP_BASE:+.4f}')")
  OK=$($PY -c "print('PASS' if $MAP_TEST-$MAP_BASE>=0.01 else 'FAIL')")
  echo "  mAP50: base=$MAP_BASE -> test=$MAP_TEST (delta=$D)  gate-C2: $OK"
else
  echo "  ERROR: mAP summary lines not found"
fi

echo "[$(date '+%F %T')] STR-CGA eval done"
} >> "$LOG" 2>&1
tail -40 "$LOG"
