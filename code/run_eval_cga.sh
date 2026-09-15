#!/usr/bin/env bash
# A/B the trained CGA vs its same-regime baseline (abl_B0_s42) on M3FD detection.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
OLDPY=/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/.venv/bin/python
[ -f checkpoints/abl_CGA_s42.pt ] || { echo "ERROR: abl_CGA_s42.pt missing"; exit 1; }
[ -f checkpoints/abl_B0_s42.pt ]  || { echo "ERROR: abl_B0_s42.pt missing";  exit 1; }
echo "[gen] baseline fused (abl_B0_s42, no cga)"
$PY gen_cga_fused.py --variant B0cmp --checkpoint checkpoints/abl_B0_s42.pt
echo "[gen] CGA fused (abl_CGA_s42, --use-cga)"
$PY gen_cga_fused.py --variant CGA --checkpoint checkpoints/abl_CGA_s42.pt --use-cga
echo "[detect] B0cmp + CGA (old venv)"
$OLDPY arb_detect.py B0cmp CGA
echo "[compare] CGA vs B0cmp"
$PY cga_compare.py CGA B0cmp
echo "CGA EVAL DONE"
