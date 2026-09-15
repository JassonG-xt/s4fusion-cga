#!/usr/bin/env bash
# Re-evaluate the crop-256 (fair-D1) checkpoints: learned-gate magnitude (did
# longer crops engage the mechanism more?), H1 local metrics, and F3 cross-scale
# drift. Run after crop-256 B0+S finish (S log shows epoch=15).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv-brss/bin/python
mkdir -p results/c256
MAN=../dataset/manifests/m3fd_test.csv
[ -f checkpoints/abl_S_c256_s42.pt ] || { echo "ERROR: abl_S_c256_s42.pt missing"; exit 1; }

echo "== learned state-gates: crop256 vs crop128 (mean 0.0139 / max 0.099) =="
$PY - <<'PY'
import torch, numpy as np
st = torch.load("checkpoints/abl_S_c256_s42.pt", map_location="cpu")["model"]
v = [torch.tanh(st[k]).tolist() for k in sorted(st) if k.endswith("boundary_state_scale")]
a = np.array(v)
print(f"crop256 gates: blocks={len(v)} mean|gate|={np.abs(a).mean():.4f} max|gate|={np.abs(a).max():.4f}")
PY

echo "== H1 local metrics (tile 507) =="
$PY evaluate_brss.py --data-root ../dataset --manifest "$MAN" --checkpoint checkpoints/abl_B0_c256_s42.pt \
  --boundary-mode none --fusion-scales none --gate-type scalar \
  --output-dir results/c256/B0_m3fd --metrics-csv results/c256/B0_m3fd.csv --tile-size 507 --tile-overlap 64 --amp --no-save-rgb
$PY evaluate_brss.py --data-root ../dataset --manifest "$MAN" --checkpoint checkpoints/abl_S_c256_s42.pt \
  --boundary-mode state --fusion-scales none --gate-type scalar \
  --output-dir results/c256/S_m3fd --metrics-csv results/c256/S_m3fd.csv --tile-size 507 --tile-overlap 64 --amp --no-save-rgb
$PY analyze_h1.py --b0-csv results/c256/B0_m3fd.csv --s-csv results/c256/S_m3fd.csv \
  --b0-ckpt checkpoints/abl_B0_c256_s42.pt --s-ckpt checkpoints/abl_S_c256_s42.pt

echo "== F3 cross-scale drift =="
$PY run_f3.py --manifest "$MAN" --checkpoint checkpoints/abl_B0_c256_s42.pt --boundary-mode none  --out-csv results/c256/B0_f3.csv --limit 120
$PY run_f3.py --manifest "$MAN" --checkpoint checkpoints/abl_S_c256_s42.pt --boundary-mode state --out-csv results/c256/S_f3.csv --limit 120
$PY analyze_f3.py --b0-csv results/c256/B0_f3.csv --s-csv results/c256/S_f3.csv
