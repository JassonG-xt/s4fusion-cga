#!/usr/bin/env bash
# Three-seed colour-fidelity audit + three-seed selector statistics.
# Inference/CPU only: no training, no detector inference.
set -u
cd /mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code
PY=./.venv-brss/bin/python
VI=baselines/inputs/m3fd/vi
IR=baselines/inputs/m3fd/ir
mkdir -p results_metrics

echo "### regression: reproduce the frozen seed-42 colour table ###"
$PY rebuild_rgb.py --vi-dir "$VI" --fused-dir results/arb/CGA_str/images --out-dir results/arb/CGA_str/images_rgb
$PY eval_metrics_extended.py --ir-path "$IR" --vi-path "$VI" \
  --fused-path results/arb/CGA_str/images --fused-rgb-path results/arb/CGA_str/images_rgb \
  --use-y --out-csv results_metrics/ext_CGA_str.csv
$PY - <<'EOF'
import csv
a={r['name']:r for r in csv.DictReader(open('results/arb/CGA_str/color_metrics.csv'))}
b={r['name']:r for r in csv.DictReader(open('results_metrics/ext_CGA_str.csv'))}
ks=[k for k in ('SF','AG','EOR','ECS','OBR','RGB_SSIM','CIEDE2000','Colorfulness') if k in a['00000.png']]
worst=0.0
for n in a:
    if n.startswith('__') or n not in b: continue
    for k in ks:
        worst=max(worst,abs(float(a[n][k])-float(b[n][k])))
print('seed-42 regression max |diff| over', len(a)-2, 'images x', len(ks), 'metrics =', worst)
EOF

for S in 42 123 3407; do
  T=CGA_str; [ "$S" = "42" ] || T=CGA_str_s$S
  B=B0_e32_s$S
  echo "### seed $S : $T vs $B ###"
  for C in "$T" "$B"; do
    $PY rebuild_rgb.py --vi-dir "$VI" --fused-dir results/arb/$C/images --out-dir results/arb/$C/images_rgb
    $PY eval_metrics_extended.py --ir-path "$IR" --vi-path "$VI" \
      --fused-path results/arb/$C/images --fused-rgb-path results/arb/$C/images_rgb \
      --use-y --out-csv results_metrics/ext_$C.csv
  done
  $PY eval_metrics_extended.py --baseline-csv results_metrics/ext_$B.csv \
    --candidate-csv results_metrics/ext_$T.csv \
    --stats-out results_metrics/color_pair_$T.csv --bootstrap 1000
done

echo "### three-seed selector statistics ###"
for S in 123 3407; do
  $PY export_select_stats.py --checkpoint checkpoints/abl_CGA_str_s$S.pt --tag s$S
done

echo "### DONE ###"
