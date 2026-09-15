#!/bin/bash
# A2 fix: evaluate the OFFICIAL S4Fusion weights (model.pkl) on the full 300-image
# M3FD manifest with the unified tiled protocol, so the main table has a proper
# official-weights baseline row distinct from the same-budget retrain (B0cmp_full).
# Provenance note (2026-09-09): the existing results/baselines/s4fusion_m3fd.csv
# was generated (2026-08-19) from the B0-retrain images, NOT the official weights —
# pixel probe confirms the two models differ (mean abs diff 6.3-8.4). This script
# writes to s4fusion_official_m3fd.csv and leaves the mislabeled file untouched.
set -e
cd "$(dirname "$0")"
.venv-brss/bin/python evaluate_brss.py --data-root ../dataset \
  --manifest ../dataset/manifests/m3fd_test.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --boundary-mode none --fusion-scales none --gate-type scalar \
  --output-dir results/baseline_official \
  --metrics-csv results/baselines/s4fusion_official_m3fd.csv \
  --tile-size 507 --tile-overlap 64 --amp --no-save-rgb \
  >> logs/s4fusion_official_eval.log 2>&1
echo "[a2-fix] official-weights eval done $(date)"
