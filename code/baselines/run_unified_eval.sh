#!/usr/bin/env bash
# Unified baseline pipeline: prepare inputs -> (user runs official inference) -> align -> metrics.
# GPU required for inference steps; metrics can run on CPU/WSL.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-../../code/.venv-brss/bin/python}

DATASETS="${DATASETS:-m3fd}"
METHODS="${METHODS:-meta_fusion,cddfuse,emma,fusionmamba,dcevo,wmamba}"

echo "[1/3] prepare inputs"
$PY prepare_baseline_inputs.py --datasets "$DATASETS" --manifest-dir ../dataset/manifests --out-root ./inputs

echo "[2/3] align + unified metrics"
$PY collect_outputs.py --methods "$METHODS" --datasets "$DATASETS" \
  --inputs ./inputs --baseline-out ./outputs --metrics-out ../results/baselines

echo "[3/3] done -> ../results/baselines/<method>_<dataset>.csv (rows per image + __mean__/__std__)"