# BRSS-Fusion implementation

This directory is an isolated implementation workspace for the rejected-paper
rebuild. The first-submission PDF and reviewer comments remain outside `code/`
and are not modified.

## Implemented

- Boundary Reliability Estimator (BRE) with IR, visible and shared weights.
- Reliability-aware Patch Mark and modulation of Cross-SS2D `B`, `C`, and
  `Delta` terms.
- Spatial boundary residual recovery with zero-initialized compatibility.
- MS-Global context as an auxiliary input to BRE.
- Boundary, reliability, cross-resolution and structural-distillation losses.
- Full fine-tuning and teacher/student training entry point for a 4 GB GPU.

## Baseline compatibility

`boundary_mode=none` reproduces the previous architecture. `boundary_mode=full`
adds BRSS parameters. Boundary state and residual scales are initialized to
zero, so loading an existing S4Fusion checkpoint with `strict=False` starts from
the baseline behavior.

## 4 GB training command

Create a clean environment first. Do not reuse the old `.venv`, because it
contains the obsolete PyPI `argparse` package, which shadows Python's standard
library and breaks modern pytest.

```bash
/usr/bin/python3.11 -m venv .venv-brss
.venv-brss/bin/python -m pip install --upgrade pip
.venv-brss/bin/python -m pip install -r requirements-brss.txt
.venv-brss/bin/python -m pip uninstall -y argparse
.venv-brss/bin/python -m pytest tests/test_brss.py -q
```

The downloaded and audited manifests can be used directly:

```bash
cd S4Fusion-main-Innovation-2026/code
.venv-brss/bin/python train_brss.py \
  --data-root ../dataset \
  --train-manifest ../dataset/manifests/train_all.csv \
  --val-manifest ../dataset/manifests/val_all.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --samples-per-epoch 2048 --epochs 20 \
  --boundary-mode full --fusion-scales 4,8 --gate-type dynamic --amp
```

On the 4 GB laptop GPU, `--samples-per-epoch 2048` defines an epoch as 2,048
balanced sampled crops instead of all 15,079 training pairs. Report both the
sample count and resulting optimizer-update count in the manuscript. The
default value `0` still traverses the complete training manifest.

Every epoch writes both the best checkpoint and a resumable `_last` checkpoint.
Resume an interrupted run with `--resume checkpoints/brss_fusion_last.pt` while
keeping the same model and sampling arguments.

```bash
.venv-brss/bin/python train_brss.py \
  --train-split /path/to/train.txt \
  --val-split /path/to/val.txt \
  --m3fd-zip /path/to/archive.zip \
  --roadscene-zip /path/to/RoadScene-master.zip \
  --checkpoint /path/to/model.pkl \
  --crop-size 128 --batch-size 1 --accumulation-steps 8 \
  --boundary-mode full --fusion-scales 4,8 --amp
```

Train the full teacher first. To train a smaller student, set a smaller
`--depths` value and provide `--teacher-checkpoint` plus the teacher's
`--teacher-depths`.

For a one-batch end-to-end CLI check, append
`--epochs 1 --max-train-steps 1 --max-val-steps 1`. These limits are only for
debugging and must not be used for reported experiments.

## Verified locally

- The full local suite reports 9 passed and 3 optional legacy-audit tests
  skipped because their 2025 report generator is not copied into this folder.
- LLVIP, M3FD, FMB and MSRS samples load as paired one-channel crops.
- The original checkpoint loads with no unexpected keys; the missing keys are
  the newly introduced global-gate and BRSS parameters.
- A full 123x123 train step with cross-resolution losses completed on the 4 GB
  RTX 3050, with about 530 MiB peak reserved CUDA memory in the smoke process.
- Baseline, previous MS-Global and a BRSS pilot checkpoint all completed the
  same 507x507 overlap-tiled M3FD inference/metric pipeline.

## Required experiment order

1. S4Fusion baseline and previous MS-Global checkpoint.
2. `boundary_mode=residual`, `state`, and `full` ablation.
3. Full BRSS teacher with three random seeds.
4. High-resolution and SOTA evaluation under one shared protocol.
5. Distilled student efficiency experiment.

Do not claim quantization or high-resolution superiority until the corresponding
native-resolution datasets and hardware measurements have been completed.

## Unified manifest evaluation

Original S4Fusion baseline:

```bash
.venv-brss/bin/python evaluate_brss.py \
  --data-root ../dataset \
  --manifest ../dataset/manifests/m3fd_test.csv \
  --checkpoint ../../S4Fusion-main/model/model.pkl \
  --output-dir results/baseline/m3fd \
  --metrics-csv results/baseline/m3fd.csv \
  --fusion-scales none --gate-type scalar --boundary-mode none \
  --tile-size 507 --tile-overlap 64 --amp --save-rgb
```

Previous MS-Global checkpoint:

```bash
.venv-brss/bin/python evaluate_brss.py \
  --data-root ../dataset \
  --manifest ../dataset/manifests/m3fd_test.csv \
  --checkpoint ../../S4Fusion-main-Innovation/model/model_ms_fusion_only_w1e-3_e10.pkl \
  --output-dir results/ms_global/m3fd \
  --metrics-csv results/ms_global/m3fd.csv \
  --fusion-scales 4,8 --gate-type scalar --boundary-mode none \
  --tile-size 507 --tile-overlap 64 --amp --save-rgb
```

For a trained BRSS checkpoint, use `--fusion-scales 4,8 --gate-type dynamic
--boundary-mode full`. Use the same tile size and overlap for every compared
method. The evaluator records the inference mode in every CSV row. Native
padded inference is available with `--tile-size 0`, but it should only be used
when every compared model fits at full resolution.
