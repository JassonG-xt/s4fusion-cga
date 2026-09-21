# ENVIRONMENT — what is pinned, what is inherited, and why it matters

The single most important reproducibility fact about this release is that the
**fusion side is pinned and the detector side is not**. That asymmetry is what
bounds the reproducibility claim in the manuscript and it is stated here in
full.

## 1. Fusion side — pinned and shipped

`code/requirements-brss.txt`:

```
torch==2.1.0+cu118
torchvision==0.16.0+cu118
timm==0.9.12
einops==0.7.0
numpy>=1.24,<2.0
pillow>=10
scipy>=1.10
scikit-image>=0.21
pytest>=8,<9
ninja
transformers==4.37.2
causal-conv1d==1.1.3.post1
mamba-ssm==1.1.1
```

- Python 3.11, Linux under WSL2, single consumer GPU with 4 GB VRAM.
- The exact hardware and driver for the reported latency numbers are recorded in
  the manuscript (Table 2 and its *Measurement settings* paragraph): one
  NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), driver 592.27, cuDNN 8.7.0
  (`torch.backends.cudnn.version() == 8700`), GPU otherwise idle.
- Metrics: `code/eval_metrics.py` (Y-channel EN / SF / AG / MI / QABF / VIF);
  `code/eval_metrics_extended.py` for the colour audit.

## 2. Detector side — inherited, not pinned

The downstream detection endpoint uses the detection branch checkpoint released
with DCEvo (`DCEvo_detect_branch.pt`) and the official M3FD XML→YOLO ground-truth
construction, at input size 640 with a detection counted at IoU ≥ 0.5.

That stack comes from the DCEvo release, whose **own** `requirements.txt` pins:

```
torch==2.0.0+cuda117
timm==1.0.15
einops==0.8.1
opencv-python==4.11.0.86
scikit-image==0.21.0
seaborn==0.13.2
kornia==0.2.0
pygad==3.4.0
ultralytics==2.0.14
```

**This archive does not pin or ship that stack**, and the exact resolved versions
of the detector environment actually used for the reported runs were not
recorded. Two consequences, both of which the manuscript states rather than
glosses over:

1. The **statistics** in the paper are exact and recomputable from the released
   per-image and per-object intermediate results, because they are computed over
   stored predictions, not over a fresh detector pass.
2. **Re-running the detector is not guaranteed to reproduce those predictions bit
   for bit.** A different `ultralytics` or `torch` build can move a small number
   of detections, and therefore a small number of counts. Treat the detector pass
   as the weakest link in the chain.

## 3. What this means for a reader

| You want to… | Feasible from this archive alone? |
|---|---|
| recompute any statistic in the paper | **yes** — from the released CSVs / per-object vectors |
| reproduce the cluster-aware analysis (the main detection reading) | **yes** — `cluster_audit_final.py` over the released predictions |
| re-run the fusion pipeline end to end | **yes**, with the pinned fusion environment and the source datasets |
| re-run detection and get the same counts exactly | **no guarantee** — detector stack inherited, not pinned |
| regenerate fused images or the qualitative figures | **yes**, with the source datasets (not bundled) |
