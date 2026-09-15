# BRSS-Fusion implementation status

## Completed in code

- Isolated 2026 code workspace created without modifying the first-submission files.
- Boundary Reliability Estimator with IR/VI/shared spatial probabilities.
- Reliability-aware modulation of Cross-SS2D B, C and Delta terms.
- Reliability-aware Patch Mark and spatial boundary residual path.
- Zero-initialized compatibility with existing S4Fusion checkpoints.
- MS-Global coarse context supplied to the boundary estimator.
- Boundary, reliability, resolution-consistency and distillation objectives.
- 4 GB GPU training entry point with AMP and gradient accumulation.
- Fixed-size balanced epoch sampling plus model/optimizer/scaler resume support.
- Audited manifest loader and inverse-frequency balanced multi-dataset sampling.
- Unit tests for normalization, gradients, checkpoint compatibility, baseline mode and distillation.
- Mechanism-ablation evaluation hooks: `--global-op` (H2 external-op control),
  `--reliability-override {uniform,shuffle}` (H3 falsification) and
  `--dump-boundary` (F1-F5 state-response maps + learned gate scales), with
  regression tests (`tests/test_brss.py`, 8 passed).
- Clean Python 3.11/CUDA 11.8 environment under `code/.venv-brss`.
- Real RTX 3050 forward/backward smoke test with the original `model.pkl`.
- Unified manifest evaluator with native padding, overlap-tiled inference,
  grayscale/RGB output and EN/SF/AG/MI/QABF metrics.
- Full local test result: 9 passed, 3 optional legacy-audit checks skipped.
- Reviewer-comment remediation matrix.

## Completed data preparation

- LLVIP, M3FD, FMB and MSRS downloaded, validated and split without altering official test sets.
- TNO and RoadScene retained as out-of-distribution test sets only.
- Combined manifests contain 15,079 train, 3,149 validation and 4,489 test pairs.
- Zero corrupt pairs, unmatched pairs or train/validation/test overlap were found.

## Requires experiment execution

> **更新（2026-09-13）**：以下各项已由闸门-1 冻结收尾，本文件转为历史文档；当前状态与唯一数据源以 `GATE1_FREEZE.md` 为准。

- ~~Train BRSS teacher for seeds 42, 123 and 3407.~~ **已完成**：三种子 CGA_str + B0 全部训练并评测（`checkpoints/abl_*`、`code/results/seeds/seeds_table.md`）。
- ~~Run residual/state/full mechanism ablations.~~ **已完成并证伪**：H1 状态调制不改善结构指标（`logs/eval_H1.log`，冻结于 GATE1 §3）。
- ~~Reproduce the agreed SOTA methods under the unified evaluator.~~ **已完成**：8 基线统一协议主表含 FusionMamba（`code/results/baselines/SOTA_MAIN_TABLE.md`，2026-09-13）。
- ~~Train the compact distilled student and measure runtime.~~ **已运行且为负结果**：B2 蒸馏 val 全 NaN、best 未落盘（`logs/distill.log`、`B1_B2_EVIDENCE_20260912.md` §4），按预注册不调参不复跑，叙事降级为 limitation。
- ~~Generate RGB results and the final statistical tables.~~ **已完成**：images_rgb 300 张/组、种子表、颜色指标 CSV 均已产出并冻结（GATE1 §2/§5）。
- ~~Rewrite the English manuscript after results are frozen.~~ **进行中（C-10，2026-09-13 启动）**：writing/ 目录按闸门-2 推进；目标期刊已变更为免版面费三四区序列（首选 The Visual Computer，见 `期刊初选备忘-20260913.md` 修订节）。

These items cannot be truthfully marked complete before datasets, checkpoints,
training runs and measurements exist. The implementation now provides the code
path needed to produce them.
