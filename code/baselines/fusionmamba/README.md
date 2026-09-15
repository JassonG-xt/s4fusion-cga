# FusionMamba — 基线复现卡片

> 状态：引未比 → 补实验（K2）。Visual Intelligence 2024, 2:37, DOI:10.1007/s44267-024-00072-9（225+ 引）。
> 官方实现：GitHub `GeorgeWangno1/FusionMamba`（作者 README 提供推理命令与权重）。

## 复现步骤（按官方 README 执行）

```bash
cd baselines_2025
git clone https://github.com/GeorgeWangno1/FusionMamba.git FusionMamba
cd FusionMamba
# 下载官方预训练权重（README 指引）
# 推理（官方命令示例）：
python test_fusion_IVIF.py \
  --ir_folder  ../../code/baselines/inputs/m3fd/ir \
  --vi_folder  ../../code/baselines/inputs/m3fd/vi \
  --out_folder ../../code/baselines/outputs/fusionmamba/m3fd
```

## 注意

- Mamba 融合类，输出 RGB；统一协议 `--use-y` 口径。
- 与 BRSS 同属 SSM 家族 —— 相关工作 §4（Mamba/SSM）中的直接对照对象。
- 验收：`run_unified_eval.sh --methods fusionmamba` 产出 `results/baselines/fusionmamba_m3fd.csv`。
