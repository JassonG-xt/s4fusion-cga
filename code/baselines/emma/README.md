# EMMA — 基线复现卡片

> 状态：引未比 → 补实验（K2）。CVPR 2024（首投版已引为[20]，保留）。
> 官方实现：GitHub `LauZ1234/EMMA`（作者 README 提供推理命令与权重）。

## 复现步骤（按官方 README 执行）

```bash
cd baselines_2025
git clone https://github.com/LauZ1234/EMMA.git EMMA
cd EMMA
# 下载官方预训练权重（README 指引）至 model/ 或 ckpt/
# 推理（官方命令示例）：
python test.py --dataset M3FD \
  --ir_path  ../../code/baselines/inputs/m3fd/ir \
  --vi_path  ../../code/baselines/inputs/m3fd/vi \
  --save_dir ../../code/baselines/outputs/emma/m3fd
```

## 注意

- EMMA 为等变自监督 + 可解释成像先验，输出 RGB。
- 若官方脚本以数据集目录结构（train/test 子目录）输入，用 `inputs/m3fd/` 的直接布局或按仓库要求组织。
- 验收：`run_unified_eval.sh --methods emma` 产出 `results/baselines/emma_m3fd.csv`。
