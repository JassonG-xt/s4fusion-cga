# CDDFuse — 基线复现卡片

> 状态：引未比 → 补实验（K2）。DOI:10.1109/cvpr52729.2023.00572（CVPR 2023）。
> 官方实现：GitHub `Zhaozixiang1228/CDDFuse`（作者 README 提供推理命令）。

## 复现步骤（按官方 README 执行，以下为模板）

```bash
cd baselines_2025
git clone https://github.com/Zhaozixiang1228/CDDFuse.git CDDFuse
cd CDDFuse
# 下载官方预训练权重（README 指引）至 ./model/
# 推理（官方命令示例，以仓库 README 为准）：
python test_demo.py 或 test_IVIF.py \
  --IR_dir  ../../code/baselines/inputs/m3fd/ir \
  --VI_dir  ../../code/baselines/inputs/m3fd/vi \
  --Fused_dir ../../code/baselines/outputs/cddfuse/m3fd
```

## 注意

- CDDFuse 为 Transformer/相关分解类，输出为 RGB 全彩（保留可见光颜色）→ 统一协议 `--use-y` 取 Y 通道评测，与其余基线口径一致。
- 若仓库测试脚本按数据集目录整批处理（`--DataFolder` 型），先把 `inputs/m3fd/` 直接作为其输入目录。
- 验收：`run_unified_eval.sh --methods cddfuse` 产出 `results/baselines/cddfuse_m3fd.csv`。
