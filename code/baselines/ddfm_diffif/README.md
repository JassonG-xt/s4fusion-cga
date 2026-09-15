# DDFM / Diff-IF — 扩散基线复现卡片（二选一）

> 状态：新增候选。主表 8 个中扩散类占 1 席。
> DDFM: ICCV 2023, DOI:10.1109/iccv51070.2023.00742（358 引）—— 官方实现 GitHub `Zhaozixiang1228/IVIF-DDFM`。
> Diff-IF: Information Fusion 2024, DOI:10.1016/j.inffus.2024.102450（186 引）—— 官方实现见论文页链接。

## 选择建议

- **首选 DDFM**：官方代码/权重完整、与 CDDFuse 同作者、推理脚本成熟，社区复现多。
- Diff-IF 为备选（若 DDFM 权重下载受阻）。

## DDFM 复现步骤（按官方 README 执行）

```bash
cd baselines_2025
git clone https://github.com/Zhaozixiang1228/IVIF-DDFM.git IVIF-DDFM
cd IVIF-DDFM
# 下载官方预训练权重（README 指引）
# 推理（官方命令示例）：
python test_IVIF_ddfm.py \
  --IR_dir  ../../code/baselines/inputs/m3fd/ir \
  --VI_dir  ../../code/baselines/inputs/m3fd/vi \
  --Fused_dir ../../code/baselines/outputs/ddfm/m3fd
```

## 注意

- 扩散模型采样耗时显著（每图多步），300 对 M3FD 预计耗时较长 —— 可先 50 对冒烟验证再全量。
- 输出 RGB；统一协议 `--use-y` 口径。
- 验收：`run_unified_eval.sh --methods ddfm` 产出 `results/baselines/ddfm_m3fd.csv`。
