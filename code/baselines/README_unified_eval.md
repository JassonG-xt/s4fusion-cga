# 统一基线评测协议（K2, ≥8 基线同口径）

> 目标：闸门-1 硬门槛 —— 所有基线用**同一评测器、同一测试集、同一指标集**，输出可直接入论文主表。
> 原则：推理用各基线**官方实现与官方权重**（复现说明见各基线目录），指标用本项目统一评测器，禁止混合口径。
> 状态：协议与脚本就绪；**需在 GPU 环境执行**（WSL 无法访问 Windows GPU）。执行后贴回真实 stdout 才算闭环。

---

## 0. 目录约定

```
code/baselines/
├── README_unified_eval.md        # 本文件
├── prepare_baseline_inputs.py    # manifest → 各基线输入目录（同命名、同分辨率）
├── run_unified_eval.sh           # 一键：推理(可选) + 统一指标 → 主表 CSV
├── collect_outputs.py            # 各基线输出 → 统一 fused/ 目录（+ 分辨率对齐）
├── meta_fusion/                  # 优先补齐的基线：复现说明
│   └── README.md
├── cddfuse/                      # README（官方仓库 + 复现要点）
├── emma/                         # README
├── fusionmamba/                  # README
├── ddfm_diffif/                  # README（DDFM / Diff-IF 二选一）
└── ../../baselines_2025/         # 已比基线：DCEvo、W-Mamba（权重已在本机）
```

## 1. 基线清单（主表 8 个）

| # | 方法 | 类别 | 权重来源 | 推理入口 | 状态 |
|---|------|------|----------|----------|------|
| 1 | **S4Fusion** | SSM（主干） | `S4Fusion-main/model/model.pkl` | `code/evaluate_brss.py --boundary-mode none` | 已比（BRSS 主干对照） |
| 2 | **MetaFusion** | 任务驱动/元学习 | 官方仓库 `wdzhao123/MetaFusion`（pytorch1.8.1） | `test.py --test_ir_root ... --test_vis_root ... --save_path ...` | **必补（优先）** |
| 3 | CDDFuse | Transformer/相关分解 | 官方仓库（CVPR2023） | 官方 test.py | 引未比 → 补 |
| 4 | EMMA | 等变自监督 | 官方仓库（CVPR2024） | 官方 test.py | 引未比 → 补 |
| 5 | FusionMamba | Mamba 融合 | 官方仓库（Visual Intelligence 2024） | 官方 test.py | 引未比 → 补 |
| 6 | DCEvo | 进化学习 | `baselines_2025/DCEvo/ckpt/DCEvo_fusion.pth` | `test_Fusion.py`（注意强制 resize，见 §4） | 已比 |
| 7 | W-Mamba | 小波域 SSM | `baselines_2025/W-Mamba/Model/Infrared_Visible_Fusion/models/best_WMamba.pth` | `Test.py`（crop128 重叠重建） | 已比 |
| 8 | DDFM 或 Diff-IF | 扩散 | 官方仓库 | 官方 test.py | 新增（二选一，建议 Diff-IF 若资源紧） |

> 选 8 的原因：S4Fusion（主干）+ 4 个审稿人点名（MetaFusion/CDDFuse/EMMA/FusionMamba）+ 已比 2 个（DCEvo/W-Mamba）+ 扩散 1 个。五类全覆盖（CNN 由 DCEvo 官方对比表兼引，主表用 SSM/Transformer/Diffusion/任务驱动 4+2+1+1）。

## 2. 统一评测器（同一口径的核心）

- **入口**：`code/eval_metrics.py`（无需 BRSS 模型，纯图像指标）
- **参数固定**：
  ```bash
  .venv-brss/bin/python eval_metrics.py \
    --ir-path   <ir_dir>     --vi-path <vi_dir> --fused-path <fused_dir> \
    --use-y --out-csv <method>_<dataset>.csv \
    --skip-topiq --skip-musiq        # 感知指标单独跑（pyiqa 慢），主表先出结构+MI+VIF
  ```
- **指标集**：EN / SF / AG / MI / QABF / VIF（`--paper-only` 出 SF/AG/VIF/QABF/TOPIQ/MUSIQ 时用完整模式）
- **逐图输出**：CSV 含每图行 + `__mean__`/`__std__` 行 → Wilcoxon 成对检验可直接复用
- **分辨率对齐**：所有 fused 输出必须与 IR 输入**同分辨率**；官方输出不一致的，`collect_outputs.py` 双线性 resize 回 IR 尺寸并在复现说明中注明（诚实登记偏差，不藏）。

## 3. 测试集（同一 manifest）

| 用途 | 数据集 | manifest | 图对 | 说明 |
|---|---|---|---|---|
| 主表 | M3FD test | `manifests/m3fd_test.csv` | 300 | 与 BRSS H1 评测同集 |
| 泛化/OOD | TNO | `manifests/tno_test.csv` | 40+ | OOD |
| 高分辨率压力 | LLVIP | `manifests/llvip_test.csv`（`test_high_resolution.csv`） | — | HR 轴 |

> 脚本默认 `--datasets m3fd`；扩展时 `--datasets m3fd,tno,llvip`。

## 4. 已知基线口径偏差（如实登记）

| 基线 | 偏差 | 处理 |
|---|---|---|
| DCEvo `test_Fusion.py` | 强制 `Resize((768,1024))` | 输出 resize 回 IR 原分辨率再评测；README 注明 |
| W-Mamba `Test.py` | `--in_channel 1` + YCbCr 颜色恢复 | 用官方默认（Y 通道融合 + UV 恢复）；注明 |
| MetaFusion | 训练/测试 resize 512×384（官方 README） | 用官方权重直推，输出对齐回 IR 分辨率 |

> 所有"对齐回原分辨率"仅用于指标可比性，不作为方法本身的分辨率能力声明。

## 5. 一键执行

```bash
cd code/baselines
# 1) 生成各基线输入（从 manifest 取 IR/VI 对，命名 sample_id）
.venv-brss/bin/python prepare_baseline_inputs.py --datasets m3fd \
  --manifest-dir ../dataset/manifests --out-root ./inputs

# 2) 在各基线目录跑官方推理（GPU 环境），输出统一命名 sample_id.png
#    MetaFusion:  cd meta_fusion && python test.py --test_ir_root ../../inputs/m3fd/ir \
#                 --test_vis_root ../../inputs/m3fd/vi --save_path ../outputs/meta_fusion/m3fd
#    DCEvo:       bash ../../baselines_2025/DCEvo/run_dcevo_m3fd.sh
#    W-Mamba:     bash ../../baselines_2025/W-Mamba/run_wmamba_m3fd.sh

# 3) 收集输出（分辨率对齐）并统一出指标
bash run_unified_eval.sh --datasets m3fd --methods meta_fusion,cddfuse,emma,fusionmamba,dcevo,wmamba
```

## 6. 闭环红线

- 每个基线：官方权重 + 官方推理 + 贴回真实 stdout（指标行）才算完成。
- 禁止以"预期数值"填入主表。本目录内任何未跑出的数字一律标 `TBD`。
- 跑完后把 CSV 提交到 `code/results/baselines/`（git 入库，与 BRSS 结果并列）。
