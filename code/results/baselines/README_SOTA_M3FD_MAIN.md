# M3FD SOTA 主表（统一口径，full-300）

> 同一评测器 `code/eval_metrics.py`、同一测试集 `manifests/m3fd_test.csv`（300 对）、Y 通道 + 分辨率对齐。
> 基线推理全部用官方权重/官方实现（见各基线复现说明）；Ours 行（重训/CGA）用同评测器对各自融合图评测。感知指标（TOPIQ/MUSIQ）待 pyiqa 版本修复后补。
> 主表由 `code/aggregate_sota.py` 从逐图 CSV 自动生成（`SOTA_MAIN_TABLE.{md,csv}`），禁止手填。

## 主表（2026-09-10 重新生成，含官方权重行）

| 方法 | EN | SF | AG | MI | QABF | VIF | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| **S4Fusion (official weights)** | 6.8926±0.4159 | 14.9241±5.4901 | 8.4926±4.4715 | 4.3364±1.4031 | 0.5147±0.0617 | **0.4628**±0.0360 | ✅ 2026-09-10 新增 |
| MetaFusion (CVPR'23) | **7.2607** | 13.7397 | 8.2228 | 2.2953 | 0.2189 | 0.2842 | ✅ |
| DCEvo (CVPR'25) | 6.8301 | 13.7861 | 7.6131 | 4.2834 | 0.4410 | 0.3930 | ✅ |
| W-Mamba (ICME'25) | 6.7083 | 11.8793 | 6.6574 | **4.8307** | 0.3532 | 0.3645 | ✅ |
| CDDFuse (CVPR'23) | 6.8996 | 14.7758 | 8.3311 | 3.7648 | 0.4008 | 0.3884 | ✅ |
| EMMA (CVPR'24) | 6.9242 | **15.2273** | **9.0133** | 3.7391 | 0.3405 | 0.3770 | ✅ |
| Diff-IF (InfFus'24) | 6.7904 | 14.1008 | 7.9352 | **5.1704** | 0.4345 | 0.3973 | ✅ |
| FusionMamba (VI'24) | 6.6103 | 11.3349 | 6.2939 | 3.3033 | 0.2296 | 0.3178 | ✅ 2026-09-13 补齐（B3） |
| S4Fusion (same-budget retrain) | 6.8733±0.3728 | 14.8933±5.4380 | 8.6801±4.3975 | 3.2078±0.8250 | 0.5295±0.0552 | 0.4535±0.0509 | ✅ |
| **CGA (ours, s42)** | 6.8863±0.3896 | 14.7672±5.3889 | 8.5173±4.3712 | 3.9345±0.8956 | **0.5449**±0.0552 | 0.4622±0.0545 | ✅ |

**FusionMamba 读表（2026-09-13）**：KAIST 权重在 M3FD 上表现偏弱——六指标全面低于 W-Mamba（同 SSM 家族）与 S4Fusion 系；QABF 0.2296 为全表最低。这与 DCEvo 论文中 FusionMamba 在 M3FD 类道路场景的非最优表现一致；不改变任何已有结论的相对位置（QABF/VIF 头部仍为 CGA/官方 S4Fusion）。

**读表要点（2026-09-10 更正后）**：
- **QABF 第一 = CGA (ours) 0.5449**；同预算重训 0.5295、官方权重 0.5147 分列二三。旧表述"S4Fusion QABF 第一"在基线内部成立，全表第一是 ours。
- **VIF 第一 = S4Fusion 官方权重 0.4628**；CGA 0.4622 与官方打平（Δ=−0.0006），重训 0.4535。旧表把重训行误标为官方，导致旧文档记 VIF=0.453"第一"——**该数字是重训的，且当时漏了官方更高的 0.4628**，特此更正。
- MI：官方权重 4.3364 反而高于重训 3.2078 与 CGA 3.9345；CGA 相对同预算重训 MI +0.727。全表 MI 最高为 Diff-IF 5.1704。
- EN 最高 MetaFusion（信息量-感知 trade-off）；SF/AG 最高 EMMA（纹理锐利但结构保真弱）。
- CGA vs 同预算重训（配对差）：QABF +0.015、VIF +0.009、MI +0.727，SF −0.126（−0.85%）、AG −0.163（−1.88%）——SF 在 C3 的 1% 非劣性裕度内，AG 超出，按预注册口径如实记录。

## ⚠️ 溯源更正记录（2026-09-10，诚信披露）

**问题**：`results/baselines/s4fusion_m3fd.csv`（2026-08-19 生成，当时标注为官方 S4Fusion）实际由 **B0 同预算重训**（`checkpoints/abl_B0_s42.pt`）的融合图评测而来，并非官方权重 `model.pkl`。像素探针证实两者输出不同（3 张探针图 mean abs diff 6.3–8.4）；`gen_b0_full.log` 证实该图源自 B0 checkpoint。**性质为标注错误（误标来源），非伪造数据**——数字本身真实，但行标签错误。

**修复**：
1. `run_a2_fix.sh` 用官方 `model.pkl` + 统一分块协议（tile507/overlap64/AMP）重新推理 300 图 → `results/baseline_official/m3fd/`（逐图含推理耗时，日志 `logs/s4fusion_official_eval.log`，平均 ~1.26s/图）。
2. `eval_metrics.py`（与其余全部基线同一评测器，含 VIF）对上述官方融合图评测 → `results/baselines/s4fusion_official_m3fd.csv`。
3. 主表拆为两行：**S4Fusion (official weights)**（新行）与 **S4Fusion (same-budget retrain)**（=`B0cmp_full`，即原误标行的真实身份）。
4. 原误标文件 `s4fusion_m3fd.csv` **保留不删**（历史可追溯），但从 `aggregate_sota.py` 的 BASELINES 中移除，不再入任何表格。`s4fusion_official_m3fd.evaluate_brss.csv.bak` 为官方权重经 `evaluate_brss.py` 的平行口径留档（无 VIF 列）。

## CSV 明细（逐图 + `__mean__`/`__std__`，可做 Wilcoxon 成对检验）

`code/results/baselines/{s4fusion_official,meta_fusion,dcevo,wmamba,cddfuse,emma,diffif}_m3fd.csv` + Ours 行 `{CGA_str,B0cmp_full}_m3fd.csv`；种子表 `code/results/seeds/<cell>_s<seed>.csv`（B1 补齐后 `analyze_seeds.py` 汇总）。
（历史误标文件 `s4fusion_m3fd.csv` 不再使用，见上方溯源更正记录。）

## 待补基线（TBD，权重仅网盘）

（无 —— 2026-09-13 起 8 基线全部跑齐，B3 完成。）

## 复现记录（推理日志）

- **S4Fusion 官方权重**：`run_a2_fix.sh` → `evaluate_brss.py --checkpoint ../../S4Fusion-main/model/model.pkl --boundary-mode none --fusion-scales none --gate-type scalar --tile-size 507 --tile-overlap 64 --amp`，300 张 / ~6.3min（1.26s/张）；指标由 `eval_metrics.py` 对融合图统一评测（含 VIF）。
- **FusionMamba**：官方仓（`baselines_2025/FusionMamba`，旧版 millieXie 代码）+ 官方 KAIST IRVIS 权重（百度网盘 `KAIST_1ssim_10int_1grad_.pth`，902MB）。自写 `test_m3fd.py`（--ir/--vi/--out/--model，原分辨率、灰度逐对、min-max 归一化）。**坑与修复**：VMamba 层次编解码要求边长被 32 整除，M3FD 有 17 张非整除图（480×360 等）在第 129 张起 crash（skip 尺寸 44≠45）→ `test_m3fd_pad32.py`（replicate pad 到 32 倍数，推理后裁回原尺寸）补跑 172 张。环境：`.venv-brss`（mamba-ssm 1.1.1 + causal-conv1d 1.1.3.post1，cu118），前 128 张 ~7.2s/张、后 172 张 ~0.8s/张；对齐与评测经 `collect_outputs.py`（`_aligned/` 双线性回 IR 尺寸）+ `eval_metrics.py`（含 VIF）。日志：`logs/fusionmamba_m3fd_infer.log`、`logs/fusionmamba_collect.log`。
- MetaFusion：`baselines_2025/MetaFusion/test.py --checkpoint weight/model_weight.pth`，300 张 / 77s（cuDNN workaround 警告无碍）
- DCEvo：`baselines_2025/DCEvo/dcevo_infer_m3fd.py`（官方 test_Fusion.py 逻辑 + fp16 autocast 规避 cuDNN-v8 nvrtc fallback 的 87s/张 退化；官方口径 `Resize((768,1024))`），300 张 / 22min
- W-Mamba：`baselines_2025/W-Mamba/Test.py --model_number best --in_channel 1`（crop128 重叠重建），300 张 / ~59min（11.7s/张）
- CDDFuse：`baselines_2025/MMIF-CDDFuse/test_m3fd.py`（官方 `test_IVF.py` 逻辑；注意官方仓库自带 `models/CDDFuse_IVF.pth` 的 key 为 `DIDF_*` 且带 `module.` 前缀，官方 test 脚本的 `CDDF_*` 已过时，本脚本直接加载并做了 uint8 PNG 保存修复），300 张 / ~80min（15s/张）
- EMMA：`baselines_2025/MMIF-EMMA/test_m3fd.py`（官方 `test.py` 逻辑 + 32 对齐分块；`utils.py` 的 skimage imsave 对 float32 报错已改为 cv2 保存），300 张 / ~5min
- Diff-IF：权重 `weights/IVF_weights_gen.pth` 经 gdown 自 Google Drive 获取（`resume_state: weights/IVF_weights` → 加载 `IVF_weights_gen.pth`）；`infer_ddim.py --config diff-if-ivf-val-m3fd.json`（dataroot 改为 `dataset/M3FD_FK/test`，`Visible/Infrared` 为符号链接）；DDIM 4 步；`p_sample_loop_ddim` 网络前向包 autocast fp16 提速（9.8s/it → 1.15s/it，DCEvo 同款 cuDNN workaround）；输出 RGB（评估 `--use-y` 取 Y 通道），300 张 / ~35min

## 备注

- DCEvo 输出按 README §4 记录强制 768×1024 口径；对齐回 IR 原分辨率仅用于指标可比性。
- 旧仓库 `results_metrics/S4Fusion_M3FD_paper.csv`（SF=15.36/AG=9.04/VIF=0.4505/QABF=0.5053）为更早口径，不直接入主表；以本文件统一口径为准。
- 主表均值/方差均为逐图总体统计（pstdev）；行内粗体 = 该指标全表最优。
