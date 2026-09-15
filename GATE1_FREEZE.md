# 闸门-1 结果冻结清单（GATE-1 FREEZE，2026-09-13）

> 依据 `机制消融实验协议.md` §诚信约束总则与 §闸门-1 读数档位表。本清单落盘即视为**结果冻结**：此后进入写作阶段（LaTeX/英文成稿），任何数字改动须显式解冻记录（`GATE1_UNFREEZE_*.md`）。
> 全部数字来自真实 stdout/CSV/log，出处逐项标注；无手填。
> 冻结时点：B1（四种子）、B2（蒸馏，负结果）、B3（FusionMamba 第 8 基线）、机制可视化（gate_scale/冲突面板）全部收尾；git 提交链 `749bbb0 → 30e90cf → 20997e1 → 5a1bf0c`。

## 1. 主对比表（M3FD full-300，统一评测器 `eval_metrics.py`，Y 通道，8 基线 + S4Fusion 三行体系）

来源：`code/results/baselines/SOTA_MAIN_TABLE.{md,csv}`（`aggregate_sota.py` 自动生成）；逐图 CSV 在 `code/results/baselines/<method>_m3fd.csv`。

| 方法 | EN | SF | AG | MI | QABF | VIF |
|---|---:|---:|---:|---:|---:|---:|
| S4Fusion (official weights) | 6.8926±0.4159 | 14.9241±5.4901 | 8.4926±4.4715 | 4.3364±1.4031 | 0.5147±0.0617 | **0.4628**±0.0360 |
| MetaFusion (CVPR'23) | **7.2607** | 13.7397 | 8.2228 | 2.2953 | 0.2189 | 0.2842 |
| CDDFuse (CVPR'23) | 6.8996 | 14.7758 | 8.3311 | 3.7648 | 0.4008 | 0.3884 |
| EMMA (CVPR'24) | 6.9242 | **15.2273** | **9.0133** | 3.7391 | 0.3405 | 0.3770 |
| DCEvo (CVPR'25) | 6.8301 | 13.7861 | 7.6131 | 4.2834 | 0.4410 | 0.3930 |
| W-Mamba (ICME'25) | 6.7083 | 11.8793 | 6.6574 | **4.8307** | 0.3532 | 0.3645 |
| Diff-IF (InfFus'24) | 6.7904 | 14.1008 | 7.9352 | **5.1704** | 0.4345 | 0.3973 |
| FusionMamba (VI'24) | 6.6103 | 11.3349 | 6.2939 | 3.3033 | 0.2296 | 0.3178 |
| S4Fusion (same-budget retrain, B0) | 6.8733 | 14.8933 | 8.6801 | 3.2078 | 0.5295 | 0.4535 |
| **CGA (ours, s42)** | 6.8863 | 14.7672 | 8.5173 | 3.9345 | **0.5449** | 0.4622 |

冻结要点：
- QABF 全表第一 = CGA 0.5449；VIF 全表第一 = 官方权重 0.4628（CGA 0.4622 与之差 −0.0006，统计上平手）。
- FusionMamba 为全表最弱行（KAIST 权重、道路场景 OOD），如实记录，不改变头部结论。
- 溯源更正已入档：2026-08-19 的 s4fusion_m3fd.csv 实为 B0 重训图像（标注错误，commit 749bbb0 修复为三行体系）。

## 2. 跨种子稳健性（3 种子 ×300 图，逐图配对池化 Wilcoxon + Holm）

来源：`code/results/seeds/seeds_table.md`（`analyze_seeds.py`）；逐图 CSV `results/seeds/{CGA_str,B0}_s{42,123,3407}.csv`。

| 指标 | 池化均值差 (CGA−B0) | p_holm | 方向一致性 |
|---|---:|---:|---|
| EN | +0.0174 | 1.31e-16 | 3+/0− |
| MI | +0.7066 | 1.68e-146 | 3+/0− |
| QABF | +0.0113 | 4.71e-137 | 3+/0− |
| VIF | +0.0073 | 1.26e-97 | 3+/0− |
| SF | −0.0713 | 1.09e-122 | 0+/3− |
| AG | −0.1295 | 4.35e-147 | 0+/3− |

冻结要点：六指标种子方向 100% 一致；种子级检验按预注册省略（n=3 最小 p=0.25）。

## 3. 预注册假设判定

### H1（B/C/Δ 状态调制）——证伪（frozen，2026-08）
来源：`logs/eval_H1.log`；SF −0.0190 / AG −0.0126 / QABF −0.0005，Holm p = 2.90e-44 / 8.13e-40 / 5.63e-27；"S>B0 且 Holm p<0.05 的结构指标：无"。

### H5（CGA 冲突门控仲裁）——三种子全部复现主判据
来源：`logs/after_seeds_v2.log`（[stats] 段）；s42 详情在 `logs/eval_str_cga.log`（2026-08-25）。

| 种子 | 高冲突 recall Δ | recovered/lost | p（精确二项） |
|---|---:|---|---:|
| s42 | +0.0194（0.4385→0.4579） | 20/3 | 4.88e-04 |
| s123 | +0.0159（0.4385→0.4544） | 18/4 | 4.34e-03 |
| s3407 | +0.0159（0.4385→0.4544） | 18/4 | 4.34e-03 |

### 更严预注册门（C1/C2/C3）——窄幅未过 / 口径更新
- gate-C1（高冲突 recall ≥ +0.02）：三种子 FAIL（+0.0194 / +0.0159 / +0.0159）。来源同上。
- gate-C2（mAP50 ≥ +0.01 vs B0cmp 0.7197）：三种子 FAIL（s42 +0.0085→0.7282；s123 +0.0028→0.7225；s3407 +0.0035→0.7232）。来源：`after_seeds_v2.log` full-300 检测行。
- gate-C3（SF/AG 相对非劣性 ≥ −1%，`struct_noninferior.py`）：**s123 PASS**（SF −0.39%，AG −0.60%）；**s3407 PASS**（SF −0.21%，AG −0.36%）；s42 边界例外（SF −0.85% PASS，AG −1.00% 恰在阈值 FAIL）。**冻结口径：C3 于 2/3 种子通过，s42 为 AG 边界例外**（相对 s42 单种子时期的"AG 失败"口径，此为更新，依据 `after_seeds_v2.log` [gate-C3] 两段）。

### 档位判定（协议 §179-183）
H5 主判据满足（三种子显著正增益 + recovered−lost ≥ 3）但三更严门未过 → **介于 A 档与 B 档之间，按 B 偏 A 处理**：主叙事保留（机制显著、三种子稳健），论文措辞固定为 *"statistically significant but below the stricter pre-registered gate thresholds"*，不写"通过预注册门槛"。

## 4. 机制可视化（gate-1 承重图素材）

来源：`code/results/cga_vis/`（`cga_visualize.py`，checkpoint abl_CGA_str_s42）。
- **tanh(gate_scale) = 0.9998**：零初始化全局门控训练至饱和——机制确实被"打开"，排除"门控留在 0、增益另有来源"的自我证伪情形。
- 6 张最高学习冲突样本（01406/01422/01413/01415/01393/01432）：mean|CGA−B0| 0.0125–0.0170、差图峰值至 0.594——改动是**选择性**的（集中在冲突区），非全局扰动。
- npz 面板数据（含手工冲突图 / 学习冲突图 / commit / 差图）随目录存档。

## 5. 次级证据（冻结）

- **F3 跨尺度漂移（D1）**：drift-AUC B0 0.8088 → CGA 0.7839（Δ=−0.0249，p=1.41e-05，4/5 分辨率显著更低漂移）→ **SUPPORTED**。来源：`logs/f3_cga_analyze.log`、`results/abl/CGA_str_f3.csv`。
- **颜色指标（K6，300 图）**：CIEDE2000 −0.229（p=0.080，不显著改善）；EOR +0.0016（p=8.7e-09 显著）；RGB_SSIM −0.0093、Colorfulness −0.0043（与 AG 代价同向）；如实报告"混合但可解释"。来源：`results/arb/CGA_str_color_vs_B0cmp.csv`。
- **B2 蒸馏（C2 可靠性图蒸馏）——负结果**：epoch 1 val NaN、epoch 2–20 全 NaN、best checkpoint 从未落盘（val=inf）；仅存不可用 `_last.pt`。按预注册不调参不复跑；C2 叙事降级为局限/失败记录。来源：`logs/distill.log`、`results/seeds/B1_B2_EVIDENCE_20260912.md` §4。
- **NaN 现象（如实记录）**：CGA 系种子后期 val NaN（s123 ep12 起 / s3407 ep14 起，best 分别 0.3090/0.3063 于 NaN 前保存）；B0 系两种子 20 epoch 全程无 NaN。与 s42 同型；假说（commit 损失后期发散）留待后续，本轮不调参。来源：`logs/seeds_queue.log`。
- **FusionMamba 口径注记**：VMamba 编解码要求边长被 32 整除，17/300 图经 replicate pad-32 推理后裁回原尺寸（`test_m3fd_pad32.py`）；前 128 张原尺寸直推。来源：`logs/fusionmamba_m3fd_infer.log`。

## 6. 复现命令索引（全部已存在于仓库）

- 主表：`aggregate_sota.py`；种子表：`analyze_seeds.py --glob-cga 'results/seeds/CGA_str_s*.csv' --glob-b0 'results/seeds/B0_s*.csv'`
- 检测 A/B：`gen_cga_fused.py --variant <cell> --checkpoint <ckpt> --use-cga --full` + `arb_detect_full300.py`（旧 venv）；高冲突统计：`arb_full300_stats.py`；C3：`struct_noninferior.py`
- 机制可视化：`cga_visualize.py --checkpoint checkpoints/abl_CGA_str_s42.pt --auto 6`
- 种子训练链：`run_seeds_queue_v2.sh`（marker 断点）+ `run_after_seeds_v2.sh`（评测+蒸馏链）
- FusionMamba：`baselines_2025/FusionMamba/test_m3fd.py` + `test_m3fd_pad32.py` → `collect_outputs.py --methods fusionmamba`

---
**冻结声明**：以上数字即论文写作的唯一数据源。闸门-1 后进入 C 阶段（K7 重写 → 期刊初选 → 英文成稿 → 评审）。

> **补录指针(2026-09-14)**:本冻结文件原文不动;新增逐种子复核、CGA vs 官方配对检验、best epoch 披露、s123/s3407 产物校验、h5 内容控制组统计,以及 s3407 val NaN 起始更正(ep14→ep13)——全部见 `GATE1_AMEND_20260914.md`(依据 `writing/09-TIER1-ANALYSIS-20260914.md`)。
