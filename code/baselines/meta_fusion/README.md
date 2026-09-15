# MetaFusion — 优先补齐基线复现说明

> 地位：K2 审稿人点名、首投版"引未引、比未比"的**最高优先级缺口**。文献报告 §3 已核实：CVPR 2023, pp. 13955–13965, DOI:10.1109/cvpr52729.2023.01341（235+ 引）。
> 官方实现：`wdzhao123/MetaFusion`（PyTorch，pytorch 1.8.1 / cv2 4.5.5，README 提供推理命令）。

---

## 1. 获取

```bash
cd baselines_2025
git clone https://github.com/wdzhao123/MetaFusion.git MetaFusion
cd MetaFusion
# 官方 README 要求 pytorch 1.8.1 / cv2 4.5.5；本机为 cu118 环境，需独立 venv 或兼容运行
# 权重随仓库提供（README 说明下载位置）
```

## 2. 推理（官方命令）

```bash
python test.py \
  --test_ir_root  ../../code/baselines/inputs/m3fd/ir \
  --test_vis_root ../../code/baselines/inputs/m3fd/vi \
  --save_path     ../../code/baselines/outputs/meta_fusion/m3fd
```

- 输出文件名以输入 stem（即 sample_id）命名 → 与统一协议自动对齐。
- 注意官方实现可能将输入 resize 至固定尺寸（README 记录 512×384 训练/测试惯例）→ `collect_outputs.py --no-align` 关闭对齐后由统一协议 `--align` 强制回 IR 原分辨率，并在论文复现说明中注明。
- 若官方 `test.py` 输出路径结构不同（子目录），把真实输出整理为 `outputs/meta_fusion/m3fd/*.png` 即可。

## 3. 已核实引用信息（写论文用）

- Zhao W, Xie S, Zhao F, He Y, Lu H. MetaFusion: Infrared and Visible Image Fusion via Meta-Feature Embedding From Object Detection. CVPR 2023: 13955-13965. DOI: 10.1109/cvpr52729.2023.01341.
- 论文声明：M3FD 训练 2940 对 / 测试 1260 对（注意：与本项目 300 对测试口径不同，需在复现说明中说明测试子集来源——本项目 manifest `m3fd_test.csv` 为 300 对子集，见 README_unified_eval §3）。

## 4. 验收

```bash
bash ../../code/baselines/run_unified_eval.sh --datasets m3fd --methods meta_fusion
# 产物：code/results/baselines/meta_fusion_m3fd.csv（逐图 + __mean__/__std__）
```

> 红线：不出现 `results/baselines/meta_fusion_m3fd.csv` 且数字入表 = 禁止。贴真实 stdout 才算完成。
