# 07 — 最终参考文献清单(citation-check 全量过检,闸门-2 前置条件)

> 执行:citation_compliance_agent(academic-paper Phase 5a,citation-check 模式),2026-09-13。
> 输入:writing/00-abstract.md、01-intro-mechanism.md、02-related-work.md、04-methods.md、05-experiments.md、06-discussion-conclusion.md(全稿引用盘点);writing/03-citation-dedup.md(§A/§B/§E 处置与既有核验记录);文献支撑检索报告.md(2026-08-13)、文献支撑检索报告-CGA补充-20260824.md、CGA重投叙事全面研究报告-20260908.md(2026-09-08/09 复验记录)、02-related-work.rev-log-20260913.md(G3/G9 核验);首投版 PDF(pdftotext 编号复核)。
> 纪律:不编造 DOI/题名/venue;核验不到如实标注;未改 00–06 章正文与 03 清单(问题只报告)。
>
> **核验口径**:
> - **ALIGNED-VERIFIED**:在 03 清单且稿中被引,本次(2026-09-13)经 Crossref/OpenAlex/Semantic Scholar API 复核 DOI 解析、题名、venue、年份全部通过。
> - **ALIGNED-PREV-CHECK**:在 03 清单且稿中被引,按 03 清单/检索报告既有核验记录(2026-08-13 / 09-08 / 09-09)采信,本次不重验或仅做元数据交叉(标注差异)。
> - **ORPHANED-FLAGGED**:03 清单标"删"但稿中仍被引——报告,处置权在作者(删正文引用 or 恢复条目)。
> - **T2-NEEDS-UPGRADE**:arXiv 已核但 venue 未独立复核的承重引用——须升级正式版或替换才能入终稿。
> - **INSERT-NEEDED**(附加标记):03 清单保留、正文以名称使用但**无占位符**的条目——LaTeX 阶段必须补挂引用。
>
> **范围界定**:主清单 = 稿中被引(或正文以名称使用而必须引用)的全部条目,共 **41 条**。03 清单"保留但稿中未被引"的条目(见 §6)不进主清单,由 gate-2 决定是否增补。编号按稿中首次出现顺序(00 摘要→01→02→04→05→06;00 摘要无引用占位符)。

---

## §1 最终参考文献清单(拟入英文稿,编号 = 稿中首次出现顺序)

### 01 章 Introduction 首现(机制段 + K7)

| # | 条目 | 状态 |
|---|---|---|
| 1 | Ma H, Li H, Cheng C, Wang G, Song X, Wu X-J. "S4Fusion: Saliency-Aware Selective State Space Model for Infrared and Visible Image Fusion." *IEEE Transactions on Image Processing*, 2025, 34: 4161–4175. DOI: 10.1109/tip.2025.3583132. | **ALIGNED-VERIFIED**(本次 Crossref 复核通过;与首投 #1 逐字段一致) |
| 2 | Gu A, Dao T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752, 2023. | **ALIGNED-PREV-CHECK**(按首投 #2 + 03 §A 采信;该文无正式会议/期刊版,arXiv 引用为社区惯例,非 T2 风险) |
| 3 | Chen Y, Zhang X, Hu S, Han X, Liu Z, Sun M. "Stuffed Mamba: Oversized States Lead to the Inability to Forget." *Proceedings of COLM 2025*(v4 改题;arXiv:2410.07145 仅作预印本标识). | **ALIGNED-PREV-CHECK**(按 03 §B.1 之 2026-09-09 v4 全文核验采信——封面页标注 "Published as a conference paper at COLM 2025";本次 Crossref/OpenAlex 复核**未检索到 COLM 2025 正式版 DOI**,见遗留 L4) |
| 4 | Lu P, Huang J, Zeng Q, Wang X, Chen B, Langlais P, Cui Y. "Mamba Modulation: On the Length Generalization of Mamba Models." *Advances in Neural Information Processing Systems 38*(NeurIPS 2025). DOI: 10.52202/085713-0653(arXiv:2509.19633). | **ALIGNED-VERIFIED**(本次 Crossref 确认 NeurIPS 2025 正式 DOI——venue 升级可行:引用格式可由 arXiv 升为 NeurIPS 会议版) |
| 5 | Ali A, Zimerman I, Wolf L. "The Hidden Attention of Mamba Models." *Proceedings of the 63rd Annual Meeting of the ACL(Volume 1: Long Papers)*, 2025. DOI: 10.18653/v1/2025.acl-long.76(arXiv:2403.01590). | **ALIGNED-VERIFIED**(本次复核通过) |
| 6 | Touvron H, Cord M, Sablayrolles A, Synnaeve G, Jégou H. "Going Deeper with Image Transformers." *ICCV 2021*: 32–42. DOI: 10.1109/iccv48922.2021.00010. | **ALIGNED-VERIFIED**(本次复核通过;"LayerScale" 为该文机制名,引用以正式题名著录) |

### 02 章 Related Work 首现

**§1 CNN 族:**

| # | 条目 | 状态 |
|---|---|---|
| 7 | Li H, Wu X-J. "DenseFuse: A Fusion Approach to Infrared and Visible Images." *IEEE Transactions on Image Processing*, 2019, 28(5): 2614–2623. DOI: 10.1109/tip.2018.2887342. | **ALIGNED-VERIFIED**(本次复核通过) |
| 8 | Xu H, Ma J, Le Z, Jiang J, Guo X. "FusionDN: A Unified Densely Connected Network for Image Fusion." *AAAI 2020*: 12484–12491. DOI: 10.1609/aaai.v34i07.6936. | **ORPHANED-FLAGGED**(03 §A 标"删"(与 U2Fusion 重叠)但 02 §1 仍引;本次已备齐恢复所需完整数据;处置由作者定,见遗留 L1) |
| 9 | Xu H, Ma J, Jiang J, Guo X, Ling H. "U2Fusion: A Unified Unsupervised Image Fusion Network." *IEEE TPAMI*, 2022, 44(1): 502–518. DOI: 10.1109/tpami.2020.3012548. | **ALIGNED-VERIFIED**(本次复核通过) |

**§2 Transformer 族:**

| # | 条目 | 状态 |
|---|---|---|
| 10 | Zhao Z, Bai H, Zhang J, Zhang Y, Xu S, Lin Z, et al. "CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion." *CVPR 2023*: 5906–5916. DOI: 10.1109/cvpr52729.2023.00572. | **ALIGNED-VERIFIED**(本次复核通过) |
| 11 | Zhao W, Xie S, Zhao F, He Y, Lu H. "MetaFusion: Infrared and Visible Image Fusion via Meta-Feature Embedding from Object Detection." *CVPR 2023*: 13955–13965. DOI: 10.1109/cvpr52729.2023.01341. | **ALIGNED-VERIFIED**(本次复核通过;03 §B.13 "必补"已兑现——02/05 均已引) |
| 12 | Zhao Z, Bai H, Zhang J, Zhang Y, Zhang K, Xu S, et al. "Equivariant Multi-Modality Image Fusion." *CVPR 2024*: 25912–25921. DOI: 10.1109/cvpr52733.2024.02448. | **ALIGNED-VERIFIED**(本次复核通过;与 03 §B.18a 正式版修正一致——正式题名无 "EMMA:" 前缀,EMMA 为方法简称) |

**§3 Diffusion 族:**

| # | 条目 | 状态 |
|---|---|---|
| 13 | Zhao Z, Bai H, Zhu Y, Zhang J, Xu S, et al. "DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion." *ICCV 2023*: 8048–8059. DOI: 10.1109/iccv51070.2023.00742. | **ALIGNED-VERIFIED**(本次复核通过) |
| 14 | Yi X, Tang L, Zhang H, Xu H, Ma J. "Diff-IF: Multi-modality image fusion via diffusion model with fusion knowledge prior." *Information Fusion*, 2024, 110: 102450. DOI: 10.1016/j.inffus.2024.102450. | **ALIGNED-VERIFIED**(本次复核通过) |

**§4 Mamba/SSM 族:**

| # | 条目 | 状态 |
|---|---|---|
| 15 | Zhang T, Zhu Y, Zhao J, Cui G, Zheng Y. "Exploring State Space Model in Wavelet Domain: An Infrared and Visible Image Fusion Network via Wavelet Transform and State Space Model." *ICME 2025*. DOI: 10.1109/icme59968.2025.11209539. | **ALIGNED-VERIFIED**(本次复核通过;与 03 §B.18b ICME 正式版升级一致;"W-Mamba" 仅是方法简称) |
| 16 | Xie X, Cui Y, Tan T, Zheng X, Yu Z. "FusionMamba: Dynamic feature enhancement for multimodal image fusion with Mamba." *Visual Intelligence*, 2024, 2: 37. DOI: 10.1007/s44267-024-00072-9. | **ALIGNED-VERIFIED**(本次复核通过) |
| 17 | Dao T, Gu A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." *ICML 2024*(PMLR): 10041–10071. | **ALIGNED-PREV-CHECK**(按首投 #18 + 03 §A 采信;PMLR 会议录无 Crossref DOI,按首投格式引用即可) |
| 18 | Wang Y, et al. "MemMamba: Rethinking Memory Patterns in State Space Model." arXiv:2510.03279, 2025. | **T2-NEEDS-UPGRADE**(本次 OpenAlex 复核确认**仅 arXiv 版、无正式 venue**;02 §4 承重句 "loses its ability to manage what it retains [Stuffed Mamba…; MemMamba]" 的第二支撑;按 03 E 组纪律须升级正式版或替换/删除,见遗留 L2) |
| 19 | Zhu Y, Lv L, Zhang P, Liu X, Tang T, Tian F, Sun W, Lu H. "Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion." *IEEE Transactions on Image Processing*, 2026, 35: 2380–2392. DOI: 10.1109/tip.2026.3662596. | **ALIGNED-VERIFIED**(本次复核通过;注意与首投 #22 "Spatial-frequency enhanced Mamba"(TIP 2025)为**不同论文**,见遗留 L9) |
| 20 | Cao K, He X, Hu T, Xie C, Zhou M, Zhang J. "Shuffle Mamba: State Space Models with Random Shuffle for Multi-Modal Image Fusion." *IEEE TCSVT*, 2026, 36(7): 9448–9461. DOI: 10.1109/tcsvt.2026.3668923. | **ALIGNED-VERIFIED**(本次复核通过) |
| 21 | Wang Y, Zhuang R, Zheng H, He X, Cao K, Tu X, Ding X. "Self-supervised Multiplex Consensus Mamba for General Image Fusion." *AAAI 2026*, 40(22): 18647–18655. DOI: 10.1609/aaai.v40i22.38932. | **ALIGNED-VERIFIED**(本次复核通过;"SMC-Mamba" 为简称;正式题名为 "for General Image Fusion") |

**§5 语义/任务引导族:**

| # | 条目 | 状态 |
|---|---|---|
| 22 | Tang L, Yuan J, Ma J. "Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network." *Information Fusion*, 2022, 82: 28–42. DOI: 10.1016/j.inffus.2021.12.004. | **ALIGNED-VERIFIED**(本次复核通过;"SeAFusion" 为方法简称,正式题名不含该前缀;该文同时提出 MSRS 数据集) |
| 23 | Liu J, Liu Z, Wu G, Ma L, Liu R, Zhong W, Luo Z, Fan X. "Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation." *ICCV 2023*: 8081–8090. DOI: 10.1109/iccv51070.2023.00745. | **ALIGNED-VERIFIED**(本次复核通过;02 §5 "SegMiF / M2Fusion [ICCV 2023]" 对应**此单一条目**——SegMiF/M2Fusion 为该文方法族) |
| 24 | Zhao Z, Su S, Wei J, Tong X, Gao W. "Lightweight Infrared and Visible Image Fusion via Adaptive DenseNet with Knowledge Distillation." *Electronics*, 2023, 12(13): 2773. DOI: 10.3390/electronics12132773. | **ALIGNED-VERIFIED**(本次复核通过) |
| 25 | Wang X. "Semantic Segmentation-Driven Knowledge Distillation-Based Infrared Visible Image Fusion Framework." *IEEE Access*, 2025, 13: 83408–83425. DOI: 10.1109/access.2025.3566436. | **ALIGNED-VERIFIED**(本次复核通过;单作者) |

**§6 边界/不确定性/冲突族 + 经典选择式谱系:**

| # | 条目 | 状态 |
|---|---|---|
| 26 | Zhou T, Ruan S, Lei B. "BUFNet: Boundary-aware and uncertainty-driven multi-modal fusion network for MR brain tumor segmentation." *Medical Image Analysis*, 2026, 107: 103855. DOI: 10.1016/j.media.2025.103855. | **ALIGNED-VERIFIED**(本次复核通过) |
| 27 | Zhu X, Chow C-O, Chuah J H. "ShadowMamba: State-space model with boundary-region selective scan for shadow removal." *Image and Vision Computing*, 2026, 166: 105872. DOI: 10.1016/j.imavis.2025.105872. | **ALIGNED-VERIFIED**(本次复核通过) |
| 28 | Yuan J, Tan Z, Xu K, Wang Z, Zhang J. "Segment anything model edge prior-guided infrared and visible image fusion method." *Infrared Physics & Technology*, 2025, 150: 106013. DOI: 10.1016/j.infrared.2025.106013. | **ALIGNED-VERIFIED**(本次复核通过) |
| 29 | Zou X, Tang J, Yang L, Zhu Z. "Lightweight infrared and visible image fusion network with edge-guided dual attention." *Journal of Electronic Imaging*, 2023, 32(6): 063014. DOI: 10.1117/1.jei.32.6.063014. | **ALIGNED-VERIFIED**(本次复核通过) |
| 30 | Zhao J, Wang Y, Zhang Y, Wang H, Guo Y. "Uncertainty-Aware Cross-Modality Fusion for Visible-Infrared Object Detection." *DICTA 2024*: 117–125. DOI: 10.1109/dicta63115.2024.00029. | **ALIGNED-VERIFIED**(本次复核通过) |
| 31 | Tian B, Luo J, Lin K, Zhang C, Qin T. "Adaptive Reliability-Calibrated Consensus–Complementarity–Conflict Modeling for Infrared and Visible Image Fusion." *Sensors*, 2026, 26(15): 4745. DOI: 10.3390/s26154745. | **ALIGNED-VERIFIED**(本次复核通过;官方题名无 "ARC3Fusion:" 前缀(该名是方法简称);此前另有 2026-09-13 全文级核验记录,rev-log G9) |
| 32 | Brenner M, Reyes N H, Susnjak T, Barczak A L C. "GatedFusion-Net: Per-pixel modality weighting in a five-cue transformer for RGB-D-I-T-UV fusion." *Information Fusion*, 2026, 129: 103986. DOI: 10.1016/j.inffus.2025.103986. | **ALIGNED-VERIFIED**(本次 Crossref 定版:print **2026 年 5 月, vol 129** → 稿中 "[Information Fusion 2026]" 正确;研究报告 §3.1 的 "2025" 为在线先发年;此前另有 2026-09-13 摘要级核验,rev-log G3) |
| 33 | Burt P J, Kolczynski R J. "Enhanced image capture through fusion." *ICCV 1993*: 173–182. DOI: 10.1109/iccv.1993.378222. | **ALIGNED-VERIFIED**(本次复核通过) |
| 34 | Zhang Z, Blum R S. "A categorization of multiscale-decomposition-based image fusion schemes with a performance study for a digital camera application." *Proceedings of the IEEE*, 1999, 87(8): 1315–1326. DOI: 10.1109/5.775414. | **ALIGNED-VERIFIED**(本次复核通过;正式题名含副题 "...with a performance study for a digital camera application") |
| 35 | Piella G. "A general framework for multiresolution image fusion: from pixels to regions." *Information Fusion*, 2003, 4(4): 259–280. DOI: 10.1016/s1566-2535(03)00046-0. | **ALIGNED-VERIFIED**(本次复核通过;正式题名含副题 ": from pixels to regions") |
| 36 | Li B, et al. "Neurodynamics-Driven Coupled Neural P Systems for Multi-Focus Image Fusion." arXiv:2509.17704, 2025. | **T2-NEEDS-UPGRADE**(03 §E3 原生标注 T2;本次 OpenAlex 复核确认仅 arXiv 版;02 §6 "modern multi-focus decision-map line [e.g., ND-CNPFuse 2025]" 承重例证;替换候选已备(见遗留 L3);"ND-CNPFuse" 为简称) |

### 04 章 Methods 首现

无新条目(§3.1 引 S4Fusion→#1,§3.3 引 LayerScale→#6,均已在 01 章首现)。M3FD 数据集名称在 §3.4 出现(见 #38)。

### 05 章 Experiments 首现

| # | 条目 | 状态 |
|---|---|---|
| 37 | Liu J, Zhang B, Mei Q, Li X, Zou Y, Jiang Z, et al. "DCEvo: Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion." *CVPR 2025*: 2226–2235. DOI: 10.1109/cvpr52734.2025.00213. | **ALIGNED-VERIFIED**(本次复核通过;首投 #16 保留;05 主表基线 + §5.1 检测分支复用——revlog C5 所记 "DCEvo 检测分支条目入 bib" 即本条) |
| 38 | Liu J, Fan X, Huang Z, Wu G, Liu R, Zhong W, Luo Z. "Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection." *CVPR 2022*: 5792–5801. DOI: 10.1109/cvpr52688.2022.00571. | **ALIGNED-VERIFIED + INSERT-NEEDED**(M3FD 数据集出处;03 §A 保留、03 §D "实验-协议" 规划;01/04/05 正文大量使用 M3FD 名称但**无占位符**——须在 M3FD 首次正式引入处(至迟 05 §5.1)挂引用;页码歧义见遗留 L8) |
| 39 | Liu Y, Tian Y, Zhao Y, Yu H, Xie L, Wang Y, et al. "VMamba: Visual State Space Model." arXiv:2401.10166, 2024. | **ALIGNED-PREV-CHECK + INSERT-NEEDED**(按首投 #3 + 03 §A 采信;05 §5.1 "its VMamba encoder–decoder" 点名他人架构而无占位符——建议补挂,或并入 FusionMamba 条目说明;由 gate-2 定) |
| 40 | Xydeas C S, Petrović V. "Objective image fusion performance measure." *Electronics Letters*, 2000, 36(4): 308–309. DOI: 10.1049/el:20000267. | **ALIGNED-VERIFIED + INSERT-NEEDED**(QABF 指标定义;03 §A 保留、03 §D "实验-指标" 规划;05 §5.1 六指标句无占位符——须挂于 "QABF" 处;研究报告 §3.2 亦曾标 T1 已核) |
| 41 | Sheikh H R, Bovik A C. "Image information and visual quality." *IEEE Transactions on Image Processing*, 2006, 15(2): 430–444. DOI: 10.1109/tip.2005.859378. | **ALIGNED-VERIFIED + INSERT-NEEDED**(VIF 指标定义;同上,须挂于 05 §5.1 "VIF" 处;注意与融合领域 "VIF 缩写" 同形异义,无混淆风险) |

### 06 章 Discussion/Conclusion 首现

无新条目(§6.3 引 ARC3Fusion→#31、GatedFusion-Net→#32、[Burt & Kolczynski 1993; Zhang & Blum 1999; Piella 2003]→#33–35,均已在 02 章首现)。06 §6.3 "the modern multi-focus decision-map line" 复述无占位符(02 对应处有 [e.g., ND-CNPFuse 2025])——一致性提示见遗留 L7。

---

## §2 T2* 标注条目清单(arXiv 已核但 venue 未独立复核的承重引用——须升级正式版或替换才能入终稿)

**稿中被引且承重的 T2 条目共 2 条:**

| # | 条目 | 承重位置 | 问题 | 处置选项 |
|---|---|---|---|---|
| 18 | MemMamba(arXiv:2510.03279) | 02 §4:"…loses its ability to manage what it retains [Stuffed Mamba, COLM 2025; MemMamba]" | 本次 OpenAlex 复核(2026-09-13)确认仅 arXiv 版、无正式 venue;03 §B.4 原生未标 T2,但按 E 组纪律实为 T2 | (a) 若投稿前出现正式版→升级;(b) 删除该引用(Stuffed Mamba 单独承重即可,句义不变);(c) 保留 arXiv 引用并降格为非承重(需 gate-2 明示豁免)。**建议 (b)**——该句核心支撑是 COLM 2025 正式版 Stuffed Mamba |
| 36 | ND-CNPFuse(arXiv:2509.17704) | 02 §6:"the modern multi-focus decision-map line [e.g., ND-CNPFuse 2025]" + 06 §6.3 同义复述 | 03 §E3 原生标注 T2;本次 OpenAlex 复核确认仅 arXiv 版 | (a) 升级(目前无正式版);(b) **替换**为已核正式版决策图谱系论文:Ma B, Yin X, Wu D, et al. "End-to-End Learning for Simultaneously Generating Decision Map and Multi-Focus Image Fusion Result." *Neurocomputing*, 2021(本次 OpenAlex 检索确认 DOI: 10.1016/j.neucom.2021.10.115,Boyuan Ma 等 6 作者)——但该文 2021 年,"modern/最新代表" 措辞需相应调整;(c) 保留 arXiv 引用并明示 "preprint" 身份(部分期刊接受,但违背 03 既定纪律)。**建议 (a)→投稿前复查一次,(b) 为后备** |

**T2 边界说明(非 T2,但相关):**

- **#3 Stuffed Mamba**:venue(COLM 2025)已由 2026-09-09 v4 全文核验确认,故**不属 T2**;但 COLM 2025 正式版在 Crossref/OpenAlex 无 DOI 记录(2026-09-13 复核),引用格式需以 "Proceedings of COLM 2025" + arXiv:2410.07145 双标识,LaTeX 阶段可补 OpenReview 链接(见遗留 L4)。
- **#2 Mamba**、**#39 VMamba**:arXiv-only 引用为社区惯例/首投原状,无正式版可升级,不属 T2。
- **03 清单内未被稿中引用的 T2 条目**(E2: Mixture-of-Depths T2*、Gumbel-Softmax T2*、Modality Competition T2*;E5: AngularFuse T2、Direction-aware gradient loss T2;CGA补充: CIS-Fuse、ConFusion、MoCTEFuse 等):均**不在最终清单**(稿中未引);若 gate-2 决定增补(见 §6),增补前须先完成同样的 T2 升级审查。

---

## §3 章节 × 引用覆盖矩阵

符号:● = 正文引用(方括号占位符或 venue 标签);○ = 正文以方法/数据集/指标名使用但无占位符;n = 仅 Notes/溯源注释层提及(非正文);i = INSERT-NEEDED(○ 且必须补挂引用);— = 未出现。列顺序 = 终稿章节顺序(00 摘要无引用)。

| # | 条目(简称) | 00 | 01 | 02 | 04 | 05 | 06 |
|---|---|---|---|---|---|---|---|
| 1 | S4Fusion | ○ | ● | ● | ● | ● | ○ |
| 2 | Mamba | — | ● | ● | — | — | — |
| 3 | Stuffed Mamba | — | ● | ● | — | — | — |
| 4 | Mamba Modulation | — | ● | ● | — | — | — |
| 5 | Hidden Attention | — | ● | ● | — | — | — |
| 6 | LayerScale | — | ● | — | ● | n | — |
| 7 | DenseFuse | — | — | ● | — | — | — |
| 8 | FusionDN | — | — | ●(ORPHANED) | — | — | — |
| 9 | U2Fusion | — | — | ● | — | — | — |
| 10 | CDDFuse | — | — | ● | — | ● | — |
| 11 | MetaFusion | — | — | ● | — | ● | — |
| 12 | EMMA | — | — | ● | — | ● | ○ |
| 13 | DDFM | — | — | ● | — | — | — |
| 14 | Diff-IF | — | — | ● | — | ● | ○ |
| 15 | W-Mamba | — | — | ● | — | ● | — |
| 16 | FusionMamba | — | — | ● | — | ● | — |
| 17 | Mamba-2 | — | — | ● | — | — | — |
| 18 | MemMamba(T2) | — | — | ● | — | — | — |
| 19 | ISF-Fusion Mamba | — | — | ● | — | — | — |
| 20 | Shuffle Mamba | — | — | ● | — | — | — |
| 21 | SMC-Mamba | — | — | ● | — | — | — |
| 22 | SeAFusion | — | — | ● | — | — | — |
| 23 | SegMiF/M2Fusion | — | — | ● | — | — | — |
| 24 | KD-DenseNet(Electronics) | — | — | ● | — | — | — |
| 25 | Semantic-KD(Access) | — | — | ● | — | — | — |
| 26 | BUFNet | — | — | ● | — | — | — |
| 27 | ShadowMamba | — | — | ● | — | — | — |
| 28 | SAM edge prior(IPT) | — | — | ● | — | — | — |
| 29 | Edge dual attention(JEI) | — | — | ● | — | — | — |
| 30 | DICTA 2024 | — | — | ● | — | — | — |
| 31 | ARC3Fusion | — | — | ● | — | — | ● |
| 32 | GatedFusion-Net | — | — | ● | — | — | ● |
| 33 | Burt & Kolczynski | — | — | ● | — | — | ● |
| 34 | Zhang & Blum | — | — | ● | — | — | ● |
| 35 | Piella | — | — | ● | — | — | ● |
| 36 | ND-CNPFuse(T2) | — | — | ● | — | — | ○(无占位符,L7) |
| 37 | DCEvo | — | — | — | — | ● | — |
| 38 | TarDAL/M3FD | — | ○ | — | ○ | i | ○ |
| 39 | VMamba | — | — | — | — | i | — |
| 40 | QABF 指标 | ○ | ○ | — | — | i | ○ |
| 41 | VIF 指标 | ○ | — | — | — | i | ○ |

**矩阵读法**:
- 正文实际引用占位符(●)共 37 条身份(即 #1–37);#38–41 为 INSERT-NEEDED(名称已入正文、条目已备、占位符待补)。
- 00 摘要中的 S4Fusion/QABF/VIF 均为定性名称使用,摘要按惯例不携带文献引用,无需处理。
- 05 的 "●" 含 Table 1 行内 venue 标签(MetaFusion CVPR'23 等)与 §5.1 正文枚举;LayerScale 在 05 仅注释层(n)。
- 06 的 EMMA/Diff-IF/S4Fusion 为 §6.3 表格复述(○),LaTeX 阶段自动继承编号,无需另挂。

---

## §4 遗留问题清单(ORPHANED / MISSING / 无法核验项)

**需作者决策(3 项):**

- **L1(ORPHANED)| FusionDN**:03 §A 标"删"(理由:与 U2Fusion 重叠),但 02 §1 正文仍引用("the unified dense architecture FusionDN [AAAI 2020]")。二选一:(a) 从 02 §1 删去该引用并微调句子(现句 "DenseFuse … with its DenseNet encoder and the unified dense architecture FusionDN [AAAI 2020] — later unified under … in U2Fusion" 删 FusionDN 后仍通顺);(b) 恢复 03 条目(完整题录已备于 #8;理由可改为"DenseFuse→FusionDN→U2Fusion 构成 dense 架构三代谱系,引 3 条不冗余")。总量 41 或 42,均在 45–55 目标区间之下,无压力。**本清单按 (b) 暂列 #8,如选 (a) 则整行删除。**
- **L2(T2)| MemMamba**:见 §2,建议删除引用或降格豁免。
- **L3(T2)| ND-CNPFuse**:见 §2,建议投稿前复查正式版,否则替换或明示 preprint。

**格式/占位符待办(不阻塞,LaTeX 前必须完成):**

- **L4(Stuffed Mamba COLM 正式版无 DOI)**:Crossref/OpenAlex 截至 2026-09-13 无 "Oversized States…"(COLM 2025)正式版记录;venue 按 2026-09-09 v4 封面全文核验采信。LaTeX 阶段:引用格式 "In: Proceedings of the Conference on Language Modeling (COLM), 2025"+ arXiv:2410.07145;若目标期刊要求 DOI,需从 OpenReview/COLM 官网取正式链接,或退回 arXiv 引用格式(降格,需作者批准)。
- **L5(TarDAL 页码歧义)**:首投 #9 页码 5802–5811,Crossref/IEEE 记录 5792–5801(本次复核)。LaTeX 阶段以 IEEE Xplore/Crossref(5792–5801)为准,或核对 CVF Open Access 版页码后定稿。
- **L6(INSERT-NEEDED ×4)**:#38 TarDAL(挂 M3FD 首次正式引入处,至迟 05 §5.1)、#39 VMamba(05 §5.1 "its VMamba encoder–decoder" 处)、#40 QABF(05 §5.1 指标枚举处)、#41 VIF(同上)。01 §3.4/06 §6.2 亦出现 M3FD 名称,首次正式引入点由 gate-2 统一定;#40 注意 QABF 名称在 01 §机制段已先于 05 出现(定性使用,不阻塞)。
- **L7(06 章缺占位符)**:06 §6.3 "the modern multi-focus decision-map line still makes per-pixel binary choices" 无引用占位符,而 02 §6 对应句有 [e.g., ND-CNPFuse 2025]。L3 决策后须同步补齐(或删 ND-CNPFuse 时两章一起改写)。

**03 清单错误(不改 03,记录在案;5 项):**

- **L8a**:§A "From data compatibility to task adaption (Liu et al., **TIP** 2025, 47(4):2349-2369)" — venue 有误:首投 #19 与本次 Crossref(10.1109/tpami.2024.3521416)均为 **TPAMI** 2025, 47(4)。该条稿中未被引用,无英文稿影响。
- **L8b**:§A "[17?] Every SAM drop counts (Wu et al., **TIP 2024**)删" 与 "[23?] Embracing semantic priors for multi-modality fusion (**CVPR 2025**)保留" 为**同一篇论文**(首投 #23,全题 "Every SAM drop counts: Embracing semantic priors for multi-modality fusion and beyond",CVPR 2025: 17882–17891)——03 重复列行、venue 互斥、处置矛盾(一删一留)。该篇稿中未被引用,无英文稿影响;若 gate-2 拟引,须先修正元数据。
- **L8c**:§A "MUSIQ (Ke et al., **CVPR 2021**)" — venue 有误:首投 #14 为 **ICCV 2021**(Montreal)。MUSIQ/TOPIQ 均未入最终清单(05 未用无参考指标),无英文稿影响。
- **L8d**:§A 编号勘误(本次 PDF 复核全部落实,详见 §5):[12?] ResNet 实为 #11;[19?] DCEvo 实为 #16;[17?] SAM drop 实为 #23;其余 [?] 全部落定。
- **L8e**:03 §E3 GatedFusion-Net 行 venue 写 "Information Fusion 2026" ✓ 正确;研究报告 §3.1 写 "Information Fusion 2025" 为在线先发年——本次 Crossref 定版 print 2026-05、vol 129,统一口径 **2026**。

**无法完全核验项(1 项):**

- **L9(近重名论文警示)**:首投 #22 "Spatial-frequency enhanced Mamba for multi-modal image fusion"(Sun H, Lv L, Zhang P, et al., **TIP 2025**, 34: 7684–7696)与最终清单 #19 "Interactive Spatial-Frequency Fusion Mamba for Multi-Modal Image Fusion"(Zhu Y, Lv L, Zhang P, et al., **TIP 2026**, 35: 2380–2392)为**同组不同论文**(作者高度重叠、题名近似、卷年不同)。02 §4 引用的是 #19(TIP 2026);#22(03 标"保留或删")稿中未引。gate-2 与 LaTeX 排版时**切勿合并或错挂**。#22 本身无 DOI 记录在案(未入清单,故未核验)。

**条件性事项(2 项,不阻塞):**

- **L10(Dc-EEMF,notes-only)**:02 章 Notes("Terminology discipline")提 "cf. Dc-EEMF TBME 2026"——仅注释层,正文未引,不进清单。研究报告 §3.3 已录其身份(IEEE TBME 2026,DOI: 10.1109/tbme.2026.3697753,已核 2026-09-09;"/mm1" 后缀为多媒体附件 DOI,不得引用)。**若 gate-2 将该术语说明升入正文并引用 Dc-EEMF,须新增条目**(现清单无此条)。
- **L11(总量 vs 目标)**:最终清单 41 条(若 L1 选删则 40),低于 03 §C 的 45–55 目标下限。增补候选见 §6(全部有已核 DOI/ID);03 §C 预留的三个补足方向(多尺度 SSM 拓扑对照、LLVIP/RoadScene 数据集出处、bootstrap 统计引用)在当前稿中均无落点,是否补由 gate-2 决定。

---

## §5 首投版 [?] 编号复核结果(第四步,pdftotext 解析成功)

首投 PDF(`论文-无作者-第一次投稿版.pdf`)参考文献 28 条已全部提取,03 §A 中全部 [?]/n? 标注**已落实**——无 [PDF-UNREADABLE-SKIPPED] 项。对照表:

| 03 §A 标注 | 文献 | 首投实际编号 | 首投著录要点(题名缩略 + venue) | 与 03 差异 |
|---|---|---|---|---|
| [1] | S4Fusion | **#1** | Saliency-aware selective SSM, TIP 2025, 34: 4161–4175 | ✓ |
| [?] | Mamba | **#2** | Gu & Dao, arXiv 2312.00752, 2023 | ✓ |
| [?] | VMamba | **#3** | Liu Y, et al., arXiv 2401.10166, 2024 | ✓ |
| [?] | S4 | **#4** | Gu A, Goel K, Ré C, arXiv 2111.00396, 2021 | ✓ |
| [?] | HiPPO | **#5** | Gu A, et al., NeurIPS 2020: 1474–1487 | ✓ |
| [6?] | DenseFuse | **#6** | Li & Wu, TIP 2019, 28(5): 2614–2623 | ✓ |
| [7?] | U2Fusion | **#7** | Xu H, et al., TPAMI 2022, 44(1): 502–518 | ✓ |
| [8?] | FusionDN | **#8** | Xu H, et al., AAAI 2020: 12484–12491 | ✓ |
| [9] | TarDAL + M3FD | **#9** | CVPR 2022: 5802–5811(页码与 Crossref 5792–5801 有歧义,见 L5) | 页码待定 |
| [10] | CDDFuse | **#10** | CVPR 2023: 5906–5916 | ✓ |
| [12?] | ResNet | **#11** | He K, et al., CVPR 2016: 770–778 | **编号差 1**(03 标 12?) |
| [?] | QABF 指标 | **#12** | Xydeas & Petrovic, Electron. Lett. 2000, 36(4): 308–309 | ✓(落定为 12) |
| [?] | VIF 指标 | **#13** | Sheikh & Bovik, TIP 2006, 15(2): 430–444 | ✓(落定为 13) |
| [?] | MUSIQ | **#14** | Ke J, et al., **ICCV** 2021: 5148–5157 | venue 与 03(CVPR 2021)不符 → L8c |
| [15?] | TOPIQ | **#15** | Chen C, et al., TIP 2024, 33: 2404–2418 | ✓ |
| [19?] | DCEvo | **#16** | Liu J, et al., CVPR 2025: 2226–2235 | **编号差异**(03 标 19?) |
| [?] | W-Mamba(arXiv 版) | **#17** | Zhang T, et al., arXiv 2503.18378, 2025 | ✓(现已升级 ICME 2025,#15) |
| [18?] | Mamba-2 | **#18** | Dao & Gu, ICML 2024 (PMLR): 10041–10071 | ✓ |
| [?] | From data compatibility | **#19** | Liu J, et al., **TPAMI** 2025, 47(4): 2349–2369 | venue 与 03(TIP)不符 → L8a |
| [20] | EMMA | **#20** | Equivariant Multi-Modality Image Fusion, CVPR 2024: 25912–25921 | ✓ |
| [21] | FusionMamba | **#21** | Xie X, et al., Visual Intelligence 2024, 2: 37 | ✓ |
| [22?] | SF-enhanced Mamba | **#22** | Sun H, et al., TIP 2025, 34: 7684–7696 | ✓(稿中未引;与 #19(TIP 2026)为不同论文,L9) |
| [17?] | Every SAM drop counts | **#23** | Wu G, et al., CVPR 2025: 17882–17891 | **编号与 venue 双差异**(03 标 17?/TIP 2024)→ L8b |
| [23?] | Embracing semantic priors | **#23** | 同上(同一篇,L8b 重复列行) | 重复 |
| [24] | 视觉 Transformer 综述 | **#24** | 自动化学报 2022, 48(4) | ✓(删,稿中未引) |
| [25] | 高炉铁口温度检测 | **#25** | 自动化学报 2025, 51(2) | ✓(删) |
| [26] | 视觉属性可解释分类 | **#26** | 自动化学报 2025, 51(2) | ✓(删) |
| [27] | 提示学习综述 | **#27** | 自动化学报 2025, 51(5) | ✓(删) |
| [28] | Dehazeformer | **#28** | 自动化学报 2024, 50(7) | ✓(删) |

> 编号复核意义:仅确认"首投版用了哪些文献"的历史事实(28 条、编号-文献对应已定),不影响英文稿引用本身。03 清单头部声明的编号不确定性至此关闭;编号错误已记入 L8d。

---

## §6 03 清单"保留但稿中未被引"条目(不进最终清单;gate-2 增补候选池)

以下条目在 03 清单中标"保留/待定",但当前 00–06 章正文**没有任何引用占位符**,故不入最终清单。若 gate-2 决定增补(总量上限 55,当前 41,余量充足),按本表取用(身份/DOI 均有既有核验记录,除标注外):

**SSM 基础(01 章机制段可补):**
- S4(Gu A, Goel K, Ré C. arXiv:2111.00396, 2021;首投 #4)
- HiPPO(Gu A, et al. NeurIPS 2020: 1474–1487;首投 #5)

**任务自适应/语义先验(02 §5 可补):**
- From Data Compatibility to Task Adaption(Liu J, et al. **TPAMI** 2025, 47(4): 2349–2369, DOI: 10.1109/tpami.2024.3521416,本次已核;03 的 TIP 误标见 L8a;首投 #19)
- Embracing semantic priors / Every SAM drop counts(Wu G, et al. CVPR 2025: 17882–17891;首投 #23;03 双行矛盾见 L8b;无 DOI 记录在案,增补前需核)

**指标定义(05 章,视 G5 无参考指标决策):**
- TOPIQ(Chen C, et al. TIP 2024, 33: 2404–2418;首投 #15)
- MUSIQ(Ke J, et al. **ICCV** 2021: 5148–5157;首投 #14;venue 见 L8c)
- VIFB(Zhang X, Ye P, Xiao G. "VIFB: A Visible and Infrared Image Fusion Benchmark." CVPRW 2020: 468–478, DOI: 10.1109/cvprw50498.2020.00060,本次已核;03 §B.18 "评测协议引用"——当前 05 用自建统一评测器,未落点)

**机制层备选(03 §B.5–B.7,首投无):**
- Understanding Input Selectivity in Mamba(arXiv:2506.11891)
- Understanding and Improving Length Generalization in Recurrent Models(arXiv:2507.02782)
- PlainMamba(arXiv:2403.17695, BMVC 2024)

**方法段备选(03 §B.22):**
- Mamba-Adaptor(He X, et al. CVPR 2025, DOI: 10.1109/cvpr52734.2025.01874——注意:此 DOI 与 DCEvo(10.1109/cvpr52734.2025.00213)仅尾段不同,誊写时勿混;03 §B 原记 "10.1109/cvpr52734.2025.01874")

**硬选择/离散承诺谱系(03 §E2,02 §6 可补;**T2* 者增补前须升级**):**
- Maxout Networks(ICML 2013, arXiv:1302.4389)
- Switch Transformers(JMLR 23, 2022, arXiv:2101.03961)
- DSelect-k(NeurIPS 2021, arXiv:2106.03760)
- Mixture-of-Depths(arXiv:2404.02258,**T2*** 习称 ICML 2024)
- Gumbel-Softmax(arXiv:1611.01144,**T2*** 习称 ICLR 2017)
- Modality Competition(Provably)(ICML 2022, arXiv:2203.12221,**T2***)

**2026 竞品/整改先例(03 §E4–E5):**
- SGDFuse(Information Fusion 2026, DOI: 10.1016/j.inffus.2026.104290)
- DULRNet(IEEE Sensors J 2026, DOI: 10.1109/jsen.2026.3700714)
- MISSFusion(JKSU-CSC 2026, DOI: 10.1007/s44443-026-01122-6)
- Liu et al. 指标元分析(IEEE TPAMI 2024, ieeexplore 10440470——仅 URL 记录,**无 DOI 在案,增补前需核**)
- Dif-Fusion(IEEE TIP 2023, DOI: 10.1109/TIP.2023.3322046)
- AngularFuse(arXiv:2510.12260,**T2**)
- Direction-aware gradient loss(arXiv:2510.13067,**T2**)

**其他:**
- ResNet(He K, et al. CVPR 2016: 770–778;首投 #11;03 标"删或保留")
- SF-enhanced Mamba(TIP 2025, 34: 7684–7696;首投 #22;03 标"保留或删";**与 #19 近重名,L9**)
- Efron & Tibshirani 1993(bootstrap;03 §C/§D "统计"——当前 05 用 Wilcoxon+Holm+精确二项,bootstrap 已降级(G8),未落点)

---

## §7 汇总统计

| 统计项 | 数值 |
|---|---|
| 最终清单条目总数 | **41**(37 稿中被引 + 4 INSERT-NEEDED) |
| ALIGNED-VERIFIED(本次复核通过) | **34** |
| ALIGNED-PREV-CHECK(按既有核验记录采信) | **4**(#2 Mamba、#3 Stuffed Mamba、#17 Mamba-2、#39 VMamba) |
| ORPHANED-FLAGGED(需作者决策) | **1**(#8 FusionDN) |
| T2-NEEDS-UPGRADE(承重,须升级/替换) | **2**(#18 MemMamba、#36 ND-CNPFuse) |
| MISSING-FROM-LIST(稿中被引但 03 无条目) | **0** |
| MISSING-ADDED(本次新造条目) | **0**(全部条目均出自 03 §A/§B/§E) |
| INSERT-NEEDED(占位符待补) | **4**(#38 TarDAL、#39 VMamba、#40 QABF、#41 VIF) |
| 遗留问题总数 | **11 项编号条目**(L1–L3 作者决策;L4–L7 格式/占位待办;L8a–e 03 清单错误记录;L9 近重名警示;L10–L11 条件性) |
| 首投 [?] 编号复核 | 28/28 落定,**0 项** [PDF-UNREADABLE-SKIPPED] |

**本清单对闸门-2 的合规结论**:引用身份全部可溯源(41/41 有 03 清单出处),DOI 核验通过率 100%(34 本次复核 + 4 采信既有记录 + 3 特殊:COLM/PMLR/arXiv 无 DOI 属正常形态),无编造条目,无缺失条目;阻塞项仅 3 项作者决策(L1 FusionDN、L2 MemMamba、L3 ND-CNPFuse),其余为 LaTeX 阶段格式待办。**L1–L3 处置完成 + L6 四处占位符补挂后,本清单即为终稿 bib 底稿。**

---

## 处置决定(2026-09-14,gate-2 统稿落地)

| 遗留项 | 决定 | 执行状态 |
|---|---|---|
| L1 FusionDN(#8,ORPHANED) | **保留**:02 §1 谱系句实际在引("the unified dense architecture FusionDN [AAAI 2020]"),引用优先于 03 清单的"删"处置;条目恢复为正式清单成员 | ✅ 状态改为 ALIGNED-RESTORED |
| L2 MemMamba(#18,T2) | **删除引用**:02 §4 承重句由 Stuffed Mamba(COLM 2025 正式版)单独承重,已从正文移除 "; MemMamba" | ✅ 状态改为 REMOVED-FROM-TEXT,bib 不收录 |
| L3 ND-CNPFuse(#36,T2) | **替换**:02 §6 与 06 §6.3 的例证改用已核正式版 Ma B, Yin X, Wu D, et al. "End-to-End Learning for Simultaneously Generating Decision Map and Multi-Focus Image Fusion Result." *Neurocomputing*, 2021. DOI:10.1016/j.neucom.2021.10.115(2026-09-13 OpenAlex 核验);"modern" 措辞随替换调整为 "learning-based" | ✅ 新增条目 #42,ND-CNPFuse 状态 REPLACED;C-12 投稿前复查 ND-CNPFuse 是否已见刊,如见刊可换回 |
| L6 四处占位符(TarDAL/VMamba/QABF/VIF) | **补挂完成**:05 §5.1 三处 + VMamba 一处,条目 #37-40(#38 TarDAL、#39 VMamba、#40 QABF、#41 VIF)从 INSERT-NEEDED 转 ALIGNED | ✅ 已插入 05 §5.1 |
| Mamba Modulation venue 升级机会 | bib 阶段采用 NeurIPS 2025 正式版 DOI:10.52202/0857-0653(2026-09-13 核出),正文占位符不变 | ✅ 记录在案,LaTeX bib 采用 |
| 03 清单勘误 3 处(From-data-compat TPAMI 误标 TIP、SAM-drop 双行、MUSIQ ICCV) | 按 03 纪律不改 03 本体;英文稿未引三者,无影响;勘误已留档于本文件遗留区 L8 | ✅ 已留档 |
| GatedFusion-Net 年份 | 定版 **Information Fusion 2026**(Crossref print 2026 年 5 月 vol 129;2025 为在线先发年) | ✅ 02/06 章及 bib 均按 2026 |

**终稿 bib 底稿口径**:正式清单 = 41 条 − MemMamba(#18)− ND-CNPFuse(#36)+ Ma et al. 2021(#42)= **40 条**,全部正式 venue(T2 清零),低于 45–55 目标下限——增补候选池见本文件 §6,是否增补留 gate-2 终检决策(英文稿按 TVC 常规体量 40 条已可支撑,不作硬性凑数)。
