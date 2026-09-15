# Abstract (English) + Chinese Abstract (Author's Record) + Keywords

> Status: DRAFT for gate-2. English abstract contains NO numeric magnitudes (K5 blowback clause, `00-PAPER-CONFIG.md` 纪律 1/3): qualitative descriptions only for QABF/VIF, mandatory boundary sentence included verbatim. All claims cross-checked against GATE1_FREEZE.md and writing/01-02 drafts. Chinese abstract is an author's record only (作者留档) and is NOT part of the English submission package.

## English Abstract

At pixels where infrared and visible images disagree, mainstream fusion networks resolve the conflict by blending — weighted averaging in feature space — and any blend cancels the structure that committing to the salient modality would preserve. For visual perception systems that fuse to detect, the objects lost in this cancellation matter. This paper first shows, through a pre-registered, same-budget falsification on the selective state-space backbone S4Fusion, that the prevailing alternative does not fix this: feature-space modulation of the state dynamics, motivated by state-capacity dilution, does not improve structural fusion quality under matched training budget. The residual failure lies outside the state, in the absence of an arbitration decision. We therefore introduce Conflict-Gated Arbitration (CGA), which detects per-pixel cross-modal conflict from image gradients and applies a learned, zero-initialized, image-space per-pixel commitment toward the locally salient modality, acting only where the modalities conflict, so that loading the official checkpoint reproduces the baseline exactly. Across three seeds, the primary criterion replicates: gains on high-conflict object recall are statistically significant yet below the stricter pre-registered thresholds, and significant structural costs in SF and AG accompany them. On full-reference quality the method attains the leading QABF among the eight unified-evaluation baselines and a statistical tie on VIF, though SF and MI remain below the official released weights. All numbers come from one frozen evaluation; training-free pre-checks, mechanism visualizations and a registered distillation failure are reported alongside the gains. Code and protocols are released for full reproduction.

<!-- A-1/A-4 revision 2026-09-14: (i) the full-reference sentence is now symmetric —
     it reports the SF/MI deficit against the official released weights instead of
     naming the QABF gain alone; (ii) "eight unified-protocol baselines" ->
     "eight unified-evaluation baselines" to stop implying matched training;
     (iii) meta-language trimmed ("the pre-registered primary criterion" ->
     "the primary criterion"; "pre-registered protocols" -> "protocols";
     "under the registered parameterization" -> "under matched training budget").
     K5 blowback clause still honoured: no numeric magnitude anywhere in the abstract.
     NOTE: this text is now byte-aligned with main.tex L54 except for dash glyphs
     (md em dash vs LaTeX ---), after the two had drifted (the .tex still carried
     "stays within 0.0006 of the best VIF", which violated the zero-number rule). -->


<!-- src: 全段定性主张逐项对照 GATE1_FREEZE.md §1（QABF 第一/平手）、§2（SF/AG 负向、方向一致）、§3（H5 三种子复现+更严门未过）、§4（门控饱和）；boundary 句式=fixed phrasing（"yet" 变体）；无任何具体涨幅数字。 -->

**Word count: 218.**

### Keywords (English)

infrared–visible image fusion; conflict arbitration; selective state-space models; zero-initialized gating; image-space per-pixel commitment; pre-registered ablation

<!-- 6 keywords; 术语红线合规:无 "per-pixel modality selection"、无裸 "decision-level"。 -->

---

## 中文摘要(简体,作者留档,不入英文投稿包)

红外与可见光图像逐像素冲突处,主流融合以加权混合消解,抵消了承诺显著模态所能保留的结构。预注册同预算证伪表明:状态空间主干 S4Fusion 上,特征空间状态调制不改善结构指标——瓶颈不是状态,而是仲裁的缺失。据此提出冲突门控仲裁(CGA):图像梯度检测跨模态冲突,零初始化门控在图像空间逐像素承诺显著模态,仅冲突区生效,官方权重复现基线。三种子均复现预注册主判据:高冲突检测增益统计显著但低于更严门限,伴 SF/AG 结构代价;QABF 居八基线之首,VIF 统计平手。代码与协议公开。

**字数:纯汉字 193 字(满足 150-220 留档规格);含标点与空格共 244 字符。**

### 关键词(中文)

红外-可见光图像融合;冲突仲裁;选择性状态空间模型;零初始化门控;图像空间逐像素承诺;预注册消融

---

## Notes for gate-2 integration

**数字溯源**:英文摘要与中文摘要均**零具体数字**(K5 反噬条款:涨幅数字一律不入摘要)。定性主张逐一对照 GATE1_FREEZE.md:"leading QABF among eight unified-protocol baselines"=§1 QABF 0.5449 全表第一、8 基线;"statistical tie on VIF"=§1 官方权重 0.4628 vs CGA 0.4622(−0.0006 平手);"statistically significant yet below the stricter pre-registered thresholds"=§3 档位判定固定句式("yet" 为任务书指定变体,正文 06 章用 "but");"structural costs in SF and AG"=§2 −0.0713/−0.1295(0+/3−);"registered distillation failure"=§5 B2。中文摘要边界句与英文同口径。

**与 01 章 claim-evidence map 呼应**:摘要四段式(问题→发现→方法→证据)镜像 01 章一句话论证;"the bottleneck is not the state; it is the absence of an arbitration decision" 直接承接 01 章机制段结论句;zero-init 官方权重复现=claim-evidence map "Zero-initialized gate preserves official weights" 行。摘要不含 "first/SOTA" 断言("first" 仅限 02 章负检索句原文)。

**潜在措辞冲突(需 gate-2 终检)**:
1. **"eight unified-protocol baselines"**:GATE1 §1 主表为 8 基线行(MetaFusion/CDDFuse/EMMA/DCEvo/W-Mamba/Diff-IF/FusionMamba + 官方权重行),S4Fusion 官方权重按主表计为一行基线;终稿若改计法(如官方权重不计入"基线"而称"reference row")须同步改本句为 "seven baselines"。
2. **"selective state-space models" vs "selective scanning"**:关键词第 3 项从检索性考虑选了 state space models;若 04-methods 章术语最终定为 "selective scanning" 一系,建议关键词增删对齐(二选一即可,勿同时出现两个变体)。
3. 中文摘要留档版以纯汉字计 193 字(含标点与空格共 244 字符);中文字数统计口径(是否含标点/字母缩写)由 gate-2 终检统一,当前纯汉字口径落在 150-220 区间。
