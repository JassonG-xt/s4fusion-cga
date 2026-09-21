> **⚠️ 权威源声明（2026-09-20 起，替代 2026-09-19 版）**
> 本目录是**历史写作源，已完成使命**。自 2026-09-20 的 Round-3 门控式全面修订起，本目录与正文**完全脱钩**：`main.tex` 的标题、摘要、引言、相关工作、方法、实验、讨论、局限与结论均已按"聚类感知统计重分析 + 机制操作定义校正 + 门控式路线判定"重写，本目录保留的是修订前的草稿。
> **稿件的唯一权威正文是 `latex/sn-article/main.tex`**（+ `refs.bib` + `figs/`），交付件由 `code/build_submission.sh` 产出到 `latex/submission/`。
> **禁止从 `writing/` 重新装配正文**——否则会把已作废的数字与措辞带回来，例如 §3.5 旧时延 `230.4/791.1 ms`、`n=900` 池化被当作主要跨 seed 证据、`hard commitment` 旧术语、以及 `manual conflict map` 旧称谓。本目录仅作写作史与溯源用。

# Abstract (English) + Chinese Abstract (Author's Record) + Keywords

> Status: DRAFT for gate-2. English abstract contains NO numeric magnitudes (K5 blowback clause, `00-PAPER-CONFIG.md` 纪律 1/3): qualitative descriptions only for QABF/VIF, mandatory boundary sentence included verbatim. All claims cross-checked against GATE1_FREEZE.md and writing/01-02 drafts. Chinese abstract is an author's record only (作者留档) and is NOT part of the English submission package.

## English Abstract

<!-- Route-B rewrite 2026-09-16. The abstract no longer claims that the primary
     criterion replicates: under the budget-matched base convention it did not
     (k = 1 of 3 seeds, 1 of 2 effective independent readings; GATE1_UNFREEZE
     §3.1/§3.8). It now leads with the two falsifications — the state-dynamics
     diagnosis, then the mechanism's unstable benefit — and reports the negative
     result as the finding rather than as a caveat. The selectivity clause ("acting
     only where the modalities conflict") was removed from the mechanism sentence
     because B-1 showed the conflict-triggered restriction adds nothing at either
     endpoint (§2). The quality-leaderboard sentence was dropped for the same
     reason: under this framing the paper does not lead with a gain. No digits:
     quantities are spelled out in words, and the only numeral is the proper noun
     S4Fusion. 248 words, inside the 250-word cap. -->

At pixels where infrared and visible images disagree, mainstream fusion networks resolve the conflict by blending — weighted averaging in feature space — and any blend cancels the structure that committing to the salient modality would preserve. For visual perception systems that fuse to detect, the objects lost in this cancellation matter. This paper answers with a falsification study, not a leaderboard entry. We first falsify the received diagnosis: on the selective state-space backbone S4Fusion, modulating the state dynamics in feature space, motivated by state-capacity dilution, does not improve structural fusion quality under a matched training budget. We then build the mechanism that diagnosis implies, Commitment-Gated Arbitration (CGA), which detects per-pixel cross-modal conflict from image gradients and applies a learned, zero-initialized, image-space per-pixel commitment toward the locally salient modality, so that loading the official checkpoint reproduces the baseline exactly. The mechanism is demonstrably online, but its benefit is not stable: under a second, budget-matched baseline convention the primary criterion is met in only one of three seeds, one of two effective independent readings, and significant SF and AG costs accompany the gains that do appear. Inference-time content controls show why: committing everywhere does at least as well as committing only at conflicts, and destroying the map's spatial arrangement changes nothing. What survives is a mechanism that is implemented, verified and falsified as a source of reliable gain. Training-free pre-checks, mechanism visualizations and a distillation failure are reported alongside; code, protocols and per-image results are released.

<!-- A-1/A-4 revision 2026-09-14: (i) the full-reference sentence is now symmetric —
     it reports the SF/MI deficit against the official released weights instead of
     naming the QABF gain alone; (ii) "eight unified-protocol baselines" ->
     "eight unified-evaluation baselines" to stop implying matched training;
     (iii) meta-language trimmed ("the primary criterion" ->
     "the primary criterion"; "protocols" -> "protocols";
     "under this parameterization" -> "under matched training budget").
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

**数字溯源**:英文摘要与中文摘要均**零具体数字**(K5 反噬条款:涨幅数字一律不入摘要)。定性主张逐一对照 GATE1_FREEZE.md:"leading QABF among eight unified-protocol baselines"=§1 QABF 0.5449 全表第一、8 基线;"statistical tie on VIF"=§1 官方权重 0.4628 vs CGA 0.4622(−0.0006 平手);"statistically significant yet below the stricter thresholds"=§3 档位判定固定句式("yet" 为任务书指定变体,正文 06 章用 "but");"structural costs in SF and AG"=§2 −0.0713/−0.1295(0+/3−);"registered distillation failure"=§5 B2。中文摘要边界句与英文同口径。

**与 01 章 claim-evidence map 呼应**:摘要四段式(问题→发现→方法→证据)镜像 01 章一句话论证;"the bottleneck is not the state; it is the absence of an arbitration decision" 直接承接 01 章机制段结论句;zero-init 官方权重复现=claim-evidence map "Zero-initialized gate preserves official weights" 行。摘要不含 "first/SOTA" 断言("first" 仅限 02 章负检索句原文)。

**潜在措辞冲突(需 gate-2 终检)**:
1. **"eight unified-protocol baselines"**:GATE1 §1 主表为 8 基线行(MetaFusion/CDDFuse/EMMA/DCEvo/W-Mamba/Diff-IF/FusionMamba + 官方权重行),S4Fusion 官方权重按主表计为一行基线;终稿若改计法(如官方权重不计入"基线"而称"reference row")须同步改本句为 "seven baselines"。
2. **"selective state-space models" vs "selective scanning"**:关键词第 3 项从检索性考虑选了 state space models;若 04-methods 章术语最终定为 "selective scanning" 一系,建议关键词增删对齐(二选一即可,勿同时出现两个变体)。
3. 中文摘要留档版以纯汉字计 193 字(含标点与空格共 244 字符);中文字数统计口径(是否含标点/字母缩写)由 gate-2 终检统一,当前纯汉字口径落在 150-220 区间。
