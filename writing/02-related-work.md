> **⚠️ 权威源声明（2026-09-20 起，替代 2026-09-19 版）**
> 本目录是**历史写作源，已完成使命**。自 2026-09-20 的 Round-3 门控式全面修订起，本目录与正文**完全脱钩**：`main.tex` 的标题、摘要、引言、相关工作、方法、实验、讨论、局限与结论均已按"聚类感知统计重分析 + 机制操作定义校正 + 门控式路线判定"重写，本目录保留的是修订前的草稿。
> **稿件的唯一权威正文是 `latex/sn-article/main.tex`**（+ `refs.bib` + `figs/`），交付件由 `code/build_submission.sh` 产出到 `latex/submission/`。
> **禁止从 `writing/` 重新装配正文**——否则会把已作废的数字与措辞带回来，例如 §3.5 旧时延 `230.4/791.1 ms`、`n=900` 池化被当作主要跨 seed 证据、`hard commitment` 旧术语、以及 `manual conflict map` 旧称谓。本目录仅作写作史与溯源用。

# Related Work — Classified Draft (K8)

> Status: DRAFT for gate-2. Classification: CNN → Transformer → Diffusion → Mamba/SSM → Semantic-guided → Boundary/uncertainty. Every family ends with the *technical reason* its global consistency at modal boundaries fails — the K8 requirement ("每类点出全局一致性技术原因"). Citations from literature report (DOIs verified 2026-08-13).
> Style basis: nature-writing related-work fragment (group by mechanism, each subsection ends with limitation this paper addresses).

---

## 0. Classification map (why this order)

| Family | Mechanism | Why boundary/global consistency fails |
|---|---|---|
| CNN | local receptive field, multi-scale pyramids | local receptive field → boundary conflict resolved by local averaging; global consistency only via depth/pyramid |
| Transformer | global self-attention over patches | global but mixes all tokens equally at boundaries; O(n²) cost hurts high-res; correlation-based decomposition (CDDFuse) still blends |
| Diffusion | generative denoising/prior | strong global prior but per-pixel generative *mix*; slow sampling; task-agnostic priors ignore conflict decision |
| Mamba/SSM | linear-time selective scan | linear-time global context, but bounded state capacity → long-sequence dilution (inability to manage what the state retains); scan topology ≠ decision |
| Semantic-guided | task-level semantics as extra loss/supervision | improves target awareness but guides *weights*, not the decision of which modality to trust at a conflicting pixel |
| Boundary/uncertainty/conflict (this paper's neighborhood) | edge priors, uncertainty maps, conflict modeling, per-pixel weighting | existing work coordinates conflict in *feature space* with *soft* weights (ARC3Fusion routing, GatedFusion-Net Sigmoid); BRSS/CGA arbitrates in *image space* with a *hard, zero-initialized* commitment |

---

## 1. CNN-based fusion

Early fusion networks build on autoencoder and dense designs — DenseFuse [TIP 2019] with its DenseNet encoder and the unified dense architecture FusionDN [AAAI 2020] — later unified under an unsupervised, application-agnostic training scheme in U2Fusion [TPAMI 2022]. These models achieve strong local structure preservation, but their receptive fields grow only with depth; at modality boundaries the network is forced to resolve conflict by local averaging, and global structural consistency is achieved (if at all) by stacking layers or scales rather than by an explicit decision about which modality to trust. This is the technical reason CNN fusion degrades in global consistency: the operator has no mechanism to *prefer* one modality at a conflicting pixel; it can only mix local statistics.

## 2. Transformer-based fusion

Self-attention brought global context to fusion: CDDFuse [CVPR 2023] decomposes common and modality-specific features with correlation-driven branches and restores them with a Transformer; task-driven and meta-learning formulations such as MetaFusion [CVPR 2023] adapt fusion to downstream tasks; EMMA [CVPR 2024] adds equivariant self-supervision. Attention computes a weighted *average* over all tokens, so a conflicting boundary pixel receives a blend of both modalities' evidence; moreover, quadratic complexity in token count makes full-resolution global attention expensive exactly in the high-resolution regime we target. Decomposition helps, but the output is still a weighted sum — no commitment is ever made.

## 3. Diffusion-based fusion

Diffusion models formulate fusion as iterative denoising toward a joint distribution, e.g., DDFM [ICCV 2023] and Diff-IF [Information Fusion 2024]. They capture strong global priors and are robust to modality mismatch, but the generative process *averages* modality evidence under the prior at every step — there is no mechanism to hard-commit to one modality where they disagree — and sampling cost (dozens of steps) conflicts with the deployment-efficiency goal. Their global consistency is therefore strong but slow and still non-committal at conflict pixels.

## 4. Mamba / SSM-based fusion

S4Fusion [TIP 2025] was the first to fuse infrared–visible images with selective state spaces, followed by wavelet-domain Mamba (W-Mamba [ICME 2025]) and Mamba-based multi-modal fusion (FusionMamba [Visual Intelligence 2024]). These inherit linear-time global context from the SSM family [Mamba; Mamba-2]. The technical limitation is state capacity: recent analyses show that the recurrent state, when overloaded by long sequences, loses its ability to manage what it retains [Stuffed Mamba, COLM 2025], and that only the A/Δ parameters control history decay [Hidden Attention of Mamba; Mamba Modulation]. SSM fusion thus has global context but a bounded, dilutable state — and, like all scan-based operators, it *scans and blends* rather than deciding. The 2026 wave continues to modify **scan topology** rather than the decision: spatial-frequency interactive scanning [Interactive Spatial-Frequency Fusion Mamba, TIP 2026], random shuffle scanning [Shuffle Mamba, TCSVT 2026], and self-supervised consensus among experts [SMC-Mamba, AAAI 2026] — all of which reorganize *where the scan looks*, not *what the state should commit to* at a conflicting pixel. Our own controlled ablation (§Methods, H1) shows that modulating the state dynamics (B/C/Δ) with boundary reliability does **not** improve structural metrics — the bottleneck is not the state, it is the absence of an arbitration decision.

## 5. Semantic- and task-guided fusion

Task-awareness improves fusion for downstream perception: SeAFusion [Information Fusion 2022] distills semantic information into fusion; SegMiF / M2Fusion [ICCV 2023] couple segmentation with fusion; MetaFusion [CVPR 2023] adapts via meta-learning; KD-based lightweight variants exist [Electronics 2023; IEEE Access 2025]. These methods guide *where* to look (task semantics) but still resolve conflicting pixels by weighting features; they improve target recall but do not solve the decision problem, and their semantic branches need task labels/extra networks at inference.

## 6. Boundary-aware, uncertainty-guided, and conflict-aware fusion

Boundary and uncertainty cues are an active, growing direction: boundary-aware uncertainty fusion for medical segmentation (BUFNet [MedIA 2026]); boundary-region selective scanning for shadow removal (ShadowMamba [IVC 2026]); edge-prior-guided IVIF [IPT 2025]; edge-guided dual attention [JEI 2023]; uncertainty-aware cross-modal fusion for detection [DICTA 2024]. Closest to our setting, ARC3Fusion [Sensors 2026] explicitly models consensus–complementarity–**conflict** in IVIF — its conflict estimation even includes edge-strength and orientation mismatch — and GatedFusion-Net [Information Fusion 2026] computes per-pixel modality weights. These confirm that conflict and per-pixel weighting are recognized concerns, yet both operate as *continuous, feature-space soft coordination*: ARC3Fusion routes verified residuals in feature space with a shared encoder, fusing them through per-position three-branch Softmax weights that sum to one, and GatedFusion-Net learns per-pixel Sigmoid confidence masks that gate the thermal and ultraviolet streams additively onto its RGB+DIN base for semantic segmentation. Image-space per-pixel **hard** selection has a long but separate lineage: the classical select/choose fusion rules [Burt & Kolczynski 1993; Zhang & Blum 1999; Piella 2003] formalized "choose the salient source where they disagree" for pyramid fusion, and the learning-based multi-focus decision-map line [e.g., Ma et al., Neurocomputing 2021] still makes per-pixel binary decisions — but driven by defocus in *same-modality* pairs, unrelated to cross-modal conflict. Learned *discrete* routing has also entered IVIF: a concurrent preprint [Ma et al., SSRN 2026, DOI:10.2139/ssrn.7360275] selects, per pixel, among copy-infrared, copy-visible, and attention-fusion experts via a Gumbel-Softmax straight-through estimator — the closest existing instance of per-pixel hard choice in cross-modal fusion, yet it operates on invertible-network-decoupled feature components (feature space, not image space), and its routing signal is learned informativeness rather than cross-modal gradient conflict. **To the best of our knowledge, no prior IVIF method converts cross-modal gradient-orientation/polarity conflict into a learned, image-space, per-pixel hard-commitment arbitration.** Our distinction is therefore positional, not incremental: existing conflict modeling is feature-space and soft; existing discrete routing is feature-space and conflict-agnostic; existing hard selection is same-modality and defocus-driven; we are not aware of any prior method that places a learned hard commitment at cross-modal conflict pixels in image space — through a zero-initialized gate that preserves the official S4Fusion checkpoint exactly at load time.

**Table (D-2): mechanism axes for conflict-aware fusion.** <!-- src: 手册 §5-2E; full-text verification record writing/02-related-work.rev-log-20260913.md; concurrent-work label per the G3/G9 closure note at the end of this file -->

| Axis | ARC3Fusion | GatedFusion-Net | Ma et al.\ (conc., SSRN 2026) | **CGA (ours)** |
|---|---|---|---|---|
| Operating space | feature | feature | feature | **image** |
| Conflict signal | edge strength & orientation mismatch (explicit) | learned modality confidence (implicit) | learned expert informativeness (implicit) | **cross-modal gradient orientation & polarity** |
| Decision type | soft: three-branch Softmax, weights sum to one | soft: per-pixel Sigmoid masks, applied additively | discrete: Gumbel–Softmax straight-through routing | **hard per-pixel commitment** |
| Zero-init.\ compatible | no | no | no | **yes: exact identity at load** |
| Detector endpoint reported | no | no (segmentation) | no | **yes: recall and mAP50** |

The table states the distinction on axes rather than in prose, and it is the axis distinction --- not the gain magnitude --- that the paper claims. One row is more honest than the others under route B: the trigger axis (conflict-triggered selectivity) is the part that the content controls go on to falsify, so the table is reported together with that result (§5.8 content controls) rather than as a purely favourable positioning claim. <!-- src: GATE1_UNFREEZE §2; docs/route_B_outline_20260916.md §4-L10 -->

---

## Notes for the final pass

- Keep 4–5 citations per subsection; merge CNN/Transformer if the journal limits space.
- Section 4 must cite our H1 ablation (paper's own table) — unique among baselines.
- Section 6 is where the novelty lands: "feature-space soft coordination" (ARC3Fusion, GatedFusion-Net) vs "same-modality defocus-driven hard selection" (decision-map line) vs **"image-space, conflict-triggered, learned hard commitment"** (ours).
- **Terminology discipline (2026-09-09)**: avoid "per-pixel modality selection" (contested by GatedFusion-Net's title) and bare "decision-level" (collides with the classic decision-level fusion term, cf. Dc-EEMF TBME 2026); use "**image-space per-pixel commitment/arbitration in conflict regions**" throughout.
- **Citation-status flags (updated 2026-09-13, G3/G9 closed)**: ARC3Fusion characterization is now **full-text verified** — G9 closed (method §2.4.4 Conflict-Aware Routing via Europe PMC PMC13469189 full XML: per-position three-branch Softmax with W_c+W_ir+W_vi=1, plus Sigmoid branch gates; full-text negative scan found zero argmax/binary/hard-selection; LLVIP/MSRS/TNO datasets and edge-strength + orientation-mismatch conflict terms confirmed). GatedFusion-Net characterization is **abstract-level verified** — G3 closed at abstract level (OpenAlex journal abstract: per-pixel Sigmoid gating, T/UV added onto RGB+DIN base, semantic segmentation on MM5; journal title self-corrects "selection"→"weighting", reinforcing our terminology discipline); **one residual**: method-section gate formula not verbatim verified (ScienceDirect CAPTCHA blocked; SSRN/institutional copies unreachable) — upgrade to full-text level if institutional access becomes available, does not affect the SOFT-ONLY verdict. All DOIs verified 2026-09-08/09/13; re-run citation-check before submission. Full verification record: `02-related-work.rev-log-20260913.md`.