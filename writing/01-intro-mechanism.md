# Introduction — Mechanism Paragraph (English Draft)

> Status: DRAFT for gate-2 rewrite. Numbers cited are from real logs (`eval_H1.log`, `eval_c256.log`, `eval_cga.log` [2026-07-22/23]; `eval_str_cga.log` [2026-08-25]; `arb_detect_full300.log`/`full300b.log` [2026-08-14]). Narrative follows protocol: state-dilution hypothesis → controlled falsification → conflict-arbitration mechanism. CGA quantitative claim is **no longer pending**: strengthened training (32ep, commit-weighted loss + gate LR) completed 2026-08-25 with a statistically significant high-conflict recall gain that, however, sits **below the stricter pre-registered gate thresholds** (C1/C2/C3) — the final text must report both facts.
> Style basis: nature-writing (research paper, technical-challenge variant), target: English IVIF journal.

---

## One-sentence argument

> In infrared–visible image fusion, we show that feature-space modulation of selective state-space dynamics (B/C/Δ), motivated by state-capacity dilution, **does not** transfer to structural fusion gains under controlled same-budget ablation; we instead identify the residual failure as an **image-space conflict-arbitration problem** and introduce **Conflict-Gated Arbitration (CGA)** — a zero-initialized, per-pixel commitment toward the salient modality applied only where modalities conflict — which preserves official-weight compatibility and improves downstream detection on high-conflict objects.

---

## Draft (introduction mechanism paragraph, ~330–380 words)

Selective state-space models (SSMs) have recently become the backbone of infrared–visible image fusion because their linear-time global receptive field scales better than self-attention on large inputs [S4Fusion, Mamba]. Yet their usefulness at high resolution is constrained by a fundamental capacity problem: the recurrent state has bounded capacity, and long sequences dilute the information each state slot can retain — when the contextual information exceeds the state capacity, the model even fails to learn what to forget [Stuffed Mamba, COLM 2025]; the state transition is further controlled by the spectral properties of the decay matrix, so that poorly managed states drift or vanish over long horizons [Mamba Modulation; Hidden Attention of Mamba]. This motivated us to hypothesize that fusion failures at high resolution originate *inside* the SSM state dynamics — specifically, that injecting per-pixel boundary reliability into the selective parameters **B**, **C** and **Δ** would re-allocate state capacity toward modality boundaries where cross-modal conflict is largest (Boundary-Reliability State-Space Fusion, BRSS).

We tested this hypothesis under strictly controlled conditions: every configuration was fine-tuned from the same public S4Fusion weights with identical budget (crop 128, 20 epochs, seed 42, same data), and evaluated with the same tiled protocol on M3FD. The result was a falsification. Relative to the same-budget baseline, the state-modulated model significantly *degraded* structural metrics (SF −0.019, AG −0.013, QABF −0.0005, Holm-corrected p < 0.05), gained only mutual information (+0.021), and did not reduce cross-scale drift on any of five resolutions (drift-AUC delta +0.0000, 0/5 resolutions improved). Feature-space modulation of SSM dynamics, despite its theoretical appeal, does not yield structural fusion gains; the bottleneck is not state capacity.

The failure is instead in the arbitration decision. Where the two modalities disagree — opposite edge orientation or polarity at the same pixel — any *blend* (weighted average in image or feature space) cancels structure, whereas *committing* to the locally salient modality preserves it. We verified this in a training-free pre-check on M3FD detection: committing toward the max-response modality in conflict regions recovers objects that blending loses. We therefore introduce **Conflict-Gated Arbitration (CGA)**, which (i) detects per-pixel modal conflict from image gradients, (ii) learns a per-pixel commitment toward the salient modality, and (iii) applies it **only in conflict regions** through a zero-initialized gate, so that loading the official S4Fusion checkpoint reproduces the baseline exactly and the design is compatible with prior weights [LayerScale]. Beyond avoiding the artifact tax of global blending, CGA localizes intervention to the pixels where a decision is genuinely required — a qualitatively different mechanism from feature-space modulation, and one that directly targets the objects downstream detection systems lose.

---

## Section outline (full intro, 4 paragraphs)

1. **Field stake** — fusion for downstream perception; SSMs as new backbone; linear-time global context.
2. **Bottleneck** — bounded state capacity + long-sequence dilution + modal boundary conflict at high resolution (this paragraph above).
3. **Prior work gap** — blending families (CNN/Transformer/Diffusion) and SSM baselines all *mix* at boundaries; none arbitrate (see related-work draft).
4. **Present study** — CGA mechanism, zero-init compatibility, contributions list; honest boundary: high-conflict recall gain is significant on all three seeds (p=4.88e-04 / 4.34e-03 / 4.34e-03) yet below the stricter pre-registered thresholds (+0.02 recall / mAP50 +0.01); AG structural non-inferiority passes on 2/3 seeds with s42 the borderline exception (−1.00%).

---

## Assumptions / missing inputs

- **CGA quantitative claim updated (2026-08-25, `eval_str_cga.log`)**: strengthened CGA (32 epochs, w_commit 10 + gate_lr 0.01) on the full 300-image M3FD detection protocol gives high-conflict recall 0.4385 → 0.4579 (**+0.0194**, recovered=20 lost=3, exact binomial/McNemar-style **p=4.88e-04**, significant); recall all-objects +0.0120; mAP50 0.7197 → 0.7282 (+0.0085). **Honest boundary — the stricter pre-registered gates were NOT met**: gate-C1 (+0.02 high-conflict recall threshold: 0.0194 < 0.02), gate-C2 (mAP50 delta +0.0085 < +0.01), gate-C3 (AG structural non-inferiority fails at exactly −1.00%; SF passes at −0.85%; EN passes). The draft's claims below are scoped to what the numbers support: *significant, direction-consistent gains on the primary endpoint, with the residual gap to the stricter thresholds reported as such* — never "passed the pre-registered gates." **Three-seed robustness completed 2026-09-12** (`seeds_table.md`): H5 replicates on all three seeds (+0.0194/+0.0159/+0.0159, all p<0.005); all six quality metrics sign-consistent 3/3; gate-C3 **now passes on 2/3 seeds** (s123 SF −0.39%/AG −0.60%; s3407 −0.21%/−0.36%) with s42 the AG borderline exception (−1.00%). See `GATE1_FREEZE.md` §3 for the frozen verdict wording.
- **Falsification numbers are real and frozen**: SF −0.0190 / AG −0.0126 / QABF −0.0005 / MI +0.0213 (M3FD, 300 pairs, Holm); drift-AUC B0=S=0.7895, 0/5 resolutions. These stay in the paper as a registered negative result (protocol §1: "先改叙事,不改数据").
- "High resolution" claim: native high-res data limited; test_high_resolution.csv (LLVIP) exists as pressure axis; wording above avoids specific resolution promises.
- Paragraph length target ~350 words for journal; trim in final pass.

---

## Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| State capacity of SSM is bounded and diluted over long sequences | Stuffed Mamba (COLM 2025, arXiv:2410.07145; v4 title "Oversized States Lead to the Inability to Forget" — **term check 2026-09-09: "state capacity" retained in v4, "state collapse" dropped; use "state capacity" + "inability to forget", never "state collapse"**); Mamba Modulation (arXiv:2509.19633); Hidden Attention (ACL 2025) | supported (literature) |
| State-modulated BRSS (B/C/Δ) fails to improve structural metrics vs same-budget baseline | eval_H1.log: SF −0.0190 p=2.90e-44; AG −0.0126 p=8.13e-40; QABF −0.0005 p=5.63e-27; MI +0.0213 p=1.09e-17; params +15048 (0.42%) **[freeze-status flag 2026-09-13: param count NOT in GATE1_FREEZE — omit from manuscript or add explicit freeze line before use]** | supported (own controlled ablation) |
| No cross-scale drift improvement for the H1 (feature-space) arm (D1 stress axis) | eval_c256.log: drift-AUC B0=S=0.7895; 0/5 resolutions; p=9.45e-01. **Arm note (2026-09-13)**: H1-era F3 numbers — NOT in GATE1_FREEZE; if quoted in the manuscript they need their own freeze line, else omit from the final text (the frozen CGA-arm F3 numbers in GATE1 §5 are the paper's cross-scale evidence). Distinct from CGA arm: B0 0.8088→CGA 0.7839, 4/5 resolutions | supported (own; freeze-status flag) |
| Blending cancels conflict structure; committing recovers detection | arb experiments (training-free pre-check, 90-image split): A3_max mAP50 0.6355, A2_contrast 0.5884 vs B0cmp 0.7207; note A3_max/A2 below B0cmp on aggregate mAP — the pre-check claim is scoped to recovered-vs-lost objects in conflict regions, not aggregate mAP | partially supported (training-free pre-check, smoothed) |
| CGA improves downstream detection on high-conflict objects | eval_str_cga.log (full 300, 2026-08-25): high-conflict recall 0.4385→0.4579 (+0.0194), recovered=20 lost=3, p=4.88e-04; mAP50 0.7197→0.7282 (+0.0085) | **supported (significant)**; below stricter pre-registered gates C1 (+0.02) / C2 (mAP50 +0.01) — report as such |
| Zero-initialized gate preserves official weights | LayerScale (ICCV 2021); architecture: tanh(0)=0 → output==baseline | supported (design) |

---

## Why S4Fusion as the testbed (K7, added 2026-09-13)

> Old rationale ("B/C/Δ are the mechanistic entry point into SSM dynamics") was invalidated by the H1 falsification and is retired. The rewrite below is what the paper carries; every clause traces to a frozen number or a design fact.

1. **Representative of the scan-and-blend operator class.** The question this paper asks — *does feature-space modulation of SSM state dynamics fix fusion at boundaries?* — is a question about an operator class (linear-time scan backbones that mix modal representations), not about one model. S4Fusion [TIP 2025] is the representative open, published scan-and-blend SSM fusion backbone; testing the hypothesis there makes the (negative) answer generalize to the class, which is exactly what a falsification study requires. The main table now carries 8 unified-evaluation baselines across CNN / Transformer / diffusion / SSM families (`SOTA_MAIN_TABLE`), so S4Fusion additionally anchors the SSM row of that comparison.
2. **A public checkpoint is what makes the falsification *controlled*.** Because official weights are released, every arm of every ablation (B0 retrain, B/C/Δ modulation, CGA) fine-tunes from the *same* starting point with the *same* budget — any metric delta is attributable to the intervention, not to initialization or training-set differences. Without public weights, the H1 negative result would be an anecdote; with them, it is a controlled falsification under a pre-registered protocol (`机制消融实验协议.md`).
3. **Zero-initialized compatibility validates the arbitration layer *against* the weights it modifies.** CGA's gate starts at tanh(0)=0, so loading the official checkpoint reproduces the baseline output exactly; the trained gate then saturates (tanh(gate_scale)=0.9998, frozen `cga_vis/summary.csv`), showing the mechanism is genuinely learned rather than left off. The same-budget retrain row and the official-weights row of the main table (provenance-corrected 2026-09-10, commit 749bbb0) exist only because the official model is public.

*Deliberately not claimed:* that S4Fusion is the best fusion model (the frozen table shows EMMA leading SF/AG and Diff-IF leading MI — reported as-is); or that the choice was mechanistically privileged (it was, before H1; that reason is dead and stays dead).

---

## Why this structure

- **Honest falsification first**: per protocol, H1/F3 negative results are reported as registered findings, converting a weakness into a methodological strength (controlled same-budget protocol).
- **Pivot grounded in a pre-check, now reinforced by learned-CGA evidence**: the max-response-commit pre-check (training-free) established the direction; the strengthened learned CGA now provides the statistically significant high-conflict recall gain (p=4.88e-04) — while still below the stricter pre-registered thresholds, which the intro reports as an honest boundary rather than a pass.
- **Literature anchored**: every mechanism sentence has a verified DOI/arXiv citation from the literature report (no orphan claims).
