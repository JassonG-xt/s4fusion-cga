# Image-Space Commitment Arbitration for Infrared–Visible Image Fusion — code and per-image results

Companion repository for the manuscript *"Image-Space Commitment Arbitration for Infrared–Visible Image Fusion: A Controlled Falsification and Mechanism-Audit Study"* (submitted to **The Visual Computer**).

This repository releases the code, the per-image and per-object evaluation results, the cluster-aware re-analysis and the pre-registration ledgers behind every number reported in the paper.

**This is the final review release, `v1.1-final-review-20260921` (21 September 2026).** It supersedes the earlier tag [`v1.0`](https://github.com/JassonG-xt/s4fusion-cga/releases/tag/v1.0) (15 September 2026). What each tag covers is stated precisely in §5.

---

## 1. Contents

| Path | What it is |
|---|---|
| `latex/submission/main.pdf` | **The manuscript as submitted to this round of review** |
| `latex/sn-article/main.tex`, `refs.bib`, `figs/` | LaTeX source and figure files of the manuscript |
| `code/*.py`, `code/*.sh` | Training, evaluation, statistical-analysis and figure scripts |
| `code/modules/` | Model modules, including the image-space commitment-arbitration implementation (`cga.py`) and the boundary-reliability estimator (`boundary.py`) |
| `code/cluster_audit_final.py` | **The cluster-aware re-analysis behind the paper's main detection table** (object → image → seed); asserts the frozen population (2583 objects, 878 high-conflict objects over 181 images) and fails loudly otherwise |
| `code/results/cluster_audit/` | Reference output of that script, plus the `object → image → seed` mapping (`object_image_mapping.csv`) and the arm pairs it analyses (`seed_arm_map.csv`) |
| `code/results/**/*.csv` | **Per-image and per-object metric CSVs (89 files)** — the raw basis of the main table, the cross-seed tables, the detection endpoints, the content controls, the colour audit and the selector statistics |
| `writing/` | Manuscript chapter sources (abstract, introduction, related work, methods, experiments, discussion, references) |
| `GATE1_FREEZE.md` | **Frozen results ledger** — the single source of truth for every number in the paper |
| `GATE1_AMEND_20260914.md` | Explicit amendments to the frozen ledger (no frozen value changed) |
| `GATE1_PREREG_B3_20260914.md` | Before-run pre-registration of the decision criteria for the budget-matched control |
| `机制消融实验协议.md` | The pre-registered ablation protocol (hypotheses, decision criteria, wording discipline) |
| `IMPLEMENTATION_STATUS.md`, `PRIOR_SUBMISSION_CHECK.md` | Project status and prior-submission self-check records |
| `REPRODUCE.md` | Per-table / per-figure commands, and the boundary of what can be recomputed |
| `PROVENANCE.md` | Item-by-item timeline: what was fixed when |
| `ENVIRONMENT.md` | Fusion-side pinned environment; detector-side environment as inherited |
| `HASHES.txt` | SHA-256 of the manifests, the trained checkpoints and the protocol ledgers |

## 2. What is **not** included

| Excluded | Reason |
|---|---|
| Third-party datasets (M3FD, RoadScene, TNO, MSRS, LLVIP, FMB) | Their licences do **not** permit redistribution. Please obtain each dataset from its original provider. |
| Trained model checkpoints (`*.pt`) | Size. Their SHA-256 is in `HASHES.txt`; the reported claims rest on the released per-image CSVs. |
| Released baseline weights and the official upstream `model.pkl` | Third-party / upstream assets |
| Derived fused images (~12 GB) | Size; regenerable by running `gen_cga_fused.py` / `evaluate_brss.py` on the source datasets |
| The DCEvo detector stack | Third-party; see `ENVIRONMENT.md` for why this bounds reproducibility |

## 3. Reproducibility scope — stated accurately

- **Code: available.** Every script behind every table is included.
- **Statistics: recomputable.** Every reported statistic can be recomputed from the released per-image and per-object intermediate results **without retraining**. This includes the cluster-aware re-analysis (`code/cluster_audit_final.py`), which reproduces the paper's main detection table from the stored per-object detection vectors.
- **Fusion-side pipeline: re-runnable** end to end from the pinned environment (`requirements-brss.txt`).
- **Detector endpoint: recomputable, but not guaranteed bit-exact.** The detector-side library stack is inherited from the DCEvo release and is **not** pinned in this archive, so re-running the detector may not reproduce the predictions bit for bit. The statistics over the released predictions are exact; the detector pass itself is the weakest link. See `ENVIRONMENT.md`.
- **Data: not bundled.** The source datasets must be obtained separately from their original providers.

The scripts contain absolute paths from the original working environment and must be re-pointed at your local dataset locations. We therefore describe this as *code available + per-image results available; datasets obtainable from their original sources*, and we do **not** claim one-command full reproduction.

## 4. Environment used

- Fusion side: Python 3.11, PyTorch 2.1.0 + CUDA 11.8, Linux (WSL2) on a single consumer GPU (4 GB VRAM). Pinned in `requirements-brss.txt`.
- Metrics: `code/eval_metrics.py` (Y-channel EN / SF / AG / MI / QABF / VIF) and `code/eval_metrics_extended.py` (colour audit).
- Statistics: paired Wilcoxon with Holm correction; exact binomial (McNemar-style) for the object-level sensitivity endpoint; image-clustered net-sign, exact cluster sign-flip and image-cluster bootstrap for the primary detection reading.
- Detector: inherited from the DCEvo release (`ultralytics`, input size 640, IoU ≥ 0.5); not pinned here.

## 5. Release note — what each tag covers (please read)

This repository was created **fresh for public release**: it contains a snapshot commit of the items listed in §1 and a tag, not the internal development history. Its commit chain therefore **does not correspond** to the development repository whose hashes appear inside `GATE1_FREEZE.md`. Those internal hashes are kept as-is because the ledger is a frozen record; treat them as internal provenance, not as references resolvable here.

| Tag | Date | Covers |
|---|---|---|
| `v1.0` | 15 Sep 2026 | Fusion-quality evidence and the content-control detection tables. **Not** a snapshot of every number in the paper: it was frozen before the budget-matched 32-epoch baselines, the additional feature-space-arm seeds and inference-time controls, and the cluster-aware re-analysis were produced. |
| `v1.1-final-review-20260921` | 21 Sep 2026 | Everything above **plus** the budget-matched baselines and their per-seed comparisons, the extra feature-space-arm seeds and content controls, the selector-distribution summaries, the cluster-aware re-analysis with its object → image → seed mapping, and the manuscript PDF of this submission round. |

`v1.0` remains available and continues to describe exactly what it always described.

## 6. Licence

- **Code** (`code/`, and other program files): **MIT** — see [`LICENSE`](LICENSE).
- **Documentation, ledgers and tabular results** (`*.md`, `*.csv`): **CC-BY-4.0**.
- **Third-party datasets**: not covered; governed by their own terms.

## 7. How to cite

Please cite the manuscript and this archive:

> *Author(s)* (2026). *Image-Space Commitment Arbitration for Infrared–Visible Image Fusion — code and per-image results* (v1.1-final-review-20260921) [Computer software / Dataset]. GitHub. https://github.com/JassonG-xt/s4fusion-cga/releases/tag/v1.1-final-review-20260921
