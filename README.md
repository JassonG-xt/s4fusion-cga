# Conflict-Gated Arbitration for Infrared–Visible Image Fusion — code and per-image results

Companion repository for the manuscript *"Conflict-Gated Arbitration for Infrared–Visible Image Fusion"* (submitted to **The Visual Computer**).

This repository releases the code, the per-image evaluation results and the pre-registration ledgers behind every number reported in the paper.

---

## 1. Contents

| Path | What it is |
|---|---|
| `code/*.py`, `code/*.sh` | Training, evaluation, statistical-analysis and figure scripts, including the single-entry batch runner `run_p1_experiments.sh` |
| `code/modules/` | Model modules, including the Conflict-Gated Arbitration implementation (`cga.py`), the boundary reliability estimator (`boundary.py`) and the fusion block |
| `code/results/**/*.csv` | **Per-image metric CSVs (41 files)** — the raw per-image basis of the main table, the cross-seed table and the detection-endpoint statistics |
| `writing/` | Manuscript chapter sources (abstract, introduction, related work, methods, experiments, discussion, references) |
| `latex/sn-article/` | LaTeX manuscript source (`main.tex`, `refs.bib`) and figure sources |
| `GATE1_FREEZE.md` | **Frozen results ledger** — the single source of truth for every number in the paper |
| `GATE1_AMEND_20260914.md` | Explicit amendments to the frozen ledger (no frozen value changed; one epoch record corrected) |
| `GATE1_PREREG_B3_20260914.md` | Before-run pre-registration of the decision criteria for the budget-matched control |
| `机制消融实验协议.md` | The pre-registered ablation protocol (hypotheses, decision criteria, wording discipline) |
| `IMPLEMENTATION_STATUS.md`, `PRIOR_SUBMISSION_CHECK.md` | Project status and prior-submission self-check records |

## 2. What is **not** included

| Excluded | Reason |
|---|---|
| Third-party datasets (M3FD, RoadScene, TNO, MSRS, LLVIP, FMB) | Their licences do **not** permit redistribution. Please obtain each dataset from its original provider. |
| Trained model checkpoints (`*.pt`) | Size; the reported claims rest on the released per-image CSVs |
| Released baseline weights and the official upstream `model.pkl` | Third-party / upstream assets |
| Derived fused images (~12 GB) | Size; regenerable by running `gen_cga_fused.py` / `evaluate_brss.py` on the source datasets |

## 3. Reproducibility scope — stated accurately

- **Code: available.** Every script listed above is included.
- **Per-image results: available.** Every reported metric can be recomputed from the CSVs here.
- **Data: not bundled.** The source datasets must be obtained separately from their original providers.
- The scripts contain absolute paths from the original working environment and must be re-pointed at your local dataset locations. We therefore describe this as *code available + per-image results available; datasets obtainable from their original sources*, and we do **not** claim one-command full reproduction.

## 4. Environment used

- Python 3.11, PyTorch 2.1.0 + CUDA 11.8, Linux (WSL2) on a single consumer GPU (4 GB VRAM).
- Metrics: `code/eval_metrics.py` (Y-channel EN / SF / AG / MI / QABF / VIF).
- Statistics: paired Wilcoxon with Holm correction; exact binomial for the detection endpoint; bootstrap for mAP confidence intervals.

## 5. Release note (please read)

This repository was created **fresh for public release**: it contains a single snapshot commit of the items listed in §1, and a `v1.0` tag. Its commit history therefore **does not correspond** to the internal development repository whose commit chain is cited inside `GATE1_FREEZE.md` (e.g. `749bbb0 → 30e90cf → 20997e1 → 5a1bf0c`). Those internal hashes are kept as-is in the ledger because the ledger is a frozen record; treat them as internal provenance, not as references resolvable in this repository.

## 6. Licence

- **Code** (`code/`, and other program files): **MIT** — see [`LICENSE`](LICENSE).
- **Documentation, ledgers and tabular results** (`*.md`, `*.csv`): **CC-BY-4.0**.
- **Third-party datasets**: not covered; governed by their own terms.

## 7. How to cite

Please cite the manuscript and this archive:

> *Author(s)* (2026). *Conflict-Gated Arbitration for Infrared–Visible Image Fusion — code and per-image results* (v1.0) [Computer software / Dataset]. GitHub. https://github.com/JassonG-xt/s4fusion-cga/releases/tag/v1.0
