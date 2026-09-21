# PROVENANCE — what was fixed when

This is the release-side counterpart of Table 13 of the manuscript. The ledgers
named here are the record; no date below is inferred after the fact.

| Record | Date | Prospective for |
|---|---|---|
| Mechanism protocol (H1 arm, criterion, gates) | 2026-08-13 | the H1 arm and the same-budget ablations that follow |
| Seed-42 CGA checkpoint written | 2026-08-25 | — (a model artifact, not a criterion) |
| H5 arbitration hypothesis and pass criterion registered | 2026-09-09 | the budget-matched arbitration re-test only |
| Results freeze (main table, baselines, mechanism record) | 2026-09-13 | — (records results already obtained) |
| Amendment: per-seed Wilcoxon, best-epoch and NaN disclosure | 2026-09-14 | — (re-analysis of frozen CSVs; no new inference) |
| Pre-registration V1: criterion layering, gate mapping, aggregation rule | 2026-09-14 | the B-3 budget-matched runs, before they were launched |
| Pre-registration V2: independent-readings rule and wording rules | 2026-09-15 | the same runs; wording fixed before outcomes |
| Budget-matched B0\_e32 runs and gate verdicts recorded | 2026-09-15/16 | — (outcomes) |
| Public release `v1.0` | 2026-09-15 | — (a snapshot of the released repository) |
| H1 extra seeds and the content controls | 2026-09-19 | — (post-hoc additions to the frozen set) |
| Cluster-aware re-analysis | 2026-09-20 | — (post-hoc analysis of stored predictions) |
| Final review release `v1.1-final-review-20260921` | 2026-09-21 | — (a snapshot of the released repository; supersedes `v1.0`) |

## How to read the table

- The **criterion and the reporting rules** for the arbitration re-test were fixed
  before that re-test was run (`GATE1_PREREG_B3_20260914.md` and
  `GATE1_PREREG_B3_V2_20260915.md`). The wording rules are why the paper reports
  the outcome as *"statistically significant but below the stricter gate
  thresholds"* and never as having passed the gates.
- The **architecture, the training recipe, the seed-42 checkpoint and the choice
  of testbed** all pre-date those records. The study is therefore a *later-stage
  prospective re-test with a staged lock*, **not** a full pre-registration.
- The **cluster-aware re-analysis is post-hoc**. It was run on 2026-09-20, after
  the re-test outcome it describes had been obtained, in response to a review
  point about within-image dependence. It is reported in the manuscript as a
  post-hoc robustness analysis and is not presented as a pre-registered
  confirmatory test. The object-level exact binomial was the pre-specified
  endpoint; the pooled across-seed and Fisher statistics are descriptive.
- No row is prospective for anything earlier than its own date, and this archive
  makes no claim that it is.

## Ledger files in this release

| File | Role |
|---|---|
| `GATE1_FREEZE.md` | frozen results ledger (13 Sep 2026) |
| `GATE1_AMEND_20260914.md` | amendments to the frozen ledger; no frozen value changed |
| `GATE1_PREREG_B3_20260914.md` | pre-registered decision rules for the budget-matched control (14 Sep 2026) |
| `机制消融实验协议.md` | pre-registered ablation protocol: hypotheses, decision criteria, wording discipline |

`GATE1_PREREG_B3_V2_20260915.md` (the wording-and-evidence amendment of
15 September 2026) and `GATE1_UNFREEZE_20260915.md` (the matched-budget record of
15–16 September 2026) are retained in the author's working repository and are
available to the editor and the reviewers on request.

## Commit hashes cited inside the ledgers

The ledgers cite hashes from the internal development repository
(e.g. `749bbb0 → 30e90cf → 20997e1 → 5a1bf0c`). This public repository does not
carry that history — it is a whitelist snapshot with its own single-commit
chain — so those hashes are **internal provenance only** and cannot be resolved
here. They are left in the ledgers because the ledgers are frozen records.
