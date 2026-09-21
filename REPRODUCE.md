# REPRODUCE — what each script regenerates, and what it cannot

Scripts carry absolute paths from the original working environment (`/mnt/e/...`,
WSL2). Re-point them at your local dataset and checkpoint locations before
running. Nothing below retrains a model unless it is listed under
*Training*.

## 1. The detection verdict and the content controls (paper Tables 6, 11)

```bash
# primary reading: object -> image -> seed, budget-matched per seed,
# single shared baseline, and the two content controls
python code/cluster_audit_final.py \
    --json   out/cluster_audit_final.json \
    --export-mapping out/mapping

# out/mapping/object_image_mapping.csv   2583 objects (image_id, class, conflict, high_conflict)
# out/mapping/seed_arm_map.csv           the six (seed, test arm, base arm) pairs analysed
```

The script asserts the frozen population **2583 objects / 878 high-conflict
objects over 181 images** and aborts if the inputs resolve to anything else.
A reference run is shipped as
`code/results/cluster_audit/cluster_audit_final.json`; the numbers it prints are
the ones in the paper (s42: 18/4 object-level, 13/4 image-level, sign-flip
`p = 0.0211`, bootstrap CI `[+0.0046, +0.0296]`; s123 and s3407 do not reject).

Inputs it needs, all reconstructible from the released material:

| Input | What it is |
|---|---|
| `dataset/manifests/m3fd_test.csv` | the 300-image test manifest (SHA-256 in `HASHES.txt`) |
| `archive.zip` (M3FD annotations) | official XML ground truth — obtain from the dataset provider |
| `code/results/arb/<arm>/runs_full/det/labels/*.txt` | detector predictions per arm, YOLO format |

## 2. Table-by-table map

| Paper item | Script | Reads | Writes |
|---|---|---|---|
| Table 3, main comparison | `aggregate_sota.py` | `code/results/baselines/*.csv` | `code/results/baselines/SOTA_MAIN_TABLE.csv` |
| Table 4, per-seed (budget-matched) | `analyze_seeds.py`, `report_effects.py` | `code/results/seeds/CGA_str_s*.csv`, `B0_e32_s*.csv` | `code/results/seeds/effects_budget_matched.csv` (rows `scope` = s42 / s123 / s3407) |
| Table 5, pooled CGA−B0 (descriptive) | `analyze_seeds.py`, `report_effects.py` | `code/results/seeds/CGA_str_s*.csv`, `B0_s*.csv` | `code/results/seeds/effects_primary.csv` (row `scope` = pooled) |
| Table 6 + Table 11 | `cluster_audit_final.py` | manifest, annotations, `arb/*/runs_full/det/labels/` | stdout + `--json` + `--export-mapping` |
| Table 11, evaluator mAP50 column | — (recorded output) | see below | `code/results/h5/control_baseline_map50.csv` |
| Tables 7–8, object-level sensitivity | `arb_full300_stats.py` | `arb/*/runs_full/det/labels/`, annotations | `code/results/h5/effects_single_baseline*.csv`, `effects_budget_matched.csv` |
| Table 9, selector statistics | `export_select_stats.py` | selector activations on the 300 test pairs | `code/results/select_stats/select_*.csv` |
| Table 10, colour audit | `rebuild_rgb.py`, `eval_metrics_extended.py` | fused outputs, visible references | `code/results/arb/*/color_metrics.csv` |
| Table 12, parameter-free rules | `arb_full300_stats.py` over the `P_*` arms | `arb/P_*/runs_full/det/labels/` | `code/results/h5/effects_vs_cga.csv` |
| Table 13, provenance timeline | — | the ledgers named in `PROVENANCE.md` | — |
| Gate C1/C2/C3 verdicts | `struct_noninferior.py`, `c3_report.py`, `c3_report_evalmetrics.py` | per-image fusion CSVs | unrounded boundary values and one-sided bounds |
| Table 2, incremental cost | `profile_brss_complexity.py`, `profile_percentiles.py` | model + a single GPU | latency / memory profiles |
| Fig. 1 (soft vs hard) | `results/make_soft_vs_hard_fig.py` | closed form + synthetic pair | `figs/soft_vs_hard.pdf` |
| Fig. 3–4 (mechanism panels) | `cga_visualize.py` | fused outputs and maps | `figs/*_panel.png` |
| Fig. 5 (qualitative) | `assemble_qual_fig.py` | fused outputs of every method | `figs/qualitative_*.png` |

### Where the content-control mAP50 column comes from

The ultralytics evaluator returns mAP50 as a single aggregate scalar and writes no
per-image AP, so the three values in the paper's content-control table are the
evaluator's own printed output. They are shipped here rather than recomputed:

- `code/results/h5/control_baseline_map50.csv` — the numbers, with a `source`
  column naming the file each one was read from (parsed, not retyped).
- `code/results/full300_evaluator_lines.txt` — the verbatim evaluator lines for
  the learned map and the shared baseline, with file and line number.
- `code/results/b1_content_control_stdout.txt` — the original stdout of the
  content-control runs (uniform and shuffled), shipped in full. It also records
  the prediction-coverage check on those two arms.

Everything else in Table 11 (recall, precision, false positives per image, and
every test statistic) is recomputable with `cluster_audit_final.py`.

## 3. Training

```bash
bash code/run_seeds_queue_v2.sh      # marker-file checkpointing; safe to re-run
bash code/run_after_seeds_v2.sh      # the follow-up evaluation queue
```

Fine-tunes from the official public S4Fusion checkpoint: 128×128 crops, batch 1
with 8-step accumulation, 2048 crops/epoch, AdamW at 2e-5, gradient-norm clip
10.0, mixed precision, seeds 42 / 123 / 3407. The CGA arm extends the schedule to
32 epochs and gives the module parameters a separate learning rate of 0.01 with
zero weight decay. Checkpoints are written atomically and every run supports
`--resume`, so an interrupted queue continues rather than restarts.

**Two properties of this chain are findings of the paper, not incidents**: the
CGA-line seeds develop late-epoch validation NaN (the baseline line does not),
and every reported CGA number comes from a best-validation checkpoint selected
*before* NaN onset. Reproducing those numbers means selecting checkpoints by the
same rule.

## 4. The boundary — read this before filing a mismatch

- **Exact**: every statistic in the paper, recomputed from the released
  per-image / per-object CSVs. The cluster-aware analysis is included in this set.
- **Exact**: the fusion-side pipeline, from the pinned environment.
- **Not guaranteed bit-exact**: the detector pass. The detector-side library
  stack is inherited from the DCEvo release and not pinned here (see
  `ENVIRONMENT.md`). Statistics computed over the released predictions are
  exact; re-running the detector may move a small number of predictions and
  therefore a small number of counts.
- **Not reproducible from this archive alone**: anything that needs the source
  images (fusion inference, visualisation) or the checkpoints. Both are
  obtainable elsewhere; their SHA-256 is in `HASHES.txt`.
