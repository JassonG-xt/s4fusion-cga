"""Aggregate the SOTA comparison table from real per-image CSVs.

Phase A2 of the 2026-09-09 work plan. Produces the M3FD main comparison table
(mean +/- std per method) from:
  1. baseline per-image CSVs in results/baselines/<method>_m3fd.csv (7 methods,
     produced by baselines/collect_outputs.py + eval_metrics.py)
  2. Ours (CGA_str, seed 42) and same-budget baseline (B0cmp_full) per-image
     metrics, computed here from the fused images in results/arb/<cell>/images
     with the SAME eval_metrics.py core so the protocol is unified.

Output: results/baselines/SOTA_MAIN_TABLE.md + .csv (mean rows only for csv).
All numbers are generated from real files; nothing is hand-entered.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ["EN", "SF", "AG", "MI", "QABF", "VIF"]

# method display name -> per-image csv (results/baselines/)
# NOTE (2026-09-10 provenance fix): s4fusion_m3fd.csv (2026-08-19) was generated
# from the B0 same-budget retrain images, NOT the official weights (pixel probe:
# mean abs diff 6.3-8.4 between the two models' outputs). It is therefore excluded
# from the table; the official row reads s4fusion_official_m3fd.csv (produced by
# eval_metrics.py from results/baseline_official/m3fd, the official model.pkl
# outputs under the unified tiled protocol), and the retrain row is B0cmp_full.
BASELINES = {
    "S4Fusion (official weights)": "s4fusion_official_m3fd.csv",
    "MetaFusion": "meta_fusion_m3fd.csv",
    "CDDFuse": "cddfuse_m3fd.csv",
    "EMMA": "emma_m3fd.csv",
    "DCEvo": "dcevo_m3fd.csv",
    "W-Mamba": "wmamba_m3fd.csv",
    "Diff-IF": "diffif_m3fd.csv",
    "FusionMamba": "fusionmamba_m3fd.csv",  # TBD until weights arrive (B3)
}

# Ours rows: image dir -> label. Evaluated with eval_metrics.py (same as baselines).
OURS = {
    "B0cmp_full": "S4Fusion (same-budget retrain)",
    "CGA_str": "CGA (ours)",
}


def eval_image_dir(cell: str) -> Path:
    """Run eval_metrics.py on results/arb/<cell>/images vs the unified inputs."""
    out_csv = HERE / f"results/baselines/{cell}_m3fd.csv"
    if out_csv.exists():
        print(f"[skip] {out_csv} exists", flush=True)
        return out_csv
    cmd = [
        sys.executable, "eval_metrics.py",
        "--ir-path", "baselines/inputs/m3fd/ir",
        "--vi-path", "baselines/inputs/m3fd/vi",
        "--fused-path", f"results/arb/{cell}/images",
        "--use-y", "--skip-perceptual",
        "--out-csv", str(out_csv),
    ]
    print(f"[run] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)
    return out_csv


def load_per_image(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig") as h:
        rows = [r for r in csv.DictReader(h) if r.get("name") not in ("__mean__", "__std__")]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ours-eval", action="store_true",
                    help="only aggregate existing CSVs (no image evaluation)")
    args = ap.parse_args()

    cells: dict[str, Path] = {}
    for method, fname in BASELINES.items():
        p = HERE / "results/baselines" / fname
        cells[method] = p
    for cell, label in OURS.items():
        p = HERE / "results/baselines" / f"{cell}_m3fd.csv"
        if not p.exists():
            if args.skip_ours_eval:
                print(f"[warn] {cell}: no CSV, --skip-ours-eval set, skipping", flush=True)
                continue
            p = eval_image_dir(cell)
        cells[label] = p

    # aggregate
    table: dict[str, dict[str, tuple[float, float, int]]] = {}
    for label, path in cells.items():
        if not path.exists():
            print(f"[TBD ] {label}: {path.name} missing (leave as TBD)", flush=True)
            continue
        rows = load_per_image(path)
        stats = {}
        for m in METRICS:
            vals = [float(r[m]) for r in rows if r.get(m) not in (None, "")]
            stats[m] = (statistics.mean(vals), statistics.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))
        table[label] = stats

    # write csv (mean rows)
    out_csv = HERE / "results/baselines/SOTA_MAIN_TABLE.csv"
    with out_csv.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["method", "n"] + METRICS)
        for label, stats in table.items():
            w.writerow([label, stats["EN"][2]] + [f"{stats[m][0]:.4f}" for m in METRICS])

    # write markdown with mean±std
    md = ["# SOTA main comparison table — M3FD (300 pairs, unified evaluator)", "",
          "| Method | " + " | ".join(METRICS) + " |",
          "|---" * (len(METRICS) + 1) + "|"]
    for label, stats in table.items():
        cells_ = [f"{stats[m][0]:.4f}±{stats[m][1]:.4f}" for m in METRICS]
        md.append(f"| {label} | " + " | ".join(cells_) + " |")
    for method in BASELINES:
        if method not in table:
            md.append(f"| {method} | TBD (weights unavailable / pending B3) |")
    out_md = HERE / "results/baselines/SOTA_MAIN_TABLE.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    for line in md[2:]:
        print(line)


if __name__ == "__main__":
    main()
