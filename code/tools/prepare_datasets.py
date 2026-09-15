#!/usr/bin/env python3
"""Audit paired datasets and build deterministic split manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIELDS = [
    "dataset", "sample_id", "scene_id", "split", "ir_path", "vi_path",
    "label_path", "width", "height", "ir_mode", "vi_mode",
]


def image_map(directory):
    if not directory or not directory.is_dir():
        return {}
    return {p.stem: p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


def detect_layouts(root, dataset):
    candidates = {
        "llvip": [("infrared", "visible"), ("LLVIP/infrared", "LLVIP/visible")],
        "m3fd": [("Ir", "Vis"), ("ir", "vi"), ("M3FD/Ir", "M3FD/Vis")],
        "fmb": [("Infrared", "Visible"), ("ir", "vi"), ("FMB/Infrared", "FMB/Visible")],
        "msrs": [("train/ir", "train/vi"), ("Infrared", "Visible"), ("ir", "vi")],
        "tno": [("ir", "vi")],
        "roadscene": [("ir", "vi")],
    }
    layouts = []
    if dataset in {"tno", "roadscene"} and (root / "ir").is_dir() and (root / "vi").is_dir():
        return [(root / "ir", root / "vi", "test")]
    if dataset == "m3fd" and (root / "full" / "Ir").is_dir() and (root / "full" / "Vis").is_dir():
        return [(root / "full" / "Ir", root / "full" / "Vis", None)]
    if dataset in {"msrs", "fmb"}:
        for split in ("train", "test"):
            for ir_name, vi_name in (("ir", "vi"), ("Infrared", "Visible")):
                ir_path, vi_path = root / split / ir_name, root / split / vi_name
                if ir_path.is_dir() and vi_path.is_dir():
                    layouts.append((ir_path, vi_path, split))
            # Archives sometimes add one extra top-level directory.
            for ir_path in ((root / split).rglob("Infrared") if (root / split).is_dir() else ()):
                vi_path = ir_path.parent / "Visible"
                if vi_path.is_dir():
                    layouts.append((ir_path, vi_path, split))
        if layouts:
            return layouts
    for ir_rel, vi_rel in candidates[dataset]:
        if (root / ir_rel).is_dir() and (root / vi_rel).is_dir():
            return [(root / ir_rel, root / vi_rel, None)]
    for ir_name, vi_name in (("Ir", "Vis"), ("ir", "vi"), ("Infrared", "Visible"), ("infrared", "visible")):
        for ir_path in root.rglob(ir_name):
            vi_path = ir_path.parent / vi_name
            if vi_path.is_dir():
                return [(ir_path, vi_path, None)]
    raise FileNotFoundError(f"cannot detect {dataset} IR/VI layout below {root}")


def official_name_splits(dataset_root):
    result = {}
    for path in dataset_root.rglob("*.txt"):
        name = path.stem.lower()
        if name not in {"train", "val", "test", "pred"}:
            continue
        split = "test" if name in {"test", "pred"} else name
        try:
            for line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
                token = line.strip().split()[0] if line.strip() else ""
                if token:
                    result[Path(token).stem] = split
        except OSError:
            continue
    return result


def scene_id(dataset, sample_id):
    if dataset in {"llvip", "m3fd"}:
        return sample_id[:2]
    for separator in ("_", "-"):
        if separator in sample_id:
            return sample_id.split(separator)[0]
    return sample_id[:4]


def deterministic_group_val(rows, fraction=0.1, seed=42):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_id"]].append(row)
    ordered = sorted(groups, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    target = max(1, round(len(rows) * fraction))
    chosen, total = set(), 0
    for key in ordered:
        if total >= target:
            break
        chosen.add(key)
        total += len(groups[key])
    for row in rows:
        if row["scene_id"] in chosen:
            row["split"] = "val"


def infer_official_split(dataset, path):
    parts = {part.lower() for part in Path(path).parts}
    if "test" in parts or "val" in parts:
        return "test" if "test" in parts else "val"
    return "train"


def build_rows(data_root, dataset, dataset_root):
    layouts = detect_layouts(dataset_root, dataset)
    official = official_name_splits(dataset_root)
    if dataset == "m3fd" and (dataset_root / "Ir").is_dir():
        for sample in image_map(dataset_root / "Ir"):
            official[sample] = "test"
    labels = {}
    for label_dir_name in ("Annotation", "Annotations", "Label", "Labels", "labels", "Segmentation_labels"):
        for label_dir in dataset_root.rglob(label_dir_name):
            if label_dir.is_dir():
                labels.update({p.stem: p for p in label_dir.rglob("*") if p.is_file()})
    ir_files, vi_files, forced_splits = {}, {}, {}
    for ir_root, vi_root, forced_split in layouts:
        for sample, path in image_map(ir_root).items():
            ir_files[sample] = path
            if forced_split:
                forced_splits[sample] = forced_split
        vi_files.update(image_map(vi_root))
    common = sorted(ir_files.keys() & vi_files.keys())
    rows, errors = [], []
    for sample in common:
        ir_path, vi_path = ir_files[sample], vi_files[sample]
        try:
            with Image.open(ir_path) as ir_img, Image.open(vi_path) as vi_img:
                if ir_img.size != vi_img.size:
                    errors.append({"sample": sample, "error": "size_mismatch", "ir": ir_img.size, "vi": vi_img.size})
                    continue
                width, height = ir_img.size
                ir_mode, vi_mode = ir_img.mode, vi_img.mode
        except Exception as exc:
            errors.append({"sample": sample, "error": "decode", "detail": str(exc)})
            continue
        split = official.get(sample) or forced_splits.get(sample) or infer_official_split(dataset, ir_path)
        rows.append({
            "dataset": dataset,
            "sample_id": sample,
            "scene_id": scene_id(dataset, sample),
            "split": split,
            "ir_path": ir_path.relative_to(data_root).as_posix(),
            "vi_path": vi_path.relative_to(data_root).as_posix(),
            "label_path": labels[sample].relative_to(data_root).as_posix() if sample in labels else "",
            "width": width,
            "height": height,
            "ir_mode": ir_mode,
            "vi_mode": vi_mode,
        })
    train_rows = [row for row in rows if row["split"] == "train"]
    if train_rows and not any(row["split"] == "val" for row in rows):
        deterministic_group_val(train_rows)
    return rows, {
        "ir_count": len(ir_files), "vi_count": len(vi_files), "paired": len(common),
        "usable": len(rows), "ir_only": sorted(ir_files.keys() - vi_files.keys()),
        "vi_only": sorted(vi_files.keys() - ir_files.keys()), "errors": errors,
        "splits": Counter(row["split"] for row in rows),
        "resolutions": Counter(f'{row["width"]}x{row["height"]}' for row in rows),
    }


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", default="llvip,m3fd,fmb,msrs")
    args = parser.parse_args()
    roots = {name: args.data_root / name.upper() for name in args.datasets.split(",")}
    optional_eval = {"tno": args.data_root / "TNOTest", "roadscene": args.data_root / "RoadSceneTest"}
    roots.update({name: root for name, root in optional_eval.items() if root.is_dir()})
    all_rows, audit = [], {}
    for dataset, root in roots.items():
        rows, report = build_rows(args.data_root, dataset, root)
        all_rows.extend(rows)
        report["splits"] = dict(report["splits"])
        report["resolutions"] = dict(report["resolutions"])
        audit[dataset] = report
        for split in ("train", "val", "test"):
            write_manifest(args.data_root / "manifests" / f"{dataset}_{split}.csv", [r for r in rows if r["split"] == split])
    for split in ("train", "val", "test"):
        write_manifest(args.data_root / "manifests" / f"{split}_all.csv", [r for r in all_rows if r["split"] == split])
    write_manifest(args.data_root / "manifests" / "test_high_resolution.csv", [
        r for r in all_rows if r["split"] == "test" and r["dataset"] in {"llvip", "m3fd"}
        and max(int(r["width"]), int(r["height"])) >= 1024
    ])
    write_manifest(args.data_root / "manifests" / "test_ood.csv", [
        r for r in all_rows if r["split"] == "test" and r["dataset"] in {"tno", "roadscene"}
    ])
    audit_dir = args.data_root / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "dataset_inventory.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=list), encoding="utf-8")


if __name__ == "__main__":
    main()
