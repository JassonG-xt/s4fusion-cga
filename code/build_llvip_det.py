#!/usr/bin/env python3
"""Build LLVIP detection training set (YOLO format) for the DCEvo detector.

- Input: manifests/llvip_train.csv / llvip_val.csv (IR+VI pairs, XML annotations)
- Output: baselines_2025/DCEvo/datasets/LLVIP_det/{train,val}/{images,labels}
- Class: person -> 0 (single class)
- Images: both IR and VI are included (suffix _ir/_vi) so the detector is
  robust to fused-output appearance; labels are shared per pair.
"""
import csv
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026")
MANIFEST_DIR = ROOT / "dataset" / "manifests"
DATA_ROOT = ROOT / "dataset"
OUT = Path("/mnt/e/lunwen/S4Fusion-main/baselines_2025/DCEvo/datasets/LLVIP_det")

CLASS_MAP = {"person": 0}


def parse_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width", "0")) if size is not None else 0.0
    height = float(size.findtext("height", "0")) if size is not None else 0.0
    if width <= 0 or height <= 0:
        return []
    labels = []
    for obj in root.findall("object"):
        cls_raw = (obj.findtext("name", "") or "").strip().lower()
        if cls_raw not in CLASS_MAP:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        xmin = max(0.0, min(width, xmin))
        xmax = max(0.0, min(width, xmax))
        ymin = max(0.0, min(height, ymin))
        ymax = max(0.0, min(height, ymax))
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1 or bh <= 1:
            continue
        xc = (xmin + xmax) / 2.0 / width
        yc = (ymin + ymax) / 2.0 / height
        nw = bw / width
        nh = bh / height
        labels.append(f"{CLASS_MAP[cls_raw]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    return labels


def build(split: str) -> None:
    csv_path = MANIFEST_DIR / f"llvip_{split}.csv"
    img_dir = OUT / split / "images"
    lbl_dir = OUT / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_label = n_noann = 0
    with csv_path.open(encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i % 500 == 0:
                print(f"[{split}] row {i}: imgs={n_img}", flush=True)
            sid = row["sample_id"]
            xml_rel = row["label_path"]
            xml_path = DATA_ROOT / xml_rel
            labels = parse_xml(xml_path) if xml_path.is_file() else []
            if not labels:
                n_noann += 1
            for tag, rel in (("ir", row["ir_path"]), ("vi", row["vi_path"])):
                src = DATA_ROOT / rel
                name = f"{sid}_{tag}.jpg"
                if not src.is_file():
                    continue
                dst = img_dir / name
                if not dst.exists():
                    os.symlink(src, dst)
                (lbl_dir / f"{Path(name).stem}.txt").write_text(
                    "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
                )
                n_img += 1
                n_label += 1
    print(f"[{split}] images={n_img} labels={n_label} no-ann-pairs={n_noann}", flush=True)


def write_yaml() -> None:
    yaml_obj = {
        "path": str(OUT),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "person"},
    }
    import yaml as _yaml
    (OUT / "dataset.yaml").write_text(
        _yaml.safe_dump(yaml_obj, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"[yaml] {OUT / 'dataset.yaml'}")


if __name__ == "__main__":
    for split in ("train", "val"):
        build(split)
    write_yaml()
