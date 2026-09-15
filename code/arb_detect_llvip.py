#!/usr/bin/env python3
"""LLVIP detection A/B: CGA-gatelr vs B0cmp fused outputs on LLVIP test subset.

Pipeline:
1. read llvip test manifest ids (sample_ids)
2. parse local LLVIP XML -> YOLO GT (single class: person)
3. build dataset from fused arb images (CGA_llvip / B0cmp_llvip)
4. run DCEvo detect val with the freshly finetuned llvip detector
5. per-object McNemar on high-conflict matches (same logic as M3FD full-300)
"""
import csv
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TOOLS = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/tools/downstream_detection"
sys.path.insert(0, TOOLS)
import run_m3fd_detection_eval as R  # noqa: E402
import yaml  # noqa: E402

ROOT = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026")
DATA_ROOT = ROOT / "dataset"
ARB = ROOT / "code" / "results" / "arb"
import os as _os
MANIFEST = DATA_ROOT / "manifests" / _os.environ.get("LLVIP_MANIFEST", "llvip_test_sub1000.csv")
XML_ROOT = DATA_ROOT / "LLVIP" / "LLVIP" / "Annotations"
DETECT_WEIGHTS = "/mnt/e/lunwen/S4Fusion-main/baselines_2025/DCEvo/runs/detect/llvip_person4/weights/best.pt"

YAML_TEMPLATE = {
    "train": "test/images",
    "val": "test/images",
    "test": "test/images",
    "names": {0: "person"},
}


def parse_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width", "0")) if size is not None else 0.0
    height = float(size.findtext("height", "0")) if size is not None else 0.0
    if width <= 0 or height <= 0:
        return []
    labels = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = max(0.0, min(width, float(bbox.findtext("xmin", "0"))))
        ymin = max(0.0, min(height, float(bbox.findtext("ymin", "0"))))
        xmax = max(0.0, min(width, float(bbox.findtext("xmax", "0"))))
        ymax = max(0.0, min(height, float(bbox.findtext("ymax", "0"))))
        bw, bh = xmax - xmin, ymax - ymin
        if bw <= 1 or bh <= 1:
            continue
        labels.append(f"0 {(xmin + xmax) / 2.0 / width:.6f} {(ymin + ymax) / 2.0 / height:.6f} {bw / width:.6f} {bh / height:.6f}")
    return labels


def build_dataset(variant: str, ids: list[str]) -> Path:
    ds = ARB / variant / "ds_llvip"
    img_dir = ds / "test" / "images"
    lbl_dir = ds / "test" / "labels"
    if ds.exists():
        shutil.rmtree(ds)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    src = ARB / variant / "images"
    missing = 0
    for sid in ids:
        img = src / f"{sid}.png"
        if not img.is_file():
            missing += 1
            continue
        shutil.copy2(img, img_dir / img.name)
        xml = XML_ROOT / f"{sid}.xml"
        lines = parse_xml(xml) if xml.is_file() else []
        (lbl_dir / f"{sid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    yaml_obj = dict(YAML_TEMPLATE)
    yaml_obj["path"] = str(ds)
    (ds / "dataset.yaml").write_text(yaml.safe_dump(yaml_obj, sort_keys=False, allow_unicode=True))
    print(f"[{variant}] dataset ready: {len(ids) - missing}/{len(ids)} images", flush=True)
    return ds


def detect(dcevo_val, ds: Path, variant: str) -> Path:
    run_dir = ARB / variant / "runs_llvip"
    labels_dir = run_dir / "det" / "labels"
    if labels_dir.is_dir() and any(labels_dir.glob("*.txt")):
        print(f"[{variant}] reuse existing runs_llvip labels ({len(list(labels_dir.glob('*.txt')))} files)", flush=True)
        return labels_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    res = dcevo_val.run(
        data=str(ds / "dataset.yaml"), weights=str(DETECT_WEIGHTS), batch_size=1, imgsz=640,
        conf_thres=0.001, iou_thres=0.7, max_det=300, task="test", device="cpu", workers=0,
        single_cls=False, augment=False, verbose=False, save_txt=True, save_hybrid=False,
        save_conf=True, save_json=False, project=str(run_dir), name="det", exist_ok=True,
        half=False, dnn=False, min_items=0, plots=False,
    )
    results = res[0] if isinstance(res, tuple) else res
    try:
        summary = [round(float(x), 4) for x in results[:4]]
    except (TypeError, ValueError):
        summary = str(results)[:80]
    print(f"{variant} P/R/mAP50/mAP5095: {summary}", flush=True)
    return run_dir / "det" / "labels"


def yolo_to_xyxy(box, w, h):
    cx, cy, nw, nh = box[1:5]
    x1 = (cx - nw / 2) * w
    y1 = (cy - nh / 2) * h
    x2 = (cx + nw / 2) * w
    y2 = (cy + nh / 2) * h
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_stats(gt_boxes, pred_boxes, width, height, iou_thres=0.5):
    matched = set()
    tp = 0
    missed = []
    for gi, gt in enumerate(gt_boxes):
        gt_xyxy = yolo_to_xyxy(gt, width, height)
        best_iou, best_pi = 0.0, -1
        for pi, pred in enumerate(pred_boxes):
            if pi in matched or pred[0] != gt[0]:
                continue
            iou = iou_xyxy(gt_xyxy, yolo_to_xyxy(pred, width, height))
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou >= iou_thres and best_pi >= 0:
            matched.add(best_pi)
            tp += 1
        else:
            missed.append(gi)
    fp = len(pred_boxes) - len(matched)
    fn = len(gt_boxes) - tp
    return tp, fp, fn, missed


def mcnemar_exact(recovered: int, lost: int) -> float:
    n = recovered + lost
    if n == 0:
        return 1.0
    p = 0.0
    for k in range(recovered + 1):
        p += math.comb(n, k) * 0.5 ** n
    return min(2 * p, 1.0)


def main():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})
    print(f"manifest ids: {len(ids)}", flush=True)
    dcevo_val = R.load_dcevo_val()

    label_dirs = {}
    variants = _os.environ.get("LLVIP_VARIANTS", "B0cmp_llvip,CGA_llvip").split(",")
    for variant in variants:
        ds = build_dataset(variant, ids)
        label_dirs[variant] = detect(dcevo_val, ds, variant)

    gt_dir = ARB / variants[0] / "ds_llvip" / "test" / "labels"
    total = {"B0cmp": {"tp": 0, "fp": 0, "fn": 0}, "CGA": {"tp": 0, "fp": 0, "fn": 0}}
    recovered = lost = 0
    n_gt = 0
    for sid in ids:
        gt_file = gt_dir / f"{sid}.txt"
        if not gt_file.is_file():
            continue
        gt = []
        for line in gt_file.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                gt.append([int(float(parts[0]))] + [float(x) for x in parts[1:5]])
        n_gt += len(gt)
        for variant, key in [(v, v.split("_")[0]) for v in label_dirs]:
            pred_file = label_dirs[variant] / f"{sid}.txt"
            pred = []
            if pred_file.is_file():
                for line in pred_file.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and float(parts[5]) >= 0.25:
                        pred.append([int(float(parts[0]))] + [float(x) for x in parts[1:5]])
            tp, fp, fn, _ = match_stats(gt, pred, 1280, 1024)
            total[key]["tp"] += tp
            total[key]["fp"] += fp
            total[key]["fn"] += fn
        def load_pred(variant):
            pred = []
            pf = label_dirs[variant] / f"{sid}.txt"
            if pf.is_file():
                for line in pf.read_text().splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and float(parts[5]) >= 0.25:
                        pred.append([int(float(parts[0]))] + [float(x) for x in parts[1:5]])
            return pred

        b_missed = match_stats(gt, load_pred(variants[0]), 1280, 1024)[3]
        o_missed = match_stats(gt, load_pred(variants[1]), 1280, 1024)[3]
        recovered += len([gi for gi in b_missed if gi not in o_missed])
        lost += len([gi for gi in o_missed if gi not in b_missed])

    p_val = mcnemar_exact(recovered, lost)
    print("=" * 60, flush=True)
    for key in ["B0cmp", "CGA"]:
        t = total[key]
        p = t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else 0.0
        r = t["tp"] / (t["tp"] + t["fn"]) if t["tp"] + t["fn"] else 0.0
        print(f"{key}: TP={t['tp']} FP={t['fp']} FN={t['fn']} P={p:.4f} R={r:.4f}", flush=True)
    print(f"GT objects: {n_gt}, recovered={recovered}, lost={lost}, McNemar p={p_val:.4g}", flush=True)
    print(f"mAP50  diff: CGA vs B0cmp -> see per-variant P/R/mAP lines above", flush=True)


if __name__ == "__main__":
    main()