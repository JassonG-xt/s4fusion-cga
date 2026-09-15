#!/usr/bin/env python3
"""Full M3FD (300) detection A/B: trained CGA-gatelr vs same-regime B0 baseline.

Reuses the official pipeline's XML->YOLO GT construction (archive.zip has all
4200 M3FD annotations) but runs on the FULL 300-image official test set instead
of the legacy 90-image split, and on freshly fused arb images.
"""
import csv
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

TOOLS = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/tools/downstream_detection"
sys.path.insert(0, TOOLS)
import run_m3fd_detection_eval as R  # noqa: E402
import yaml  # noqa: E402

ROOT_OLD = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation")
ARB = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
MANIFEST = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/dataset/manifests/m3fd_test.csv")

NAMES = R.DCEVO_DATA_YAML_TEMPLATE["names"]


def read_annotations(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    anns = {}
    for n in zf.namelist():
        if not (n.startswith("Annotation/") and n.endswith(".xml")):
            continue
        stem = Path(n).stem
        try:
            lines = R.parse_xml_labels(zf, f"Annotation/{stem}.xml")
        except (KeyError, ET.ParseError):
            continue
        anns[stem] = lines
    return anns


def build_dataset(variant: str, ids: list[str], anns: dict[str, list[str]]) -> Path:
    ds = ARB / variant / "ds_full"
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
        lines = anns.get(sid, [])
        (lbl_dir / f"{sid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    yaml_obj = dict(R.DCEVO_DATA_YAML_TEMPLATE)
    yaml_obj["path"] = str(ds)
    (ds / "dataset.yaml").write_text(yaml.safe_dump(yaml_obj, sort_keys=False, allow_unicode=True))
    print(f"[{variant}] dataset ready: {len(ids)-missing}/{len(ids)} images, {len(ids)-missing} labels", flush=True)
    return ds


def detect(dcevo_val, ds: Path, variant: str) -> None:
    run_dir = ARB / variant / "runs_full"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    res = dcevo_val.run(
        data=str(ds / "dataset.yaml"), weights=str(R.DCEVO_WEIGHTS), batch_size=1, imgsz=640,
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
    print(f"{variant} full-300 P/R/mAP50/mAP5095: {summary}", flush=True)


def main():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})
    print(f"manifest ids: {len(ids)}", flush=True)
    with zipfile.ZipFile(ROOT_OLD / "archive.zip") as zf:
        anns = read_annotations(zf)
        have = sum(1 for i in ids if i in anns)
        print(f"annotated: {have}/{len(ids)}", flush=True)

    dcevo_val = R.load_dcevo_val()
    for variant in sys.argv[1:] or ["cga_gatelr", "B0cmp"]:
        ds = build_dataset(variant, ids, anns)
        detect(dcevo_val, ds, variant)


if __name__ == "__main__":
    main()