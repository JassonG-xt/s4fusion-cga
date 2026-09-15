"""Arbitration pre-check step 2: run the DCEvo detector on the arbitrated image
sets. Must run under the OLD venv (has cv2). Saves per-image predictions."""
import sys, shutil
from pathlib import Path

TOOLS = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/tools/downstream_detection"
sys.path.insert(0, TOOLS)
import run_m3fd_detection_eval as R  # noqa: E402
import yaml  # noqa: E402

dcevo_val = R.load_dcevo_val()  # imports DCEvo val, patches dataloader, chdir DCEVO_ROOT

ARB = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code/results/arb")
OLDLBL = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/results_metrics/downstream_detection/datasets/m3fd_baseline/test/labels")
NAMES = {0: "people", 1: "car", 2: "bus", 3: "light", 4: "moto", 5: "trunk"}

for variant in (sys.argv[1:] or ["A1_ircommit", "A2_contrast", "A3_max"]):
    ds = ARB / variant / "ds"
    (ds / "test/images").mkdir(parents=True, exist_ok=True)
    (ds / "test/labels").mkdir(parents=True, exist_ok=True)
    for img in sorted((ARB / variant / "images").glob("*.png")):
        shutil.copy2(img, ds / "test/images" / img.name)
        lbl = OLDLBL / f"{img.stem}.txt"
        if lbl.is_file():
            shutil.copy2(lbl, ds / "test/labels" / lbl.name)
    y = dict(names=NAMES, path=str(ds), train="test/images", val="test/images", test="test/images")
    (ds / "dataset.yaml").write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True))
    run_dir = ARB / variant / "runs"
    if (run_dir / "det").exists():
        shutil.rmtree(run_dir / "det")
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
    print(f"{variant} P/R/mAP50/mAP5095: {summary}", flush=True)
print("ARB DETECTION DONE", flush=True)
