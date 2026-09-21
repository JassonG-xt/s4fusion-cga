#!/usr/bin/env python3
"""B-4: a confidence interval for mAP50, and a paired test on all objects.

Two deliverables, both required by the unfreeze ledger (§4):

1. mAP50 with a bootstrap interval. The evaluator that produced the frozen numbers
   returns mAP50 as a single aggregate scalar and writes no per-image AP
   (run_m3fd_detection_eval.py reads ultralytics results[2] with save_json=False), so
   an interval cannot be obtained from it. It CAN be obtained from what is on disk:
   the per-image prediction files hold every box with its confidence, and the GT comes
   from archive.zip, so per-image AP is computable offline without re-running
   inference. The image is the resampling unit, consistent with the paper's paired
   analysis.

   Caliber disclosure, not a workaround: this is not ultralytics' mAP50. That number
   is a per-class AP averaged over classes with 101-point interpolation; this one is a
   per-image AP averaged over images, bootstrapped. The two are reported side by side
   and the difference is stated in the text rather than hidden by matching one to the
   other.

2. An exact binomial paired test on ALL annotated objects. The registered
   high-conflict endpoint filters objects by the conflict percentile, and that subset
   is defined by kappa, which is itself the trigger variable; the all-object test is
   the second line of defence and is computed here without any filter.

Read-only. Run from the code directory with the training interpreter.
"""
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np

CODE = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code")
sys.path.insert(0, str(CODE))
sys.path.insert(0, "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/tools/downstream_detection")

import diag_conflict_detection as D  # noqa: E402
import run_m3fd_detection_eval as R  # noqa: E402
from scipy import stats  # noqa: E402

ARB = CODE / "results" / "arb"
MANIFEST = CODE.parent / "dataset" / "manifests" / "m3fd_test.csv"
ARCHIVE = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation/archive.zip")
IOU = 0.5
B = 2000
SEED = 42


def load_gt(zf):
    ann = {}
    for n in zf.namelist():
        if n.startswith("Annotation/") and n.endswith(".xml"):
            stem = Path(n).stem
            try:
                ann[stem] = R.parse_xml_labels(zf, f"Annotation/{stem}.xml")
            except Exception:  # noqa: BLE001
                continue
    return ann


def per_image_ap50(preds, gt_lines, w, h):
    """AP@50 for one image, averaged over the classes that have ground truth."""
    classes = sorted({int(float(l.split()[0])) for l in gt_lines if len(l.split()) >= 5})
    aps = []
    for c in classes:
        gt = [(float(s[1]), float(s[2]), float(s[3]), float(s[4]))
              for l in gt_lines if len(l.split()) >= 5
              for s in [l.split()] if int(float(s[0])) == c]
        gt_xy = [(D.to_xyxy((c, cx, cy, bw, bh, 1.0), w, h)) for cx, cy, bw, bh in gt]
        preds_c = sorted([p for p in preds if p[0] == c], key=lambda p: -p[5])
        used = [False] * len(gt_xy)
        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        for i, p in enumerate(preds_c):
            best, best_i = 0.0, -1
            for j, g in enumerate(gt_xy):
                if used[j]:
                    continue
                v = D.iou(g, D.to_xyxy(p, w, h))
                if v > best:
                    best, best_i = v, j
            if best >= IOU and best_i >= 0:
                used[best_i] = True
                tp[i] = 1
            else:
                fp[i] = 1
        if len(gt_xy) == 0:
            continue
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / len(gt_xy)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps.append(float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])))
    return float(np.mean(aps)) if aps else np.nan


def collect(arm, ann, ids):
    """Per-image AP50 and per-object detection flags for one arm."""
    img_ap, det = {}, {}
    for sid in ids:
        gt = ann.get(sid)
        if not gt:
            continue
        p = ARB / arm / "runs_full" / "det" / "labels" / f"{sid}.txt"
        preds = D.load_yolo(p, True)
        ir = D.SRC / "Ir" / f"{sid}.png"
        if not ir.is_file():
            continue
        from PIL import Image
        w, h = Image.open(ir).size
        img_ap[sid] = per_image_ap50(preds, gt, w, h)
        for l in gt:
            s = l.split()
            if len(s) >= 5:
                g = (int(float(s[0])), float(s[1]), float(s[2]), float(s[3]), float(s[4]), 1.0)
                det[(sid, l)] = int(D.detected(D.to_xyxy(g, w, h), g[0], preds, w, h))
    return img_ap, det


def main():
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})
    with zipfile.ZipFile(ARCHIVE) as zf:
        ann = load_gt(zf)
    print(f"[B-4] manifest ids={len(ids)} ids_with_gt={sum(1 for i in ids if ann.get(i))}")

    arms = ["CGA_str", "B0cmp_full"]
    data = {a: collect(a, ann, ids) for a in arms}
    common = sorted(set(data["CGA_str"][0]) & set(data["B0cmp_full"][0]))
    print(f"[B-4] images with per-image AP in both arms: {len(common)}")

    def means(sample):
        out = {}
        for a in arms:
            v = [data[a][0][s] for s in sample]
            out[a] = float(np.nanmean(v))
        return out

    point = means(common)
    delta = point["CGA_str"] - point["B0cmp_full"]
    print(f"[B-4] per-image AP50 (image-averaged): CGA={point['CGA_str']:.4f} "
          f"B0cmp={point['B0cmp_full']:.4f}  delta={delta:+.4f}")
    print("[B-4] caliber note: ultralytics reports CGA 0.7282 vs B0cmp 0.7197 "
          "(delta +0.0085) by per-class AP with 101-point interpolation; this is a "
          "different statistic and the two are reported side by side.")

    rng = np.random.default_rng(SEED)
    boots = np.empty(B)
    for b in range(B):
        s = [common[i] for i in rng.integers(0, len(common), len(common))]
        m = means(s)
        boots[b] = m["CGA_str"] - m["B0cmp_full"]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"[B-4] paired bootstrap (B={B}, seed={SEED}, resampling images): "
          f"delta={delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"[B-4] CI contains zero: {'YES' if lo <= 0 <= hi else 'NO'}; "
          f"gate-C2 needs delta >= +0.01: {'met' if delta >= 0.01 else 'not met'}")

    # all-object paired test, no conflict filter
    n01 = n10 = n00 = n11 = 0
    for key in data["CGA_str"][1]:
        t = data["CGA_str"][1][key]
        b = data["B0cmp_full"][1].get(key)
        if b is None:
            continue
        if b == 0 and t == 1:
            n01 += 1
        elif b == 1 and t == 0:
            n10 += 1
        elif b == 0 and t == 0:
            n00 += 1
        else:
            n11 += 1
    p_all = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    n = n01 + n10 + n00 + n11
    print(f"[B-4] ALL-object paired test (no percentile filter): n={n} "
          f"recovered={n01} lost={n10} exact binomial p={p_all:.4e}")
    rec_cga = (n01 + n11) / n
    rec_b0 = (n10 + n11) / n
    print(f"[B-4] all-object recall: CGA={rec_cga:.4f} B0cmp={rec_b0:.4f} "
          f"delta={rec_cga - rec_b0:+.4f}")


if __name__ == "__main__":
    main()
