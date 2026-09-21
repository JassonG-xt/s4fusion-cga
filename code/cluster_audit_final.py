#!/usr/bin/env python3
"""Cluster-aware (object -> image -> seed) reanalysis of the detection endpoint.

This is the WP1 deliverable of the 2026-09-20 revision plan. It replaces the
object-level exact binomial as the *primary* inference for the detection
endpoint with an image-clustered analysis, and reports the object-level test
alongside it as a sensitivity analysis.

Naming (fixed in the 2026-09-21 last-round minor revision): the analysis below
is a *post-hoc cluster-aware robustness analysis* -- it was run on 2026-09-20,
after the re-test outcome it describes was already in hand, and it is reported
in the manuscript under that name and not as a pre-registered confirmatory
test. The object-level exact binomial is the *pre-specified sensitivity
endpoint*; the pooled across-seed/Fisher statistics are *descriptive*. The
manuscript's three names map one-to-one onto the three blocks of output here.

Unit hierarchy (explicit):
    object  -- nested in image (a GT box)
    image   -- nested in seed (a training run)
    seed    -- 42 / 123 / 3407

What it reports per comparison:
  * the object -> image -> seed mapping
  * object-level discordant counts and exact binomial p   (sensitivity only)
  * image-level net-sign test                             (cluster-aware)
  * exact cluster sign-flip test on |image score|         (cluster-aware)
  * image-cluster bootstrap effect and 95% CI             (cluster-aware)
  * high-conflict recall, all-object recall, precision, per-image mAP50
    and false positives, all with the SAME estimator on every arm
  * the same analysis for the learned / uniform / shuffled content controls

No training and no detector inference is rerun: annotations and stored
predictions are reused, exactly as in the review's sensitivity audit. The frozen
counts the review reproduced (2583 objects, 878 high-conflict objects, 181
images bearing them) are asserted at the end of the collection pass, so this
script fails loudly rather than silently changing the population.

Usage:
    python cluster_audit_final.py            # full report
    python cluster_audit_final.py --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import arb_full300_stats as A  # noqa: E402
import diag_conflict_detection as D  # noqa: E402

B_BOOT = 2000
BOOT_SEED = 20260920
N_IMG = 300

# (test, base) pairs. Budget-matched pairs are the fairer convention; the
# content controls share the single seed-42 checkpoint with the main table.
SEED_PAIRS = [
    (42, "CGA_str", "B0_e32_s42"),
    (123, "CGA_str_s123", "B0_e32_s123"),
    (3407, "CGA_str_s3407", "B0_e32_s3407"),
]
SINGLE_BASE_PAIR = ("CGA_str", "B0cmp_full")
CONTROL_PAIRS = [
    ("cga_uniform", "CGA_str"),
    ("cga_shuffle", "CGA_str"),
]


# --------------------------------------------------------------------------
# exact cluster sign-flip test
# --------------------------------------------------------------------------
def exact_cluster_sign_flip(weights, observed):
    """Two-sided exact random-sign test on the absolute cluster score."""
    dp = {0: 1}
    for w in weights:
        nxt = {}
        for total, count in dp.items():
            nxt[total + w] = nxt.get(total + w, 0) + count
            nxt[total - w] = nxt.get(total - w, 0) + count
        dp = nxt
    denom = 2 ** len(weights)
    return sum(c for t, c in dp.items() if abs(t) >= abs(observed)) / denom


def cluster_bootstrap(image_scores, n_objects, rng, B=B_BOOT):
    """Percentile CI for the recall delta, resampling IMAGES with replacement.

    Estimand: the *object-weighted* recall difference of the two fixed trained
    arms over the fixed 300 test images. Two conventions travel with it and are
    stated in the manuscript (Methods, "Statistical units and analysis
    families"; Table 6 caption):
      * the resampling unit is the IMAGE, because objects inside one image
        share its fused output and its detection error;
      * the DENOMINATOR is held fixed at the observed number of high-conflict
        objects (n_objects = 878). The arms are compared on one and the same
        already-annotated object set, so letting the resampled object count
        enter the denominator would inject a between-image object-count
        variability that this endpoint does not have. Passing the resampled
        object count instead would change the estimand to an image-average
        recall difference.
    """
    keys = list(image_scores)
    scores = np.array([image_scores[k] for k in keys], float)
    idx = rng.integers(0, len(scores), size=(B, len(scores)))
    boots = scores[idx].sum(axis=1) / n_objects
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --------------------------------------------------------------------------
# per-arm full-scoring pass (recall / precision / mAP50 / false positives)
# --------------------------------------------------------------------------
def score_image(sid, gt_lines, preds, W, H):
    """One image: GT count, matched GT count, prediction count, AP@0.5."""
    gts = []
    for line in gt_lines:
        s = line.split()
        if len(s) < 5:
            continue
        g = (int(float(s[0])), float(s[1]), float(s[2]), float(s[3]), float(s[4]), 1.0)
        gts.append((g[0], D.to_xyxy(g, W, H)))
    preds_xy = [(p[0], D.to_xyxy(p, W, H), p[5]) for p in preds]

    matched = 0
    for cls, gbox in gts:
        if D.detected(gbox, cls, preds, W, H):
            matched += 1

    # per-image AP at IoU 0.5, 101-point interpolation, averaged over the
    # classes present in the image (identical estimator for every arm)
    aps = []
    for cls in sorted({c for c, _ in gts}):
        n_gt_c = sum(1 for c, _ in gts if c == cls)
        p_c = sorted([(conf, box) for c, box, conf in preds_xy if c == cls],
                     key=lambda t: -t[0])
        if not p_c:
            aps.append(0.0)
            continue
        tp = np.zeros(len(p_c))
        fp = np.zeros(len(p_c))
        taken = set()
        for i, (_, pbox) in enumerate(p_c):
            best, best_j = 0.0, -1
            for j, (gcls, gbox) in enumerate(gts):
                if gcls != cls or j in taken:
                    continue
                v = D.iou(pbox, gbox)
                if v > best:
                    best, best_j = v, j
            if best >= D.IOU:
                taken.add(best_j)
                tp[i] = 1
            else:
                fp[i] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / max(n_gt_c, 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-12)
        grid = np.linspace(0, 1, 101)
        interp = [prec[rec >= t].max() if np.any(rec >= t) else 0.0 for t in grid]
        aps.append(float(np.mean(interp)))
    return len(gts), matched, len(preds_xy), (float(np.mean(aps)) if aps else 0.0)


def full_pass(arm, objects, meta, anns):
    """Aggregate detection statistics for one arm on the frozen 300 images."""
    sids = sorted(meta)
    cache = {}
    n_gt = n_matched = n_pred = 0
    ap_sum = 0.0
    det = np.zeros(len(objects), dtype=bool)
    obj_index = defaultdict(list)
    for i, (sid, _gx, _cls, _cf) in enumerate(objects):
        obj_index[sid].append(i)

    for sid in sids:
        if sid not in cache:
            p = A._label_dir(arm) / f"{sid}.txt"
            cache[sid] = D.load_yolo(p, True) if p.is_file() else []
        preds = cache[sid]
        H, W = meta[sid]
        g, m, n, ap = score_image(sid, anns[sid], preds, W, H)
        n_gt += g
        n_matched += m
        n_pred += n
        ap_sum += ap
        for i in obj_index.get(sid, ()):
            det[i] = D.detected(objects[i][1], objects[i][2], preds, W, H)

    n_img = len(sids)
    # per-object detection vector is the authoritative recall source; n_gt from
    # score_image counts the same GT set, so the two must agree.
    assert n_gt == len(objects), (arm, n_gt, len(objects))
    return {
        "arm": arm,
        "objects": len(objects),
        "detected": int(det.sum()),
        "recall_all": float(det.sum() / len(objects)),
        "predictions": n_pred,
        "matched": n_matched,
        "precision_all": float(n_matched / n_pred) if n_pred else float("nan"),
        "false_positives": n_pred - n_matched,
        "fp_per_image": float((n_pred - n_matched) / n_img),
        "map50_per_image": float(ap_sum / n_img),
        "det": det,
    }


def pair_report(test, base, objects, meta, anns, high_mask, passes, rng):
    """Object/image/seed-level analysis of one (test, base) contrast."""
    tvec = passes[test]["det"]
    bvec = passes[base]["det"]
    by_image = defaultdict(list)
    diffs = []
    for i in np.flatnonzero(high_mask):
        d = int(tvec[i]) - int(bvec[i])
        by_image[objects[i][0]].append(d)
        if d:
            diffs.append(d)

    rec = sum(d > 0 for d in diffs)
    lost = sum(d < 0 for d in diffs)
    obj_p = (stats.binomtest(min(rec, lost), rec + lost, 0.5).pvalue
             if rec + lost else 1.0)

    image_scores = {k: sum(v) for k, v in by_image.items()}
    pos = sum(v > 0 for v in image_scores.values())
    neg = sum(v < 0 for v in image_scores.values())
    tied = sum(v == 0 for v in image_scores.values())
    isign_p = (stats.binomtest(min(pos, neg), pos + neg, 0.5).pvalue
               if pos + neg else 1.0)
    wts = [abs(v) for v in image_scores.values() if v]
    obs = int(sum(image_scores.values()))
    flip_p = exact_cluster_sign_flip(wts, obs)

    hc = high_mask
    hc_n = int(hc.sum())
    lo, hi = cluster_bootstrap(image_scores, hc_n, rng)
    delta = (tvec[hc].sum() - bvec[hc].sum()) / hc_n

    all_n = len(objects)
    delta_all = float((tvec.sum() - bvec.sum()) / all_n)

    return {
        "test": test,
        "base": base,
        "high_conflict_objects": hc_n,
        "images_with_high_conflict_objects": len(by_image),
        "object_recovered": rec,
        "object_lost": lost,
        "object_exact_p": float(obj_p),
        "image_positive": pos,
        "image_negative": neg,
        "image_tied": tied,
        "image_net_sign_p": float(isign_p),
        "cluster_sign_flip_p": float(flip_p),
        "observed_cluster_score": obs,
        "nonzero_cluster_count": len(wts),
        "cluster_weight_counts": dict(Counter(wts)),
        "hc_recall_delta": float(delta),
        "hc_recall_delta_ci": [lo, hi],
        "all_recall_delta": delta_all,
        "hc_recall_base": float(bvec[hc].mean()),
        "hc_recall_test": float(tvec[hc].mean()),
        "all_recall_base": float(bvec.mean()),
        "all_recall_test": float(tvec.mean()),
        "precision_base": passes[base]["precision_all"],
        "precision_test": passes[test]["precision_all"],
        "map50_base": passes[base]["map50_per_image"],
        "map50_test": passes[test]["map50_per_image"],
        "map50_delta": passes[test]["map50_per_image"] - passes[base]["map50_per_image"],
        "fp_per_image_base": passes[base]["fp_per_image"],
        "fp_per_image_test": passes[test]["fp_per_image"],
        "fp_per_image_delta": passes[test]["fp_per_image"] - passes[base]["fp_per_image"],
    }


def fmt(r):
    ci = r["hc_recall_delta_ci"]
    return (
        f"{r['test']} vs {r['base']}\n"
        f"  high-conflict objects             = {r['high_conflict_objects']}"
        f" in {r['images_with_high_conflict_objects']} images\n"
        f"  object level (sensitivity)        = {r['object_recovered']}/"
        f"{r['object_lost']}, exact p = {r['object_exact_p']:.4g}\n"
        f"  image net sign  (cluster-aware)   = {r['image_positive']}/"
        f"{r['image_negative']} ({r['image_tied']} tied), p = {r['image_net_sign_p']:.4g}\n"
        f"  cluster sign flip (cluster-aware) = {r['nonzero_cluster_count']} !=0 "
        f"clusters, score {r['observed_cluster_score']}, p = {r['cluster_sign_flip_p']:.4g}\n"
        f"  cluster bootstrap effect          = {r['hc_recall_delta']:+.4f} "
        f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]\n"
        f"  high-conflict recall              = {r['hc_recall_base']:.4f} -> "
        f"{r['hc_recall_test']:.4f}\n"
        f"  all-object recall                 = {r['all_recall_base']:.4f} -> "
        f"{r['all_recall_test']:.4f} (delta {r['all_recall_delta']:+.5f})\n"
        f"  precision (all-object)            = {r['precision_base']:.4f} -> "
        f"{r['precision_test']:.4f}\n"
        f"  per-image mAP50                   = {r['map50_base']:.4f} -> "
        f"{r['map50_test']:.4f} (delta {r['map50_delta']:+.4f})\n"
        f"  false positives / image           = {r['fp_per_image_base']:.3f} -> "
        f"{r['fp_per_image_test']:.3f} (delta {r['fp_per_image_delta']:+.3f})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--export-mapping", default=None, metavar="DIR",
                    help="write object_image_mapping.csv and seed_arm_map.csv here")
    args = ap.parse_args()

    rows = list(csv.DictReader(A.MANIFEST.open(encoding="utf-8-sig")))
    ids = sorted({r["sample_id"] for r in rows})
    with zipfile.ZipFile(A.ROOT_OLD / "archive.zip") as z:
        anns = A.read_annotations(z)

    objects, meta, n_gt, n_src = A.collect_objects(ids, anns)
    conflicts = np.array([o[3] for o in objects])
    high_mask = conflicts > np.percentile(conflicts, 66)

    n_hc = int(high_mask.sum())
    n_hc_imgs = len({objects[i][0] for i in np.flatnonzero(high_mask)})
    print("Population (asserted against the frozen audit):")
    print(f"  all_objects                     = {len(objects)}")
    print(f"  high_conflict_objects           = {n_hc}")
    print(f"  images_with_high_conflict       = {n_hc_imgs}")
    print(f"  images_with_gt_and_sources      = {n_src}")
    assert len(objects) == 2583, len(objects)
    assert n_hc == 878, n_hc
    assert n_hc_imgs == 181, n_hc_imgs
    print("  frozen baseline reproduced: OK (2583 / 878 / 181)")

    if args.export_mapping:
        out_dir = Path(args.export_mapping)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "object_image_mapping.csv").open("w", newline="",
                                                        encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["object_index", "image_id", "class_id",
                        "conflict_score", "high_conflict"])
            for i, (sid, _gx, cls, cf) in enumerate(objects):
                w.writerow([i, sid, cls, f"{float(cf):.10g}",
                            int(bool(high_mask[i]))])
        with (out_dir / "seed_arm_map.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "test_arm", "base_arm", "convention"])
            for seed, test, base in SEED_PAIRS:
                w.writerow([seed, test, base, "budget_matched"])
            w.writerow([42, SINGLE_BASE_PAIR[0], SINGLE_BASE_PAIR[1],
                        "single_baseline"])
            for test, base in CONTROL_PAIRS:
                w.writerow([42, test, base, "content_control"])
        print(f"\nwrote object->image->seed mapping under {out_dir} "
              f"({len(objects)} objects; {len(SEED_PAIRS) + 1 + len(CONTROL_PAIRS)} arm pairs)")

    arms = sorted({a for _, t, b in SEED_PAIRS for a in (t, b)}
                  | {a for p in [SINGLE_BASE_PAIR] + CONTROL_PAIRS for a in p})
    passes = {}
    print("\nFull-scoring pass per arm:")
    for a in arms:
        passes[a] = full_pass(a, objects, meta, anns)
        p = passes[a]
        print(f"  {a:16s} recall_all={p['recall_all']:.4f} "
              f"prec_all={p['precision_all']:.4f} "
              f"mAP50={p['map50_per_image']:.4f} fp/img={p['fp_per_image']:.2f}")

    rng = np.random.default_rng(BOOT_SEED)
    out = {"population": {"objects": len(objects), "high_conflict": n_hc,
                          "images_with_high_conflict": n_hc_imgs},
           "per_arm": {a: {k: v for k, v in passes[a].items() if k != "det"}
                       for a in arms},
           "pairs": []}

    print("\n" + "=" * 72)
    print("CLUSTER-AWARE ANALYSIS -- budget-matched per seed (fairer convention)")
    print("=" * 72)
    for seed, t, b in SEED_PAIRS:
        r = pair_report(t, b, objects, meta, anns, high_mask, passes, rng)
        r["seed"] = seed
        r["convention"] = "budget_matched"
        out["pairs"].append(r)
        print(f"\n[seed {seed}]\n" + fmt(r))

    print("\n" + "=" * 72)
    print("CLUSTER-AWARE ANALYSIS -- single shared baseline (frozen convention)")
    print("=" * 72)
    r = pair_report(*SINGLE_BASE_PAIR, objects, meta, anns, high_mask, passes, rng)
    r["seed"] = 42
    r["convention"] = "single_baseline"
    out["pairs"].append(r)
    print("\n" + fmt(r))

    print("\n" + "=" * 72)
    print("CONTENT CONTROLS -- same seed-42 checkpoint as the main table")
    print("=" * 72)
    for t, b in CONTROL_PAIRS:
        r = pair_report(t, b, objects, meta, anns, high_mask, passes, rng)
        r["seed"] = 42
        r["convention"] = "content_control"
        out["pairs"].append(r)
        print("\n" + fmt(r))

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
