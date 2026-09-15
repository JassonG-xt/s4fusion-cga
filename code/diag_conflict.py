"""Cheap conflict-region diagnostic (no training, no detector). Tests whether
modal-conflict regions (both modalities have structure but it DISAGREES) are
(A) prevalent, (B) where baseline fusion loses structure, and (B2) concentrated
on task-relevant GT object boxes. Applies the D1 lesson: prove the problem is
real before building a mechanism.

Conflict(x) = min(|grad_ir|,|grad_vi|) * (1 - cos<grad_ir,grad_vi>)/2
  -> high only where BOTH have edges AND they point differently (orientation
     disagreement) or opposite (thermal polarity reversal).
Retention(x) = |grad_fused| / max(|grad_ir|,|grad_vi|)  (fusion keeps structure?)
If Retention drops sharply in conflict regions, fusion demonstrably fails there.
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import convolve

KX = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float) / 4
KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float) / 4


def grads(a):
    gx, gy = convolve(a, KX, mode="reflect"), convolve(a, KY, mode="reflect")
    return gx, gy, np.sqrt(gx * gx + gy * gy + 1e-12)


def load_L(p):
    return np.asarray(Image.open(p).convert("L"), float) / 255.0


def load_viY(p):
    return np.asarray(Image.open(p).convert("RGB").convert("YCbCr").getchannel("Y"), float) / 255.0


def parse_boxes(xml):
    out = []
    for o in ET.parse(xml).getroot().findall("object"):
        v = {t: None for t in ("xmin", "ymin", "xmax", "ymax")}
        for e in o.iter():
            if e.tag in v and e.text:
                v[e.tag] = int(float(e.text))
        if all(x is not None for x in v.values()):
            out.append((v["xmin"], v["ymin"], v["xmax"], v["ymax"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../dataset")
    ap.add_argument("--manifest", default="../dataset/manifests/m3fd_test.csv")
    ap.add_argument("--fused-dir", default="results/c256/B0_m3fd/m3fd")
    ap.add_argument("--dump-dir", default="results/conflict_diag")
    ap.add_argument("--dump-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import csv
    root = Path(args.data_root)
    with open(args.manifest, encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    if args.limit:
        rows = rows[:args.limit]

    prev, Rc, Rl, box_c, bg_c, Rbox_hi, Rbox_lo = [], [], [], [], [], [], []
    per_image = []
    for row in rows:
        sid = row["sample_id"]
        fp = Path(args.fused_dir) / f"{sid}.png"
        if not fp.is_file():
            continue
        ir = load_L(root / row["ir_path"])
        vi = load_viY(root / row["vi_path"])
        fu = load_L(fp)
        if not (ir.shape == vi.shape == fu.shape):
            h0 = min(ir.shape[0], vi.shape[0], fu.shape[0]); w0 = min(ir.shape[1], vi.shape[1], fu.shape[1])
            ir, vi, fu = ir[:h0, :w0], vi[:h0, :w0], fu[:h0, :w0]
        gxi, gyi, gi = grads(ir)
        gxv, gyv, gv = grads(vi)
        _, _, gf = grads(fu)
        cos = (gxi * gxv + gyi * gyv) / (gi * gv + 1e-9)
        conflict = np.minimum(gi, gv) * (1 - cos) / 2.0
        retention = gf / (np.maximum(gi, gv) + 1e-9)

        edge = np.maximum(gi, gv) > np.percentile(np.maximum(gi, gv), 70)  # structured pixels
        ce = conflict[edge]
        if ce.size < 50:
            continue
        hi = conflict >= np.percentile(ce, 80)   # high-conflict among edges
        lo = conflict <= np.percentile(ce, 20)
        hi &= edge; lo &= edge
        prev.append(float(hi.sum()) / float(edge.sum()))
        Rc.append(float(retention[hi].mean())); Rl.append(float(retention[lo].mean()))

        # task-relevant: conflict inside GT object boxes vs whole-image edge mean
        boxes = parse_boxes(root / row["label_path"]) if row.get("label_path") else []
        if boxes:
            bmask = np.zeros_like(edge)
            for (x0, y0, x1, y1) in boxes:
                x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(edge.shape[1], x1), min(edge.shape[0], y1)
                bmask[y0:y1, x0:x1] = True
            be = bmask & edge
            if be.sum() > 20:
                box_c.append(float(conflict[be].mean())); bg_c.append(float(conflict[edge & ~bmask].mean() if (edge & ~bmask).sum() else np.nan))
                bhi = be & (conflict >= np.percentile(conflict[be], 70))
                blo = be & (conflict <= np.percentile(conflict[be], 30))
                if bhi.sum() > 5 and blo.sum() > 5:
                    Rbox_hi.append(float(retention[bhi].mean())); Rbox_lo.append(float(retention[blo].mean()))
        per_image.append((float(conflict[edge].mean()) if boxes else 0.0, sid, row))

    Rc, Rl = np.array(Rc), np.array(Rl)
    print(f"images analyzed: {len(Rc)}")
    print(f"\n[A] conflict prevalence among edge pixels: mean {np.mean(prev)*100:.1f}%  "
          f"median {np.median(prev)*100:.1f}%  (per-image top-conflict share)")
    p_ret = stats.wilcoxon(Rc, Rl, alternative="less").pvalue
    print(f"\n[B] fusion gradient-RETENTION (higher=keeps structure):")
    print(f"    conflict regions: {Rc.mean():.3f}   low-conflict regions: {Rl.mean():.3f}   "
          f"drop: {(Rl.mean()-Rc.mean()):+.3f}  ({100*(Rl.mean()-Rc.mean())/Rl.mean():+.1f}%)")
    print(f"    paired Wilcoxon (retention_conflict < retention_low): p={p_ret:.2e}  "
          f"{'-> fusion LOSES structure in conflict regions' if p_ret<0.05 and Rc.mean()<Rl.mean() else '-> no significant drop'}")
    if box_c:
        bc, gc = np.array(box_c), np.array(bg_c)
        m = ~np.isnan(gc)
        p_box = stats.wilcoxon(bc[m], gc[m], alternative="greater").pvalue
        print(f"\n[B2] task-relevance: conflict inside GT object boxes vs background edges:")
        print(f"    box conflict: {bc[m].mean():.4f}   background: {gc[m].mean():.4f}   "
              f"p(box>bg)={p_box:.2e}  {'-> conflict concentrates on objects' if p_box<0.05 and bc[m].mean()>gc[m].mean() else '-> not concentrated'}")
    if Rbox_hi:
        rh, rl = np.array(Rbox_hi), np.array(Rbox_lo)
        p_rb = stats.wilcoxon(rh, rl, alternative="less").pvalue
        print(f"    within objects, retention hi-conflict {rh.mean():.3f} vs lo-conflict {rl.mean():.3f}  "
              f"p={p_rb:.2e}  {'-> fusion fails MORE on high-conflict objects' if p_rb<0.05 and rh.mean()<rl.mean() else ''}")

    # dump the K highest-conflict images (ir|vi|fused) for visual ghosting inspection
    per_image.sort(reverse=True)
    dd = Path(args.dump_dir); dd.mkdir(parents=True, exist_ok=True)
    for _, sid, row in per_image[:args.dump_k]:
        ir = load_L(root / row["ir_path"]); vi = load_viY(root / row["vi_path"]); fu = load_L(Path(args.fused_dir) / f"{sid}.png")
        h0 = min(ir.shape[0], vi.shape[0], fu.shape[0]); w0 = min(ir.shape[1], vi.shape[1], fu.shape[1])
        montage = np.concatenate([ir[:h0, :w0], vi[:h0, :w0], fu[:h0, :w0]], axis=1)
        Image.fromarray((montage * 255).astype(np.uint8)).save(dd / f"{sid}_ir_vi_fused.png")
    print(f"\ndumped {min(args.dump_k, len(per_image))} highest-conflict montages (IR|VI|Fused) to {dd}")


if __name__ == "__main__":
    main()
