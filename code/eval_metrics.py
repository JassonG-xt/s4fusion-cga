import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EPS = 1e-12


def list_images(root: Path):
    return sorted([p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def list_image_stems(root: Path):
    return sorted({p.stem for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS})


def read_gray(path: Path, use_y: bool):
    img = Image.open(path)
    if use_y:
        y = img.convert("YCbCr").split()[0]
        return np.asarray(y, dtype=np.float32)
    return np.asarray(img.convert("L"), dtype=np.float32)


def match_size(*imgs):
    h = min(img.shape[0] for img in imgs)
    w = min(img.shape[1] for img in imgs)
    return [img[:h, :w] for img in imgs]


def entropy(img):
    hist = np.histogram(img, bins=256, range=(0, 255))[0].astype(np.float64)
    p = hist / (hist.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def spatial_frequency(img):
    rf = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
    cf = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def average_gradient(img):
    dx = img[1:, 1:] - img[:-1, 1:]
    dy = img[1:, 1:] - img[1:, :-1]
    return float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))


def mutual_information(a, b):
    h2, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=256, range=((0, 255), (0, 255)))
    pxy = h2 / (h2.sum() + EPS)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    return float((pxy[nz] * np.log2(pxy[nz] / (px[:, None] * py[None, :] + EPS)[nz])).sum())


def conv2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    return np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))


def sobel_grad(img):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = conv2d(img, kx)
    gy = conv2d(img, ky)
    g = np.hypot(gx, gy)
    a = np.arctan2(gy, gx)
    return g, a


def qabf(ir, vi, fu, k1=10.0, k2=7.5, t1=0.85, t2=0.08):
    g_ir, a_ir = sobel_grad(ir)
    g_vi, a_vi = sobel_grad(vi)
    g_fu, a_fu = sobel_grad(fu)

    def edge_preserve(g_a, a_a):
        ratio = np.where(g_a > EPS, np.where(g_fu >= g_a, g_a / (g_fu + EPS), g_fu / (g_a + EPS)), 0.0)
        qg = 1.0 / (1.0 + np.exp(-k1 * (ratio - t1)))
        qg = qg * (1.0 / (1.0 + np.exp(-k2 * (ratio - t2))))
        da = np.abs(a_a - a_fu)
        da = np.where(da > np.pi / 2, np.pi - da, da)
        qa = 1.0 - da / (np.pi / 2)
        return qg * qa

    q_ir = edge_preserve(g_ir, a_ir)
    q_vi = edge_preserve(g_vi, a_vi)
    num = (q_ir * g_ir + q_vi * g_vi).sum()
    den = (g_ir + g_vi).sum() + EPS
    return float(num / den)


def _to_rgb_tensor(img: np.ndarray):
    # img: HxW or HxWx1, range 0..255
    import torch

    if img.ndim == 2:
        img = img[:, :, None]
    img = img.astype(np.float32) / 255.0
    img = np.repeat(img, 3, axis=2)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t


def _vif_mean(ir, vi, fu):
    # VIF is full-reference; we average VIF(IR, Fused) and VIF(VI, Fused).
    from sewar.full_ref import vifp

    vif_ir = float(vifp(ir, fu))
    vif_vi = float(vifp(vi, fu))
    return 0.5 * (vif_ir + vif_vi)


def evaluate_pair(ir, vi, fu, vif_fn=None, topiq_fn=None, musiq_fn=None, device=None):
    en = entropy(fu)
    sf = spatial_frequency(fu)
    ag = average_gradient(fu)
    mi_ir = mutual_information(fu, ir)
    mi_vi = mutual_information(fu, vi)
    mi_sum = mi_ir + mi_vi
    q = qabf(ir, vi, fu)

    vif = None
    if vif_fn is not None:
        vif = vif_fn(ir, vi, fu)

    topiq = None
    musiq = None
    if topiq_fn is not None or musiq_fn is not None:
        import torch

        t = _to_rgb_tensor(fu).to(device)
        if topiq_fn is not None:
            topiq = float(topiq_fn(t).item())
        if musiq_fn is not None:
            musiq = float(musiq_fn(t).item())

    return {
        "EN": en,
        "SF": sf,
        "AG": ag,
        "MI_IR": mi_ir,
        "MI_VI": mi_vi,
        "MI": mi_sum,
        "QABF": q,
        "VIF": vif,
        "TOPIQ": topiq,
        "MUSIQ": musiq,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion metrics for IR/VI/Fused triplets.")
    parser.add_argument("--ir-path", required=True, help="path to infrared images")
    parser.add_argument("--vi-path", required=True, help="path to visible images")
    parser.add_argument("--fused-path", required=True, help="path to fused images")
    parser.add_argument("--out-csv", default="metrics.csv", help="output csv path")
    parser.add_argument("--use-y", action="store_true", help="use Y channel for visible images")
    parser.add_argument("--limit", type=int, default=0, help="limit number of images (0 = no limit)")
    parser.add_argument("--device", default="auto", help="device for perceptual metrics (auto/cpu/cuda)")
    parser.add_argument("--topiq-name", default="topiq_nr_cgfiqa_swin", help="pyiqa metric name for TOPIQ")
    parser.add_argument("--musiq-name", default="musiq", help="pyiqa metric name for MUSIQ")
    parser.add_argument("--skip-vif", action="store_true", help="skip VIF computation")
    parser.add_argument("--skip-perceptual", action="store_true", help="skip TOPIQ/MUSIQ computation")
    parser.add_argument("--skip-topiq", action="store_true", help="skip TOPIQ even when perceptual metrics are enabled")
    parser.add_argument("--skip-musiq", action="store_true", help="skip MUSIQ even when perceptual metrics are enabled")
    parser.add_argument("--paper-only", action="store_true", help="output only paper metrics (SF/AG/VIF/QABF/TOPIQ/MUSIQ)")
    args = parser.parse_args()

    ir_dir = Path(args.ir_path)
    vi_dir = Path(args.vi_path)
    fu_dir = Path(args.fused_path)
    if not ir_dir.is_dir() or not vi_dir.is_dir() or not fu_dir.is_dir():
        raise FileNotFoundError("ir/vi/fused path not found")

    names = list_image_stems(fu_dir)
    ir_by_stem = {p.stem: p for p in ir_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    vi_by_stem = {p.stem: p for p in vi_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    if args.limit and args.limit > 0:
        names = names[: args.limit]
    names = [n for n in names if n in ir_by_stem and n in vi_by_stem and any((fu_dir / f"{n}{ext}").exists() for ext in IMG_EXTS)]
    if not names:
        raise RuntimeError("no matched image names in ir/vi/fused")

    # optional metrics (installed on demand)
    vif_fn = None
    if not args.skip_vif:
        try:
            from sewar.full_ref import vifp  # noqa: F401

            vif_fn = _vif_mean
        except Exception:
            raise RuntimeError(
                "VIF requires 'sewar'. Install via: pip install sewar"
            )

    topiq_fn = None
    musiq_fn = None
    device = None
    if not args.skip_perceptual:
        try:
            import torch
            import pyiqa
        except Exception:
            raise RuntimeError(
                "TOPIQ/MUSIQ require 'pyiqa'. Install via: pip install pyiqa"
            )
        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        def _list_metrics():
            for fn in ("list_models", "list_metrics", "list_iqa_models"):
                if hasattr(pyiqa, fn):
                    try:
                        return list(getattr(pyiqa, fn)())
                    except Exception:
                        pass
            return []

        if not args.skip_topiq:
            try:
                topiq_fn = pyiqa.create_metric(args.topiq_name, device=device)
            except Exception as e:
                avail = [m for m in _list_metrics() if "topiq" in str(m).lower()]
                raise RuntimeError(
                    f"TOPIQ metric '{args.topiq_name}' not available. Available TOPIQ metrics: {avail}. "
                    f"Original error: {e}"
                )
        if not args.skip_musiq:
            try:
                musiq_fn = pyiqa.create_metric(args.musiq_name, device=device)
            except Exception as e:
                avail = [m for m in _list_metrics() if "musiq" in str(m).lower()]
                raise RuntimeError(
                    f"MUSIQ metric '{args.musiq_name}' not available. Available MUSIQ metrics: {avail}. "
                    f"Original error: {e}"
                )

    rows = []
    try:
        from tqdm import tqdm
        iterator = tqdm(names)
    except Exception:
        iterator = names

    for name in iterator:
        fu_path = next((fu_dir / f"{name}{ext}" for ext in IMG_EXTS if (fu_dir / f"{name}{ext}").exists()), None)
        if fu_path is None:
            continue
        ir = read_gray(ir_by_stem[name], use_y=False)
        vi = read_gray(vi_by_stem[name], use_y=args.use_y)
        fu = read_gray(fu_path, use_y=False)
        ir, vi, fu = match_size(ir, vi, fu)
        metrics = evaluate_pair(ir, vi, fu, vif_fn=vif_fn, topiq_fn=topiq_fn, musiq_fn=musiq_fn, device=device)
        rows.append({"name": name, **metrics})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.paper_only:
        fields = ["name", "SF", "AG", "VIF", "QABF", "TOPIQ", "MUSIQ"]
    else:
        fields = ["name", "EN", "SF", "AG", "MI_IR", "MI_VI", "MI", "QABF"]
        if vif_fn is not None:
            fields.append("VIF")
        if topiq_fn is not None:
            fields.append("TOPIQ")
        if musiq_fn is not None:
            fields.append("MUSIQ")

    def stats(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)), float(np.std(vals))

    mean_row = {"name": "__mean__"}
    std_row = {"name": "__std__"}
    for k in fields[1:]:
        if any(r.get(k) is not None for r in rows):
            m, s = stats(k)
            mean_row[k] = m
            std_row[k] = s

    # When output is restricted (e.g. --paper-only), drop extra keys to avoid csv errors.
    rows_out = []
    for r in rows:
        rows_out.append({k: r.get(k) for k in fields})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
        writer.writerow(mean_row)
        writer.writerow(std_row)

    print(f"wrote {out_csv} ({len(rows)} images)")
    print("mean:", mean_row)


if __name__ == "__main__":
    main()
