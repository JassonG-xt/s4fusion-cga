import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

from eval_metrics import average_gradient, match_size, read_gray, sobel_grad, spatial_frequency


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EPS = 1e-12


def _list_images(root: Path):
    return sorted([p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _read_rgb(path: Path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _edge_mask(img: np.ndarray, percentile: float = 75.0):
    g, _ = sobel_grad(img)
    threshold = np.percentile(g, percentile)
    return g, g > max(float(threshold), EPS)


def edge_overlap_recall(ir: np.ndarray, vi: np.ndarray, fu: np.ndarray):
    _, edge_ir = _edge_mask(ir)
    _, edge_vi = _edge_mask(vi)
    _, edge_fu = _edge_mask(fu)
    edge_ref = np.logical_or(edge_ir, edge_vi)
    return float(np.logical_and(edge_ref, edge_fu).sum() / (edge_ref.sum() + EPS))


def edge_continuity_score(fu: np.ndarray):
    _, edge = _edge_mask(fu)
    if edge.sum() <= 1:
        return 0.0
    horizontal = np.logical_and(edge[:, 1:], edge[:, :-1]).sum()
    vertical = np.logical_and(edge[1:, :], edge[:-1, :]).sum()
    return float((horizontal + vertical) / (edge.sum() + EPS))


def object_boundary_recall(ir: np.ndarray, vi: np.ndarray, fu: np.ndarray):
    g_ir, _ = _edge_mask(ir)
    g_vi, _ = _edge_mask(vi)
    ref = np.maximum(g_ir, g_vi)
    ref_mask = ref > max(float(np.percentile(ref, 85.0)), EPS)
    _, fu_mask = _edge_mask(fu, percentile=75.0)
    return float(np.logical_and(ref_mask, fu_mask).sum() / (ref_mask.sum() + EPS))


def _ssim_single(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)))


def rgb_ssim(a_rgb: np.ndarray, b_rgb: np.ndarray):
    return float(np.mean([_ssim_single(a_rgb[..., i], b_rgb[..., i]) for i in range(3)]))


def colorfulness(rgb: np.ndarray):
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _rgb_to_lab(rgb: np.ndarray):
    rgb = np.clip(rgb.astype(np.float64) / 255.0, 0, 1)
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    mat = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = rgb @ mat.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    l = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def ciede2000_mean(a_rgb: np.ndarray, b_rgb: np.ndarray):
    try:
        from skimage.color import deltaE_ciede2000, rgb2lab

        return float(deltaE_ciede2000(rgb2lab(a_rgb / 255.0), rgb2lab(b_rgb / 255.0)).mean())
    except Exception:
        lab_a = _rgb_to_lab(a_rgb)
        lab_b = _rgb_to_lab(b_rgb)
        return float(np.linalg.norm(lab_a - lab_b, axis=-1).mean())


def evaluate_extended_pair(
    ir: np.ndarray,
    vi: np.ndarray,
    fu: np.ndarray,
    vi_rgb: np.ndarray | None = None,
    fused_rgb: np.ndarray | None = None,
):
    ir, vi, fu = match_size(ir, vi, fu)
    metrics = {
        "SF": spatial_frequency(fu),
        "AG": average_gradient(fu),
        "EOR": edge_overlap_recall(ir, vi, fu),
        "ECS": edge_continuity_score(fu),
        "OBR": object_boundary_recall(ir, vi, fu),
    }
    if vi_rgb is not None and fused_rgb is not None:
        h = min(vi_rgb.shape[0], fused_rgb.shape[0], fu.shape[0])
        w = min(vi_rgb.shape[1], fused_rgb.shape[1], fu.shape[1])
        vi_rgb = vi_rgb[:h, :w]
        fused_rgb = fused_rgb[:h, :w]
        metrics.update(
            {
                "RGB_SSIM": rgb_ssim(vi_rgb, fused_rgb),
                "CIEDE2000": ciede2000_mean(vi_rgb, fused_rgb),
                "Colorfulness": colorfulness(fused_rgb),
            }
        )
    return metrics


def paired_statistics(base_rows, cand_rows, metric: str, seed: int = 42, n_boot: int = 1000):
    base = {r["name"]: float(r[metric]) for r in base_rows if r.get("name") and not str(r["name"]).startswith("__")}
    cand = {r["name"]: float(r[metric]) for r in cand_rows if r.get("name") and not str(r["name"]).startswith("__")}
    names = sorted(set(base) & set(cand))
    if not names:
        raise ValueError(f"no overlapping rows for metric {metric}")
    diffs = np.asarray([cand[n] - base[n] for n in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(max(1, n_boot)):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(sample.mean())
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    p_value = wilcoxon_signed_rank_pvalue(diffs)
    std = diffs.std(ddof=1) if len(diffs) > 1 else 0.0
    effect = float(diffs.mean() / (std + EPS))
    return {
        "metric": metric,
        "n": len(names),
        "mean_delta": float(diffs.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_p": p_value,
        "effect_size": effect,
        "gain_ratio": float((diffs > 0).mean()),
    }


def _normal_cdf(x: float):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilcoxon_signed_rank_pvalue(diffs: np.ndarray):
    """Two-sided Wilcoxon signed-rank normal approximation without SciPy."""
    nonzero = np.asarray([d for d in diffs if abs(d) > EPS], dtype=np.float64)
    n = len(nonzero)
    if n == 0:
        return 1.0
    order = np.argsort(np.abs(nonzero))
    ranks = np.empty(n, dtype=np.float64)
    sorted_abs = np.abs(nonzero)[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(sorted_abs[j] - sorted_abs[i]) <= EPS:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    w_pos = ranks[nonzero > 0].sum()
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return 1.0
    z = (abs(w_pos - mean) - 0.5) / math.sqrt(var)
    return float(2.0 * (1.0 - _normal_cdf(abs(z))))


def _read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate S4Fusion-HR structure/color metrics or paired statistics.")
    parser.add_argument("--ir-path")
    parser.add_argument("--vi-path")
    parser.add_argument("--fused-path")
    parser.add_argument("--fused-rgb-path", default="")
    parser.add_argument("--out-csv", default="results_metrics/extended_metrics.csv")
    parser.add_argument("--use-y", action="store_true")
    parser.add_argument("--baseline-csv", default="")
    parser.add_argument("--candidate-csv", default="")
    parser.add_argument("--stats-out", default="")
    parser.add_argument("--stats-metrics", default="SF,AG,EOR,ECS,OBR,RGB_SSIM,CIEDE2000,Colorfulness")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    if args.baseline_csv and args.candidate_csv:
        base_rows = _read_csv_rows(Path(args.baseline_csv))
        cand_rows = _read_csv_rows(Path(args.candidate_csv))
        stat_rows = []
        for metric in [m.strip() for m in args.stats_metrics.split(",") if m.strip()]:
            if metric in base_rows[0] and metric in cand_rows[0]:
                stat_rows.append(paired_statistics(base_rows, cand_rows, metric=metric, n_boot=args.bootstrap))
        out = Path(args.stats_out or args.out_csv)
        fields = ["metric", "n", "mean_delta", "ci95_low", "ci95_high", "wilcoxon_p", "effect_size", "gain_ratio"]
        _write_csv(out, stat_rows, fields)
        print(f"wrote {out} ({len(stat_rows)} paired stats)")
        return

    ir_dir = Path(args.ir_path)
    vi_dir = Path(args.vi_path)
    fu_dir = Path(args.fused_path)
    rgb_dir = Path(args.fused_rgb_path) if args.fused_rgb_path else None
    rows = []
    for name in _list_images(fu_dir):
        if not (ir_dir / name).exists() or not (vi_dir / name).exists():
            continue
        ir = read_gray(ir_dir / name, use_y=False)
        vi = read_gray(vi_dir / name, use_y=args.use_y)
        fu = read_gray(fu_dir / name, use_y=False)
        vi_rgb = _read_rgb(vi_dir / name) if rgb_dir and (rgb_dir / name).exists() else None
        fu_rgb = _read_rgb(rgb_dir / name) if rgb_dir and (rgb_dir / name).exists() else None
        rows.append({"name": name, **evaluate_extended_pair(ir, vi, fu, vi_rgb=vi_rgb, fused_rgb=fu_rgb)})

    if not rows:
        raise RuntimeError("no matched images for extended metrics")
    fields = ["name"] + [k for k in rows[0] if k != "name"]
    mean_row = {"name": "__mean__"}
    std_row = {"name": "__std__"}
    for field in fields[1:]:
        vals = [float(r[field]) for r in rows if r.get(field) is not None]
        mean_row[field] = float(np.mean(vals))
        std_row[field] = float(np.std(vals))
    _write_csv(Path(args.out_csv), rows + [mean_row, std_row], fields)
    print(f"wrote {args.out_csv} ({len(rows)} images)")
    print("mean:", mean_row)


if __name__ == "__main__":
    main()
