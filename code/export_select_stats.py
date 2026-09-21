#!/usr/bin/env python3
"""R2 (review round 2026-09-18): measure the trained selector `s`.

Why: the manuscript calls CGA a *hard* per-pixel commitment, but the operator is
    K = s * I_ir + (1 - s) * I_vi,  s = sigmoid(h_s(.) + 2p)
a convex blend. At initialization the final layer is zeroed, so s = sigmoid(2p) is
0.881 / 0.119 -- a soft 88/12 blend, not a pick. Whether training drives s to the
{0, 1} endpoints is an empirical question the manuscript never answers. This script
answers it from the released checkpoint.

The quantity that decides the wording is the *commitment hardness*
    h = min(s, 1 - s)
which is 0 for a true pick and 0.1192 at initialization. If h stays near 0.119 the
operator is a blend; if training drives it toward 0 the word "hard" is earned.

Stratification (the first version of this script stratified by the 66th percentile
of the max-normalized prior, which turned out to be uninformative: that set is
dominated by near-zero prior values, and the learned map M is ~0 there). The
reported strata are the ones that actually matter:
    all          -- every pixel
    acts         -- pixels where the mechanism acts, M > 0.5
    acts_top     -- top decile of M (the strongest interventions)
    prior_top    -- top tercile of the conflict prior (reported as a contrast)

Also reported: the M distribution, the Spearman correlation between M and the prior,
and a cross-check of s against the algebraic inverse (commit - vi) / (ir - vi).

Run inside WSL with the training interpreter:
  /mnt/e/.../code/.venv-brss/bin/python export_select_stats.py --checkpoint \
      /mnt/e/.../code/checkpoints/abl_CGA_str_s42.pt
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_brss import build_model, load_checkpoint
from train_gates import _resolve_device
from evaluate_brss import _load_pair, _pad_to_valid

CODE = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/code")
MANIFEST = Path("/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/dataset/manifests/m3fd_test.csv")
OUT = CODE / "results" / "select_stats"

NBINS = 100
INIT_S_HI, INIT_S_LO = 0.8808, 0.1192     # sigmoid(+/-2), the zero-init selector value
INIT_HARDNESS = 0.1192                    # min(0.8808, 0.1192)
STRATA = ("all", "acts", "acts_top", "prior_top")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", default="s42")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--data-root", default="../dataset")
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N ids")
    return ap.parse_args()


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    print(f"ids: {len(rows)}", flush=True)

    device = _resolve_device("auto")
    model = build_model(SimpleNamespace(
        depths="1,2,1", teacher_depths="1,2,1", fusion_scales="none", gate_type="scalar",
        dynamic_gate_hidden=16, global_op="scan", boundary_mode="none", boundary_hidden=16,
        use_cga=True, cga_hidden=16)).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    gate = float(torch.tanh(model.vmunet.cga.gate_scale).item())
    print(f"trained gate: tanh(gate_scale) = {gate:.6f}", flush=True)

    bins = np.linspace(0.0, 1.0, NBINS + 1)
    hist = {k: np.zeros(NBINS, dtype=np.int64) for k in STRATA}
    n = {k: 0 for k in STRATA}
    s_sum = {k: 0.0 for k in STRATA}
    hard_sum = {k: 0.0 for k in STRATA}
    sat = {k: 0 for k in STRATA}
    m_all_sum = 0.0
    m_hist = np.zeros(NBINS, dtype=np.int64)
    # Spearman between M and the prior, pooled via rank accumulation is not exact;
    # instead accumulate the 2x2 concordance needed for a good approximation: we
    # simply collect per-image Spearman and average (reported as the mean).
    rho_list = []
    inv_abs_dev, inv_valid, inv_total = 0.0, 0, 0
    per_image = []

    def spearman(a: np.ndarray, b: np.ndarray, step: int = 7) -> float:
        aa = a.ravel()[::step]
        bb = b.ravel()[::step]
        ra = np.argsort(np.argsort(aa))
        rb = np.argsort(np.argsort(bb))
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / den) if den > 0 else float("nan")

    droot = Path(args.data_root)
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            sid = row["sample_id"]
            ir, vi, *_ = _load_pair(droot, row)
            ir, vi = ir.to(device), vi.to(device)
            ip, vp, h0, w0 = _pad_to_valid(ir, vi)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _fused, aux = model(ip, vp, return_aux=True)
            cga = aux["cga"]
            if cga is None:
                raise RuntimeError("checkpoint has no CGA module -- pass a CGA checkpoint")

            s = cga["select"][..., :h0, :w0].squeeze().float().cpu().numpy()
            m = cga["conflict_pred"][..., :h0, :w0].squeeze().float().cpu().numpy()
            commit = cga["commit"][..., :h0, :w0].squeeze().float().cpu().numpy()
            prior = cga["conflict_target"][..., :h0, :w0].squeeze().float().cpu().numpy()
            irn = ir[..., :h0, :w0].squeeze().float().cpu().numpy()
            vin = vi[..., :h0, :w0].squeeze().float().cpu().numpy()

            hard = np.minimum(s, 1.0 - s)
            masks = {
                "all": np.ones_like(s, dtype=bool),
                "acts": m > 0.5,
                "acts_top": m >= np.percentile(m, 90),
                "prior_top": prior >= np.percentile(prior, 66),
            }
            for k, mk in masks.items():
                cnt = int(mk.sum())
                if cnt == 0:
                    continue
                hist[k] += np.histogram(s[mk], bins=bins)[0]
                n[k] += cnt
                s_sum[k] += float(s[mk].sum())
                hard_sum[k] += float(hard[mk].sum())
                sat[k] += int(((s[mk] > 0.9) | (s[mk] < 0.1)).sum())

            m_all_sum += float(m.mean())
            m_hist += np.histogram(m, bins=bins)[0]
            rho_list.append(spearman(m, prior))

            denom = irn - vin
            sel = np.abs(denom) > 0.05
            if sel.any():
                s_hat = (commit - vin)[sel] / denom[sel]
                inv_abs_dev += float(np.abs(s_hat - s[sel]).mean())
                inv_valid += int(sel.sum())
            inv_total += s.size

            per_image.append({
                "sample_id": sid,
                "s_mean": round(float(s.mean()), 6),
                "hardness_mean": round(float(hard.mean()), 6),
                "sat_frac": round(float(((s > 0.9) | (s < 0.1)).mean()), 6),
                "M_mean": round(float(m.mean()), 6),
                "M_p99": round(float(np.percentile(m, 99)), 6),
                "frac_M_gt_0.5": round(float((m > 0.5).mean()), 6),
                "rho_M_prior": round(rho_list[-1], 6),
            })
            if i % 25 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)

    with (OUT / f"select_hist_{args.tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["bin_left", "bin_right", "count_all", "count_acts",
                     "count_acts_top", "count_prior_top", "frac_all", "frac_acts"])
        for b in range(NBINS):
            wr.writerow([f"{bins[b]:.4f}", f"{bins[b+1]:.4f}",
                         int(hist["all"][b]), int(hist["acts"][b]),
                         int(hist["acts_top"][b]), int(hist["prior_top"][b]),
                         f"{hist['all'][b]/max(n['all'],1):.8f}",
                         f"{hist['acts'][b]/max(n['acts'],1):.8f}"])
    with (OUT / f"select_per_image_{args.tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
        wr.writeheader()
        wr.writerows(per_image)
    with (OUT / f"select_M_hist_{args.tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["bin_left", "bin_right", "count_M"])
        for b in range(NBINS):
            wr.writerow([f"{bins[b]:.4f}", f"{bins[b+1]:.4f}", int(m_hist[b])])

    print()
    print(f"[s] gate tanh(gate_scale)          = {gate:.6f}")
    print(f"[s] init reference: s = 0.8808/0.1192, hardness = {INIT_HARDNESS:.4f}")
    print()
    print(f"{'stratum':<11}{'pixels':>12}{'share':>9}{'mean s':>10}{'hardness':>10}{'sat [0.1,0.9]':>15}")
    for k in STRATA:
        if n[k] == 0:
            print(f"{k:<11}{'0':>12}{'-':>9}{'(empty)':>10}")
            continue
        print(f"{k:<11}{n[k]:>12}{n[k]/n['all']*100:>8.1f}%"
              f"{s_sum[k]/n[k]:>10.4f}{hard_sum[k]/n[k]:>10.4f}{sat[k]/n[k]:>15.4f}")
    print()
    print(f"[s] M (learned conflict map): mean = {m_all_sum/len(rows):.4f}, "
          f"pooled P(M>0.5) = {n['acts']/max(n['all'],1):.4f}, "
          f"P(M in top decile) = {n['acts_top']/max(n['all'],1):.4f}")
    print(f"[s] mean per-image Spearman rho(M, prior) = {np.nanmean(rho_list):.4f}")
    print(f"[s] applied weight w = tanh(gate)*M: mean over pixels = "
          f"{gate*m_all_sum/len(rows):.4f}")
    print(f"[s] algebraic cross-check s=(commit-vi)/(ir-vi): valid "
          f"{inv_valid/max(inv_total,1)*100:.1f}% of pixels, "
          f"mean|dev| = {inv_abs_dev/max(len(rows),1):.3e}")
    print(f"[s] wrote 3 CSVs under {OUT}/ with tag {args.tag}")


if __name__ == "__main__":
    main()
