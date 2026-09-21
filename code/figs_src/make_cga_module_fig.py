#!/usr/bin/env python3
"""Rebuild Fig. 2 (the CGA module diagram) at a readable type size.

The previous vector figure was one wide, short row whose labels became
unreadable at 100% print scale. This version draws at the figure's final
physical size (~4.85 x 3.4 in) so the type is ~7 pt on the page, uses short
node labels with the equations moved into a legend strip, and leaves margins
wide enough that no annotation can collide with a node.

Run with any Python that has matplotlib.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

W, H = 4.85, 3.45
FS = 7.2
FS_S = 6.4

INK = "#111111"
FILL_FIX = "#e8eef7"
FILL_LEARN = "#eaf2e4"
FILL_BACK = "#f2f0f6"
FILL_OUT = "#fdf1e3"
FILL_IO = "#eef0f2"
EDGE = "#465569"

fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, fill, fs=FS, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
        linewidth=0.7, edgecolor=EDGE, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.4)
    return (x, y, w, h)


def arrow(a, b, side="right", rad=0.0, ls="solid"):
    if side == "right":
        p, q = (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2)
    elif side == "left":
        p, q = (a[0], a[1] + a[3] / 2), (b[0] + b[2], b[1] + b[3] / 2)
    elif side == "down":
        p, q = (a[0] + a[2] / 2, a[1]), (b[0] + b[2] / 2, b[1] + b[3])
    else:
        p, q = (a[0] + a[2] / 2, a[1] + a[3]), (b[0] + b[2] / 2, b[1])
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=6.0, linewidth=0.7,
        color=EDGE, linestyle=ls, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


def tag(x, y, s):
    """Small annotation placed clear of every node."""
    ax.text(x, y, s, ha="center", va="center", fontsize=FS_S, color=INK,
            style="italic", zorder=4)


BW, BH = 17.5, 15.0
def poly_arrow(points, ls=(0, (3, 2))):
    """Orthogonal dashed path with a single arrowhead at the last point."""
    ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
            color=EDGE, linewidth=0.7, linestyle=ls, zorder=1,
            solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=6.0,
        linewidth=0.7, color=EDGE, zorder=1))


# ------------------------------------------------------------------- row 1
y1 = 71.0
n_src = box(1.0, y1, 14.0, BH, "source pair\n$I_{ir},\\,I_{vi}$", FILL_IO)
n_grad = box(21.0, y1, BW, BH, "fixed Sobel\n$\\rightarrow g_{ir},g_{vi},c$", FILL_FIX)
n_kap = box(43.5, y1, BW, BH, "conflict prior\n$\\rightarrow \\hat\\kappa$", FILL_FIX)
n_head = box(66.0, y1, BW, BH, "conflict head $h_c$\n(learned)\n$\\rightarrow M$", FILL_LEARN)
n_M = box(88.5, y1, 10.5, BH, "support\n$M$", FILL_LEARN)

arrow(n_src, n_grad)
arrow(n_grad, n_kap)
arrow(n_kap, n_head)
arrow(n_head, n_M)

# ------------------------------------------------------------------- row 2
y2 = 42.0
n_src2 = box(1.0, y2, 14.0, BH, "source pair\n(again)", FILL_IO)
n_sel = box(21.0, y2, BW, BH, "selector $h_s$\n(learned, zero-init.)\n$\\rightarrow s$", FILL_LEARN)
n_K = box(43.5, y2, BW, BH, "source mixture\n$K=s\\,I_{ir}+(1{-}s)\\,I_{vi}$", FILL_OUT)
n_back = box(66.0, y2, BW, BH, "S4Fusion\n(frozen weights)\n$\\rightarrow Y$", FILL_BACK)
n_F = box(88.5, y2, 10.5, BH, "output\n$F$", FILL_OUT)

arrow(n_src2, n_sel)
arrow(n_sel, n_K)
arrow(n_back, n_F)
arrow(n_K, n_F, rad=-0.40)
arrow(n_M, n_F, side="down")

# frozen backbone path: the source pair reaches S4Fusion with no learned head
# in between. Routed below row 1 and above row 2 so it crosses nothing.
poly_arrow([(8.0, y1), (8.0, 64.5), (74.75, 64.5), (74.75, y2 + BH)])
tag(44.0, 67.6, "frozen backbone path: no learned parameter on this route")

# --------------------------------------------------------------- legend strip
ax.plot([1.0, 99.0], [30.0, 30.0], color=EDGE, linewidth=0.5, zorder=1)
ax.text(1.0, 24.0,
        "conflict prior  $\\kappa=\\min(g_{ir},g_{vi})\\,\\frac{1-c}{2}$,  "
        "$\\hat\\kappa$ = per-image max-normalised $\\kappa$",
        fontsize=FS_S, ha="left", va="center", color=INK)
ax.text(1.0, 17.5,
        "selector  $s=\\sigma(h_s[\\,I_{ir},I_{vi},g_{ir},g_{vi}\\,]+2p)$,  "
        "$p=\\pm1$ names the larger local gradient",
        fontsize=FS_S, ha="left", va="center", color=INK)
ax.text(1.0, 11.0,
        "combination  $w=\\tanh(\\alpha)\\,M$,  $F=Y+w\\odot(K-Y)$;  "
        "$\\alpha$ and the selector head are zero-initialised",
        fontsize=FS_S, ha="left", va="center", color=INK)
ax.text(1.0, 4.0,
        "At load $\\alpha=0\\Rightarrow w\\equiv0\\Rightarrow F\\equiv Y$: the official checkpoint is reproduced exactly.",
        fontsize=FS_S, ha="left", va="center", color=INK)

fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
out = "/mnt/e/lunwen/S4Fusion-main/S4Fusion-main-Innovation-2026/latex/sn-article/figs/cga_module.pdf"
fig.savefig(out, format="pdf", dpi=300)
print("wrote", out, f"({W}x{H} in, base type {FS} pt)")
