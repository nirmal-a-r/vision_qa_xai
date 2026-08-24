"""
figures.py
==========
Publication figure toolkit.

One place for every plot in the paper so that font sizes, colours, DPI and
export format are decided once instead of drifting between cells. Journals
reject figures for inconsistent typography far more often than for bad
statistics, and a per-cell `plt.plot` habit guarantees that inconsistency.

Palette is colour-blind safe (Okabe-Ito) and stays legible in greyscale print,
which still matters for IEEE/Elsevier hardcopy.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

# Okabe-Ito: distinguishable under all common forms of colour vision deficiency
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
         "#E69F00", "#56B4E9", "#F0E442", "#000000"]

FIGDIR = "figures"


def set_style(base: int = 11):
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,          # 300 dpi is the usual journal floor
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": base - 1,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": mpl.cycler(color=OKABE),
    })


def save(fig, name: str, figdir: str = None):
    """Write PDF (vector, for the manuscript) and PNG (for the notebook)."""
    d = figdir or FIGDIR
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"))
    return os.path.join(d, f"{name}.png")


# ---------------------------------------------------------------------------
# Image-strip helpers (per-stage before/after)
# ---------------------------------------------------------------------------

def image_strip(images: dict, title: str = "", boxes=None, cmaps=None,
                per_panel: float = 2.6):
    """Render an ordered {label: image} dict as one row, optional GT boxes.

    Used for every preprocessing stage so the reader sees the actual pixels at
    each step rather than a claim about them.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(per_panel * n, per_panel + 0.6))
    if n == 1:
        axes = [axes]
    for ax, (label, im) in zip(axes, images.items()):
        cmap = (cmaps or {}).get(label, "gray" if im.ndim == 2 else None)
        ax.imshow(im, cmap=cmap)
        ax.set_title(label.replace("_", " "), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if boxes is not None:
            for x1, y1, x2, y2 in boxes:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       ec=OKABE[1], lw=1.4))
    if title:
        fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    return fig


def histogram_panel(images: dict, title: str = ""):
    """Intensity histograms for the same stages - shows *why* a stage helped."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (label, im) in enumerate(images.items()):
        g = im if im.ndim == 2 else im.mean(axis=2)
        ax.hist(g.ravel(), bins=64, range=(0, 255), histtype="step",
                label=label.replace("_", " "), color=OKABE[i % len(OKABE)])
    ax.set_xlabel("intensity"); ax.set_ylabel("pixel count")
    ax.set_title(title or "Intensity distribution by stage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Result plots
# ---------------------------------------------------------------------------

def grouped_bars(categories, series: dict, ylabel: str, title: str = "",
                 annotate: bool = True, figsize=(7.2, 3.6)):
    fig, ax = plt.subplots(figsize=figsize)
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(categories))
    for i, (label, vals) in enumerate(series.items()):
        pos = x + (i - (n - 1) / 2) * w
        b = ax.bar(pos, vals, w, label=label, color=OKABE[i % len(OKABE)])
        if annotate:
            ax.bar_label(b, fmt="%.3f", fontsize=7, padding=1.5)
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout()
    return fig


def risk_coverage_plot(alphas, observed, stds=None, refused=None,
                       title="Conformal risk control"):
    """The headline validation figure: observed risk vs target, with y=x.

    Points on or below the diagonal are the guarantee holding. Refused
    operating points are drawn as open markers on the diagonal - the method
    declining to issue a certificate is a *result*, not missing data, and
    hiding them would overstate coverage.
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    a = np.asarray(alphas, dtype=float)
    o = np.asarray(observed, dtype=float)
    lim = max(a.max(), np.nanmax(o)) * 1.15

    ax.plot([0, lim], [0, lim], ls="--", c="0.5", lw=1.2, label="target (y = x)")
    ax.fill_between([0, lim], [0, lim], [lim, lim], color=OKABE[1], alpha=0.07)
    ax.text(lim * 0.45, lim * 0.80, "violation region", color=OKABE[1],
            fontsize=8, ha="center")

    ok = ~np.isnan(o)
    if stds is not None:
        ax.errorbar(a[ok], o[ok], yerr=np.asarray(stds, float)[ok], fmt="o",
                    color=OKABE[0], capsize=3, ms=6, label="observed (mean +/- sd)")
    else:
        ax.plot(a[ok], o[ok], "o", color=OKABE[0], ms=6, label="observed")
    if refused is not None and np.any(refused):
        r = np.asarray(refused, bool)
        ax.plot(a[r], a[r], "o", mfc="none", mec=OKABE[3], ms=9, mew=1.6,
                label="refused (alpha unreachable)")

    ax.set_xlabel(r"target risk $\alpha$")
    ax.set_ylabel("observed escape rate")
    ax.set_title(title)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def pareto_plot(review_loads, escapes, alpha=None, labels=None,
                title="Review load vs defect escape"):
    """The industrial trade-off curve: how much human review buys how much safety."""
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(np.asarray(review_loads) * 100, escapes, "o-", color=OKABE[0], ms=5)
    if alpha is not None:
        ax.axhline(alpha, ls="--", c=OKABE[1], lw=1.3,
                   label=fr"certified bound $\alpha$ = {alpha}")
        ax.legend(fontsize=8)
    if labels:
        for xx, yy, t in zip(review_loads, escapes, labels):
            ax.annotate(t, (xx * 100, yy), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("human review load (% of parts)")
    ax.set_ylabel("defect escape rate")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def curve_panel(curves: dict, xlabel, ylabel, title="", figsize=(5.4, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, (x, y)) in enumerate(curves.items()):
        ax.plot(x, y, label=label, color=OKABE[i % len(OKABE)])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Schematic diagrams (drawn, not screenshotted - stays vector in the PDF)
# ---------------------------------------------------------------------------

def _box(ax, xy, w, h, text, fc, fontsize=8.5, tc="white"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="none"))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, weight="bold", zorder=3)


def _arrow(ax, p0, p1, color="0.35"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=1.3, zorder=2))


def architecture_diagram():
    """End-to-end system schematic for the paper's method figure."""
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.5); ax.axis("off"); ax.grid(False)

    row = 1.35
    h = 0.85
    specs = [
        (0.10, 1.55, "Acquisition\n(line camera)", "#4C4C4C"),
        (1.80, 1.55, "CCE encoding\n[raw | CLAHE | HF]", OKABE[4]),
        (3.50, 1.55, "Detector\nRT-DETR (NMS-free)", OKABE[0]),
        (5.20, 1.55, "Conformal\ncalibration", OKABE[2]),
        (6.90, 1.55, "Faithfulness\naudit", OKABE[3]),
        (8.60, 1.55, "Risk-controlled\ntriage", OKABE[1]),
    ]
    for x, w, t, c in specs:
        _box(ax, (x, row), w, h, t, c)
    for i in range(len(specs) - 1):
        x0 = specs[i][0] + specs[i][1]
        _arrow(ax, (x0, row + h / 2), (specs[i + 1][0], row + h / 2))

    # triage outputs
    outs = [(10.35, 2.55, "ACCEPT", OKABE[2]),
            (10.35, 1.45, "REVIEW", OKABE[4]),
            (10.35, 0.35, "REJECT", OKABE[1])]
    for x, y, t, c in outs:
        _box(ax, (x, y), 1.05, 0.6, t, c, fontsize=8)
        _arrow(ax, (specs[-1][0] + specs[-1][1], row + h / 2), (x, y + 0.3))

    ax.text(5.75, 0.62, r"calibration block is disjoint from training  |  "
                        r"guarantee: $\mathbb{E}[\mathrm{escape}] \leq \alpha$",
            ha="center", fontsize=9, style="italic", color="0.3")
    ax.text(2.65, 0.95, "zero added latency", ha="center", fontsize=7.5,
            color=OKABE[4], style="italic")
    fig.tight_layout()
    return fig


def cce_diagram():
    """Why the two duplicated grey channels are wasted, and what replaces them."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
    for ax, (title, chans, colors, note) in zip(axes, [
        ("Standard pipeline", ["grey", "grey", "grey"],
         ["#9E9E9E"] * 3, "2 of 3 channels carry\nzero information"),
        ("Complementary Channel Encoding", ["raw", "CLAHE", "HF residual"],
         [OKABE[0], OKABE[4], OKABE[2]], "same shape, same FLOPs,\nsame latency"),
    ]):
        ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.axis("off"); ax.grid(False)
        for i, (c, col) in enumerate(zip(chans, colors)):
            _box(ax, (0.45 + i * 1.05, 1.25), 0.92, 0.85, c, col, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.text(2.0, 0.55, note, ha="center", fontsize=8, color="0.35",
                style="italic")
    fig.tight_layout()
    return fig
