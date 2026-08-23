"""
figures.py
==========
Publication figures, one function each, all returning a matplotlib Figure.

House style is set once in `use_paper_style()`: vector-friendly, colour-blind
safe, no chart junk, fonts large enough to survive a two-column reduction. Every
figure saves as both PDF (for LaTeX) and PNG (for the notebook and slides).

The palette is Okabe-Ito, which stays distinguishable under the three common
forms of colour blindness and in greyscale print - journals still print
greyscale, and a figure whose argument depends on distinguishing red from green
is a figure half the readers cannot check.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito colour-blind safe palette
OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000", "grey": "#7F7F7F",
}
SERIES = [OKABE["blue"], OKABE["orange"], OKABE["green"],
          OKABE["vermillion"], OKABE["purple"], OKABE["sky"]]


def use_paper_style():
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,     # embed as TrueType so editors can select text
        "ps.fonttype": 42,
    })


def save(fig, name, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# F1  Risk-coverage validation: the central claim
# ---------------------------------------------------------------------------

def fig_risk_coverage(risk_results, out_dir="figures", name="f1_risk_coverage"):
    """Target alpha vs realised escape risk, per dataset.

    The diagonal is the guarantee. Points on or below it are certified; the
    figure's whole job is to let a reader check that at a glance, so the
    feasible region is shaded rather than left implicit.
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    lim = 0.0
    for i, (tag, v) in enumerate(sorted(risk_results.items())):
        alphas, means, stds = [], [], []
        for a_str, r in sorted(v["repeated"].items(), key=lambda kv: float(kv[0])):
            if r["mean_escape_test"] is None:
                continue
            alphas.append(float(a_str))
            means.append(r["mean_escape_test"])
            stds.append(r["std"])
        if not alphas:
            continue
        lim = max(lim, max(alphas + means))
        label = tag.replace("_yolov8s", "").replace("_rtdetr-l", "").replace("_", "-")
        ax.errorbar(alphas, means, yerr=stds, marker="o", ms=5, capsize=3,
                    color=SERIES[i % len(SERIES)], label=label, lw=1.6)

    lim = max(lim * 1.15, 0.25)
    ax.fill_between([0, lim], [0, lim], [lim, lim], color=OKABE["vermillion"],
                    alpha=0.07, lw=0, zorder=0)
    ax.plot([0, lim], [0, lim], ls="--", c=OKABE["black"], lw=1.2,
            label=r"guarantee  $\mathbb{E}[L] \leq \alpha$", zorder=1)
    ax.text(lim * 0.97, lim * 0.55, "violation\nregion", ha="right", va="center",
            fontsize=8.5, color=OKABE["vermillion"], alpha=0.9)

    ax.set_xlabel(r"target risk level  $\alpha$")
    ax.set_ylabel(r"realised escape risk on test")
    ax.set_title("Conformal risk control holds on real detector outputs")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, save(fig, name, out_dir)


# ---------------------------------------------------------------------------
# F2  Drift: static vs adaptive
# ---------------------------------------------------------------------------

def fig_drift(static_losses, adaptive_losses, adaptive_lambdas, alpha,
              shift_at, out_dir="figures", name="f2_drift"):
    """The motivating result for the adaptive layer."""
    use_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    def smooth(x, k=100):
        x = np.asarray(x, dtype=float)
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k) / k, mode="valid")

    s_s, s_a = smooth(static_losses), smooth(adaptive_losses)
    ax1.plot(np.arange(len(s_s)), s_s, color=OKABE["vermillion"],
             label="static conformal threshold")
    ax1.plot(np.arange(len(s_a)), s_a, color=OKABE["blue"],
             label="adaptive (ours)")
    ax1.axhline(alpha, ls="--", c=OKABE["black"], lw=1.2, label=rf"target $\alpha={alpha}$")
    ax1.axvline(shift_at, ls=":", c=OKABE["grey"], lw=1.4)
    ax1.text(shift_at, ax1.get_ylim()[1] * 0.94, "  regime shift", fontsize=8.5,
             color=OKABE["grey"], va="top")
    ax1.set_ylabel("escape rate\n(rolling mean)")
    ax1.set_title("A static certificate silently expires when the line drifts")
    ax1.legend(loc="center left")

    ax2.plot(adaptive_lambdas, color=OKABE["blue"], lw=1.4)
    ax2.axvline(shift_at, ls=":", c=OKABE["grey"], lw=1.4)
    ax2.set_ylabel(r"threshold $\lambda_t$")
    ax2.set_xlabel("part index")
    fig.tight_layout()
    return fig, save(fig, name, out_dir)


# ---------------------------------------------------------------------------
# F3  Escape / review-load operating curve
# ---------------------------------------------------------------------------

def fig_pareto(curves, out_dir="figures", name="f3_pareto"):
    """Escape rate against human review load - the axis a plant manager buys on.

    `curves`: {label: (review_loads, escape_rates)}
    """
    use_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for i, (label, (rl, er)) in enumerate(sorted(curves.items())):
        order = np.argsort(rl)
        ax.plot(np.asarray(rl)[order], np.asarray(er)[order], marker="o", ms=4,
                color=SERIES[i % len(SERIES)], label=label)
    ax.set_xlabel("human review load  (fraction of parts)")
    ax.set_ylabel("defect escape rate")
    ax.set_title("What the guarantee costs in inspector time")
    ax.legend()
    fig.tight_layout()
    return fig, save(fig, name, out_dir)


# ---------------------------------------------------------------------------
# F4  Benchmark against published work
# ---------------------------------------------------------------------------

def fig_benchmark(ours, published, dataset="NEU-DET", out_dir="figures",
                  name="f4_benchmark"):
    """Horizontal bars vs published numbers, ours highlighted.

    Published figures are quoted from their papers and were obtained under each
    author's own split, so this is an indicative comparison, not a controlled
    one - the axis label says so rather than leaving the reader to assume
    otherwise.
    """
    use_paper_style()
    items = sorted(published.items(), key=lambda kv: kv[1]) + [(k, v) for k, v in ours.items()]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [OKABE["grey"]] * len(published) + [OKABE["blue"]] * len(ours)

    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(items) + 1.4))
    y = np.arange(len(items))
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"mAP@0.5 on {dataset}  (published numbers use each author's own split)")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{dataset}: this work in context")
    fig.tight_layout()
    return fig, save(fig, name, out_dir)


# ---------------------------------------------------------------------------
# F5  Dataset overview
# ---------------------------------------------------------------------------

def fig_dataset_grid(samples, out_dir="figures", name="f5_datasets"):
    """One annotated example per dataset.

    `samples`: [(dataset_name, image_rgb, [xyxy boxes])]
    """
    use_paper_style()
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, (name_, img, boxes) in zip(axes, samples):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        for (x1, y1, x2, y2) in boxes:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=OKABE["vermillion"], lw=1.6))
        ax.set_title(name_, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    return fig, save(fig, name, out_dir)


# ---------------------------------------------------------------------------
# F6  Per-class (Mondrian) risk control
# ---------------------------------------------------------------------------

def fig_mondrian(per_class, alpha_by_class, out_dir="figures", name="f6_mondrian"):
    """Realised vs targeted risk per defect class.

    `per_class`: {class_name: realised_risk}
    """
    use_paper_style()
    names = list(per_class)
    real = [per_class[k] for k in names]
    targ = [alpha_by_class[k] for k in names]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(5.4, 0.4 * len(names) + 1.6))
    ax.barh(y + 0.18, targ, height=0.34, color=OKABE["grey"], label="target")
    ax.barh(y - 0.18, real, height=0.34, color=OKABE["blue"], label="realised")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("escape risk")
    ax.set_title("Per-class risk budgets: critical defects held tighter")
    ax.legend()
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig, save(fig, name, out_dir)
