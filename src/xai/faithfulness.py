"""
faithfulness.py
===============
Quantitative auditing of attribution maps.

Most XAI in industrial inspection stops at "here is a heatmap, it looks like it
is on the defect". That is unfalsifiable. This module measures whether a
saliency map actually reflects what the model used, with metrics that have
accepted definitions in the literature:

  deletion AUC       Petsiuk, Das & Saenko, "RISE" (BMVC 2018). Progressively
                     erase the highest-attributed pixels; a faithful map makes
                     the score collapse fast, so LOWER is better.

  insertion AUC      Same paper, mirrored: start from a blurred image and add
                     the highest-attributed pixels back. HIGHER is better.
                     Insertion is the more robust of the two because deletion
                     can be gamed by pushing the input off-manifold.

  pointing game      Zhang et al., "Top-down Neural Attention by Excitation
                     Backprop" (IJCV 2018). Does the arg-max of the map land
                     inside a ground-truth box? Binary, cheap, coarse.

  energy pointing    Wang et al., "Score-CAM" (CVPRW 2020). Fraction of total
                     attribution mass falling inside ground-truth boxes.
                     Continuous, and far less noisy than the pointing game.

  sanity check       Adebayo et al., "Sanity Checks for Saliency Maps"
                     (NeurIPS 2018). Re-randomise the model weights; a map that
                     barely changes is measuring the image, not the model, and
                     is worthless regardless of how good it looks.

`faithfulness_score` fuses insertion and energy-pointing into a single [0, 1]
scalar. That scalar is what the triage policy gates on, which is the reason
these metrics have to be per-image rather than dataset-level averages.
"""

from __future__ import annotations

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Perturbation curves
# ---------------------------------------------------------------------------

def _ordered_pixels(heatmap: np.ndarray, descending: bool = True) -> np.ndarray:
    """Flat pixel indices ordered by attribution."""
    flat = heatmap.reshape(-1)
    order = np.argsort(flat)
    return order[::-1] if descending else order


def deletion_curve(
    image: np.ndarray,
    heatmap: np.ndarray,
    score_fn,
    steps: int = 20,
    baseline: str = "blur",
) -> np.ndarray:
    """Model score as the top-attributed pixels are progressively removed.

    `score_fn(img_uint8_rgb) -> float` must return the scalar the explanation is
    supposed to explain (here: the confidence of the top detection).

    A faithful heatmap yields a steeply decreasing curve.
    """
    h, w = heatmap.shape[:2]
    order = _ordered_pixels(heatmap)
    base = _make_baseline(image, baseline)

    n_px = h * w
    cuts = np.linspace(0, n_px, steps + 1).astype(int)
    scores = []
    work = image.copy()
    for k in range(len(cuts) - 1):
        idx = order[cuts[k]:cuts[k + 1]]
        yy, xx = np.unravel_index(idx, (h, w))
        work[yy, xx] = base[yy, xx]
        scores.append(float(score_fn(work)))
    return np.asarray([float(score_fn(image))] + scores)


def insertion_curve(
    image: np.ndarray,
    heatmap: np.ndarray,
    score_fn,
    steps: int = 20,
    baseline: str = "blur",
) -> np.ndarray:
    """Model score as the top-attributed pixels are progressively restored."""
    h, w = heatmap.shape[:2]
    order = _ordered_pixels(heatmap)
    work = _make_baseline(image, baseline)

    n_px = h * w
    cuts = np.linspace(0, n_px, steps + 1).astype(int)
    scores = [float(score_fn(work))]
    for k in range(len(cuts) - 1):
        idx = order[cuts[k]:cuts[k + 1]]
        yy, xx = np.unravel_index(idx, (h, w))
        work[yy, xx] = image[yy, xx]
        scores.append(float(score_fn(work)))
    return np.asarray(scores)


def _make_baseline(image: np.ndarray, kind: str) -> np.ndarray:
    """Reference image for 'information removed'.

    Blur is preferred over a constant fill: a zero/grey patch is far off the
    natural image manifold, and the resulting score drop then measures
    distribution shift rather than the importance of the erased pixels.
    """
    if kind == "blur":
        k = max(3, (min(image.shape[:2]) // 8) | 1)
        return cv2.GaussianBlur(image, (k, k), 0)
    if kind == "zero":
        return np.zeros_like(image)
    if kind == "mean":
        return np.full_like(image, image.mean(axis=(0, 1)).astype(image.dtype))
    raise ValueError(f"unknown baseline {kind!r}")


def normalized_auc(curve: np.ndarray, flat_rel_tol: float = 0.05) -> float:
    """Trapezoidal AUC on a unit x-axis, rescaled by the curve's own range.

    Rescaling makes the number comparable across images whose absolute
    confidences differ, which the triage gate needs since it compares
    faithfulness across a whole production batch.

    The flatness guard is not cosmetic. If perturbing the image barely moves the
    score, the curve is noise and (auc - lo) / (hi - lo) divides one tiny number
    by another, producing a confident-looking value from nothing - an almost
    flat curve was returning 0.95. Whenever the observed swing is below
    `flat_rel_tol` of the curve's own scale we return 0.5, i.e. "this test was
    uninformative for this image", which is the honest answer and keeps the
    gate from trusting a meaningless measurement.
    """
    curve = np.asarray(curve, dtype=float)
    x = np.linspace(0.0, 1.0, len(curve))
    auc = float(np.trapezoid(curve, x)) if hasattr(np, "trapezoid") else float(np.trapz(curve, x))
    lo, hi = float(curve.min()), float(curve.max())
    scale = max(abs(hi), abs(lo), 1e-12)
    if (hi - lo) < flat_rel_tol * scale:
        return 0.5
    return (auc - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Localisation agreement
# ---------------------------------------------------------------------------

def pointing_game(heatmap: np.ndarray, boxes_xyxy) -> bool:
    """True if the map's arg-max falls inside any ground-truth box."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    if boxes.shape[0] == 0:
        return False
    y, x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape[:2])
    inside = (boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3])
    return bool(inside.any())


def energy_pointing_game(heatmap: np.ndarray, boxes_xyxy) -> float:
    """Fraction of total attribution mass inside the ground-truth boxes.

    Chance level is the fraction of image area the boxes cover, so a raw value
    means little on its own - `faithfulness_score` compares against that
    baseline rather than treating 0.5 as neutral.
    """
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    hm = np.clip(np.asarray(heatmap, dtype=float), 0, None)
    total = hm.sum()
    if boxes.shape[0] == 0 or total <= 0:
        return 0.0
    h, w = hm.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        xi2, yi2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = True
    return float(hm[mask].sum() / total)


def box_area_fraction(boxes_xyxy, shape) -> float:
    """Chance level for the energy pointing game."""
    boxes = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    h, w = shape[:2]
    if boxes.shape[0] == 0:
        return 0.0
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        xi1, yi1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        xi2, yi2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = True
    return float(mask.mean())


# ---------------------------------------------------------------------------
# Composite score used by the triage gate
# ---------------------------------------------------------------------------

def faithfulness_score(
    heatmap: np.ndarray,
    boxes_xyxy,
    image: np.ndarray = None,
    score_fn=None,
    steps: int = 12,
    w_insertion: float = 0.5,
) -> float:
    """Per-image faithfulness in [0, 1] for gating the triage policy.

    Combines two complementary signals:

      * localisation - energy pointing game, corrected for the chance level set
        by box area, so a map that simply spreads mass everywhere scores 0.
      * causal       - insertion AUC, which needs `image` and `score_fn`. When
        those are not supplied the score falls back to localisation only, which
        is cheap enough to run on every image in a batch.

    Deliberately excludes deletion AUC: it correlates strongly with insertion
    but is the more off-manifold of the two, so including both would double-count
    the same evidence while importing the weaker metric's artefacts.
    """
    epg = energy_pointing_game(heatmap, boxes_xyxy)
    chance = box_area_fraction(boxes_xyxy, heatmap.shape)
    # Rescale so chance -> 0 and perfect concentration -> 1.
    loc = 0.0 if chance >= 1.0 else float(np.clip((epg - chance) / (1.0 - chance), 0.0, 1.0))

    if image is None or score_fn is None:
        return loc

    ins = normalized_auc(insertion_curve(image, heatmap, score_fn, steps=steps))
    return float(np.clip((1 - w_insertion) * loc + w_insertion * ins, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Sanity check (Adebayo et al.)
# ---------------------------------------------------------------------------

def map_rank_correlation(map_a: np.ndarray, map_b: np.ndarray) -> float:
    """Spearman rank correlation between two saliency maps.

    Used for the model-randomisation sanity check: explain with the trained
    model, re-randomise the weights, explain again. A high correlation means the
    "explanation" is an edge detector in disguise.
    """
    from scipy.stats import spearmanr

    a = np.asarray(map_a, dtype=float).reshape(-1)
    b = np.asarray(map_b, dtype=float).reshape(-1)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    rho, _ = spearmanr(a, b)
    return float(0.0 if np.isnan(rho) else rho)
