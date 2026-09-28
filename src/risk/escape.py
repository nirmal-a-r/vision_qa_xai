"""
escape.py
=========
The two defect-escape events used throughout the project, defined in one place.

Why this module exists
----------------------
Earlier versions of the repository used three different notions of "escape"
(a fraction of missed boxes in ``conformal.py``, an "any box missed" event in the
old ``spi.py`` and a part-level event in the triage code). They are different
risks with different numbers. The paper uses exactly two, both 0/1 per image:

* **Part escape (primary).** A defective part is auto-accepted, i.e. no detection
  on the image clears the threshold. This is what reaches the customer.
* **Localised escape (secondary).** Some ground-truth defect has no matching
  detection (IoU >= ``iou_thresh``) that clears the threshold. Stricter; needed
  for rework and root-cause analysis.

Each event reduces to ONE scalar per defective image, its *escape score* ``c``:

    part:       c = max score over all detections on the image
    localised:  c = min over ground-truth boxes g of
                    max score over detections matching g

with ``c = -inf`` when the relevant set is empty (the part always escapes).
Then the escape loss at threshold ``lam`` is simply ``1{c < lam}``, so conformal
risk control on escape is exactly split-conformal on the scalar ``c``.

Clean images carry no defect and cannot escape; their score is ``nan`` and they
are excluded from calibration (they still matter for review-load statistics).
"""

from __future__ import annotations

import numpy as np

NEG_INF = float("-inf")
KINDS = ("part", "localized")


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes ``a`` (N, 4) and ``b`` (M, 4), both xyxy. Returns (N, M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _gt(record) -> np.ndarray:
    return np.asarray(record.get("gt_boxes", []), dtype=float).reshape(-1, 4)


def _pred(record):
    pb = np.asarray(record.get("pred_boxes", []), dtype=float).reshape(-1, 4)
    ps = np.asarray(record.get("pred_scores", []), dtype=float).reshape(-1)
    return pb, ps


def part_escape_score(record) -> float:
    """Highest detection score on the image (``-inf`` if none, ``nan`` if clean)."""
    if _gt(record).shape[0] == 0:
        return float("nan")
    _, ps = _pred(record)
    return float(ps.max()) if ps.size else NEG_INF


def localized_escape_score(record, iou_thresh: float = 0.5) -> float:
    """Score of the weakest-covered ground-truth defect (``-inf`` if any is missed).

    An image avoids localised escape at threshold ``lam`` only if EVERY
    ground-truth box has a matching detection scoring at least ``lam``.
    """
    gt = _gt(record)
    if gt.shape[0] == 0:
        return float("nan")
    pb, ps = _pred(record)
    if pb.shape[0] == 0:
        return NEG_INF
    hit = box_iou_matrix(gt, pb) >= iou_thresh
    per_gt = [ps[hit[g]].max() if hit[g].any() else NEG_INF for g in range(gt.shape[0])]
    return float(min(per_gt))


def escape_score(record, kind: str = "part", iou_thresh: float = 0.5) -> float:
    if kind == "part":
        return part_escape_score(record)
    if kind == "localized":
        return localized_escape_score(record, iou_thresh)
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def escape_scores(records, kind: str = "part", iou_thresh: float = 0.5) -> np.ndarray:
    """Escape scores of the DEFECTIVE images only (clean images dropped)."""
    s = np.array([escape_score(r, kind, iou_thresh) for r in records], dtype=float)
    return s[~np.isnan(s)]


def empirical_escape(scores: np.ndarray, threshold: float) -> float:
    """Realised escape rate ``mean(c < threshold)`` on defective images."""
    s = np.asarray(scores, dtype=float)
    return float(np.mean(s < threshold)) if s.size else float("nan")


def crc_escape_threshold(scores: np.ndarray, alpha: float) -> float:
    """Exact conformal-risk-control threshold for 0/1 escape on real data only.

    With the m calibration scores sorted, c_(1) <= ... <= c_(m), and a fresh
    exchangeable defective part, P(c_new < c_(j)) <= j / (m + 1). The largest
    certifiable threshold is therefore c_(j*) with j* = floor(alpha (m + 1)).

    Returns ``-inf`` when j* = 0: no threshold is certifiable, every part must go
    to review (a refusal). Equivalent to ``conformal.conformal_risk_control`` on
    the 0/1 loss, written as a quantile. (The pre-2026-09 version in ``spi.py``
    returned c_(j*-1): valid, but conservative by one order statistic.)
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    c = np.sort(np.asarray(scores, dtype=float))
    m = c.size
    j = int(np.floor(alpha * (m + 1) + 1e-9))
    if m == 0 or j < 1:
        return NEG_INF
    return float(c[j - 1])


# Sentinel for "no detection at all" once ties are broken. Detector confidences
# live in [0, 1], so -1 sorts below every real score while staying finite, which
# lets missed defects be tie-broken among themselves like any other score.
MISS_SENTINEL = -1.0


def break_ties(scores, rng, scale: float = 1e-7, sentinel: float = MISS_SENTINEL,
               u=None) -> np.ndarray:
    """Random tie-breaking, as assumed by SPI and stated in the plan (Section 8.5).

    Adds an independent Uniform(0, scale) perturbation to every score and maps
    ``-inf`` (defect never detected) to ``sentinel`` first, so ties - including
    among missed defects - are broken uniformly at random. ``scale`` must be below
    half the score resolution (detections are cached to 1e-5), so distinct scores
    never swap order. ``nan`` (clean part) stays ``nan``. Pass the same ``u`` to
    perturb two arrays describing the same parts identically.
    """
    s = np.array(scores, dtype=float, copy=True)
    s[np.isneginf(s)] = sentinel
    if u is None:
        u = rng.uniform(0.0, scale, size=s.shape)
    return s + u
