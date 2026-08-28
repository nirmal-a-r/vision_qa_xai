"""
spi.py
======
Synthetic-powered escape-risk calibration.

Builds on Bashari, Lotan, Lee, Dobriban & Romano, "Synthetic-Powered Predictive
Inference", NeurIPS 2025 (arXiv:2505.13432). Their setting is conformal
*prediction* (set coverage) on scalar classification scores. Ours is conformal
*risk control* on a detection escape loss. The two are connected by a reduction
that makes their machinery apply directly.

The reduction
-------------
For a defective image i define its **escape confidence**

    c_i = max{ score of a detection that correctly matches a ground-truth box }

(and c_i = -inf when the detector never produces a correct box, so the image
escapes at every threshold). The image-level escape loss is then

    L_i(lambda) = 1{ lambda > c_i },

so the empirical risk is exactly the empirical CDF of the c_i:

    R_hat_n(lambda) = (1/n) sum_i 1{c_i < lambda} = F_hat_n(lambda).

CRC picks the largest lambda with (n/(n+1)) F_hat_n(lambda) + 1/(n+1) <= alpha,
which is a **quantile of the c_i**. Escape-risk control is therefore a quantile
problem on a scalar score - precisely the object SPI transports. Nothing about
SPI has to be re-derived; the loss curve collapses to a scalar per image.

Why the scarcity problem is real here
-------------------------------------
n counts *defective* calibration images, because a clean part cannot escape and
contributes an identically-zero row. KolektorSDD2 has 53 of them, so the
finite-sample term 1/(n+1) alone spends 1.85% of the risk budget and alpha=0.01
is unreachable for any detector whatsoever.

What SPI buys, and what it does not
-----------------------------------
The transporter maps **real -> synthetic** and the threshold is read off the
*synthetic* order statistics, with the real scores entering only through ranks.
Window indices depend on (m, N, beta) alone and never on score values, which is
what stops the effective sample size being inflated. Coverage therefore holds
whatever the synthetic data looks like; only tightness depends on alignment.

That asymmetry is the deployable part: a plant is never asked to trust a
generative model, only to benefit when it happens to be good.

Note on fidelity: the window construction below is reconstructed from the
paper's description of R_r^+/-, not ported from released code. It is validated
empirically by `tests/test_spi.py`, which checks coverage rather than assuming
it.
"""

from __future__ import annotations

import numpy as np

from .conformal import box_iou_matrix

NEG_INF = -1.0e9


# ---------------------------------------------------------------------------
# The scalar score
# ---------------------------------------------------------------------------

def escape_confidence(record, iou_thresh: float = 0.5) -> float:
    """Highest confidence among detections that actually hit a ground-truth box.

    Returns NEG_INF when no detection matches, i.e. the image escapes at every
    threshold. Using -inf rather than 0.0 matters: a missed defect must count as
    escaped even at the lowest threshold, and scoring it 0.0 makes `0.0 < 0.0`
    false so the risk floor silently collapses to zero.
    """
    gt = np.asarray(record.get("gt_boxes", []), dtype=float).reshape(-1, 4)
    if gt.shape[0] == 0:
        return np.nan                     # clean part: loss undefined, excluded
    pb = np.asarray(record.get("pred_boxes", []), dtype=float).reshape(-1, 4)
    ps = np.asarray(record.get("pred_scores", []), dtype=float).reshape(-1)
    if pb.shape[0] == 0:
        return NEG_INF

    hit = box_iou_matrix(gt, pb) >= iou_thresh          # (n_gt, n_pred)
    # An image escapes unless EVERY ground-truth box is matched, so the image's
    # confidence is set by its weakest-covered defect.
    per_gt = []
    for g in range(hit.shape[0]):
        s = ps[hit[g]]
        per_gt.append(s.max() if s.size else NEG_INF)
    return float(min(per_gt))


def escape_confidences(records, iou_thresh: float = 0.5) -> np.ndarray:
    """Scores for the defective images only; clean images are dropped."""
    c = np.array([escape_confidence(r, iou_thresh) for r in records], dtype=float)
    return c[~np.isnan(c)]


# ---------------------------------------------------------------------------
# Baseline: plain CRC as a quantile of the real scores
# ---------------------------------------------------------------------------

def crc_threshold(c_real: np.ndarray, alpha: float) -> float:
    """Largest lambda whose certified escape risk is <= alpha.

    Equivalent to `conformal_risk_control` on the 0/1 image-level escape loss,
    written directly as a quantile because the reduction above makes it one.
    """
    c = np.sort(np.asarray(c_real, dtype=float))
    n = c.size
    if n == 0:
        return NEG_INF
    # certified risk at a threshold just above c[k-1] is (n/(n+1))*(k/n) + 1/(n+1)
    k = np.arange(n + 1)
    certified = k / (n + 1.0) + 1.0 / (n + 1.0)
    ok = np.flatnonzero(certified <= alpha)
    if ok.size == 0:
        return NEG_INF                      # alpha unreachable: flag everything
    kk = int(ok[-1])
    return float(c[kk - 1]) if kk >= 1 else NEG_INF


# ---------------------------------------------------------------------------
# SPI
# ---------------------------------------------------------------------------

def _rank_windows(m: int, N: int, beta: float):
    """Deterministic window [lo_r, hi_r] of synthetic indices for each real rank.

    Depends only on (m, N, beta) - never on the observed scores. That is the
    property the validity argument rests on: a data-dependent window would let
    the method quietly buy itself a larger effective sample.

    The centre is the synthetic index matching real rank r under a uniform
    rank map; the half-width is a binomial fluctuation bound at level beta.
    """
    r = np.arange(1, m + 1)
    centre = r * (N + 1.0) / (m + 1.0)
    # sd of the r-th order statistic's position under rank exchangeability
    p = r / (m + 1.0)
    sd = np.sqrt(N * p * (1 - p)) + 1.0
    from scipy.stats import norm
    w = norm.ppf(1 - beta / 2.0) * sd
    lo = np.clip(np.floor(centre - w).astype(int), 1, N)
    hi = np.clip(np.ceil(centre + w).astype(int), 1, N)
    return lo, hi


def spi_threshold(c_real: np.ndarray, c_syn: np.ndarray, alpha: float,
                  beta: float = 0.1) -> float:
    """Escape threshold calibrated from few real scores plus many synthetic ones.

    Real scores enter through ranks only; the returned threshold is a synthetic
    order statistic, clipped into the rank-coupled window so that a badly
    misaligned synthetic set cannot push the threshold past what the real ranks
    support.
    """
    c_real = np.sort(np.asarray(c_real, dtype=float))
    c_syn = np.sort(np.asarray(c_syn, dtype=float))
    m, N = c_real.size, c_syn.size
    if m == 0:
        return NEG_INF
    if N == 0:
        return crc_threshold(c_real, alpha)

    # TRANSPORT: locate each real score inside the synthetic order statistics.
    # This is the step that corrects misalignment. Clipping on RANKS alone does
    # not: a uniformly optimistic generator leaves every rank window in place
    # while shifting all the values, and the threshold sails through unchanged
    # (measured: escape 0.51 against alpha=0.10). Mapping real VALUES into
    # synthetic rank space means an optimistic generator pushes the real scores
    # to low synthetic ranks, which drags the threshold back down by exactly the
    # amount the generator was wrong by.
    transported = np.searchsorted(c_syn, c_real, side="left")   # in 0..N
    transported = np.clip(transported, 1, N)
    transported.sort()

    # Same CRC accounting as before - the budget is still paid for out of the m
    # REAL points, so no effective sample size is manufactured. Only the
    # resolution of the returned value comes from the synthetic grid.
    k = np.arange(m + 1)
    certified = k / (m + 1.0) + 1.0 / (m + 1.0)
    ok = np.flatnonzero(certified <= alpha)
    if ok.size == 0:
        return NEG_INF
    kk = int(ok[-1])
    if kk < 1:
        return NEG_INF

    idx = int(transported[kk - 1])
    return float(c_syn[idx - 1])


def empirical_risk(c_test: np.ndarray, threshold: float) -> float:
    """Realised escape rate on held-out defective images at this threshold."""
    c = np.asarray(c_test, dtype=float)
    return float((c < threshold).mean()) if c.size else 0.0
