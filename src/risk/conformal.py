"""
conformal.py
============
Distribution-free risk control for defect inspection.

This module is the mathematical core of the method. It converts a detector's
uncalibrated confidence scores into operating points that carry *finite-sample,
distribution-free* guarantees on the quantity a QA line actually cares about:
the **defect escape rate** (fraction of true defects that are not flagged).

Three layers, in increasing order of what they buy you:

1. `conformal_risk_control` - Angelopoulos, Bates, Fisch, Lei & Schuster,
   "Conformal Risk Control" (ICLR 2024). Picks a single threshold lambda_hat
   such that the expected loss on a fresh exchangeable image is <= alpha.
   Requires the per-image loss to be monotone in lambda; ours is.

2. `learn_then_test` - Angelopoulos, Bates, Candes, Jordan & Lei, "Learn then
   Test: Calibrating Predictive Algorithms to Achieve Risk Control" (2021).
   Searches a *multi-dimensional* configuration grid (accept threshold x review
   threshold x faithfulness gate) and returns only the configurations whose
   risk is statistically certified, with family-wise error control. This is
   what makes the two-threshold triage policy valid - plain CRC only handles a
   one-dimensional monotone family.

3. `mondrian_risk_control` - per-group (e.g. per defect class) risk control, so
   a safety-critical class can be held to a tighter escape bound than a
   cosmetic one. The guarantee holds conditionally within each group.

Notation follows the papers: n calibration points, loss L_i(lambda) in [0, B],
target risk level alpha.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


class RiskNotAchievable(RuntimeError):
    """Raised when no configuration in the grid can certify the target risk."""


# ---------------------------------------------------------------------------
# 1. Conformal Risk Control (monotone, one-dimensional)
# ---------------------------------------------------------------------------

def conformal_risk_control(
    losses: np.ndarray,
    lambdas: np.ndarray,
    alpha: float,
    B: float = 1.0,
) -> float:
    """Smallest lambda whose certified risk is <= alpha.

    Parameters
    ----------
    losses : (n, m) array
        losses[i, j] = loss of calibration example i at lambdas[j]. Must be
        non-increasing along j. This is checked, not assumed.
    lambdas : (m,) array of score thresholds in DESCENDING order.
        Descending is not a style choice: escape loss *rises* with the score
        threshold (a higher bar rejects more true defects), so the grid has to
        run high -> low for the loss to be non-increasing along j, which is what
        the CRC theorem requires. Use `escape_threshold_grid()` to get this
        right. Feeding an ascending grid raises rather than silently returning
        an invalid guarantee.
    alpha : target risk level in (0, 1).
    B : upper bound on the loss (1.0 for a fraction-missed loss).

    Guarantee
    ---------
    E[L_{n+1}(lambda_hat)] <= alpha over the draw of the calibration set and
    the test point, requiring only exchangeability - no distributional
    assumptions, no asymptotics, valid at any n.

    The estimator is the CRC one:

        R_hat_n(lambda) = (1/n) sum_i L_i(lambda)
        lambda_hat = inf{ lambda : (n/(n+1)) R_hat_n(lambda) + B/(n+1) <= alpha }

    The B/(n+1) term is the finite-sample correction that makes the bound hold
    for the *next* point rather than only in expectation over calibration.
    """
    losses = np.asarray(losses, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if losses.ndim != 2:
        raise ValueError(f"losses must be 2-D (n, m), got shape {losses.shape}")
    n, m = losses.shape
    if m != lambdas.shape[0]:
        raise ValueError(f"losses has {m} columns but {lambdas.shape[0]} lambdas")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    mean_loss = losses.mean(axis=0)
    # Monotonicity is a precondition of the CRC theorem; violating it silently
    # would invalidate the guarantee, so fail loudly instead.
    increases = np.diff(mean_loss) > 1e-9
    if increases.any():
        bad = int(np.argmax(increases))
        raise ValueError(
            "CRC requires the empirical risk to be non-increasing in lambda; it "
            f"increases between index {bad} (R={mean_loss[bad]:.4f}) and "
            f"{bad + 1} (R={mean_loss[bad + 1]:.4f}). Check the loss definition "
            "or the ordering of `lambdas`."
        )

    rhs = (n / (n + 1.0)) * mean_loss + B / (n + 1.0)
    feasible = np.flatnonzero(rhs <= alpha)
    if feasible.size == 0:
        raise RiskNotAchievable(
            f"no lambda achieves alpha={alpha}: min certified risk is "
            f"{rhs.min():.4f} at lambda={lambdas[int(np.argmin(rhs))]:.4f}. "
            "Loosen alpha, improve the detector, or enlarge the calibration set "
            f"(n={n}, finite-sample penalty B/(n+1)={B / (n + 1):.4f})."
        )
    return float(lambdas[feasible[0]])


# ---------------------------------------------------------------------------
# 2. Learn then Test (multi-dimensional, non-monotone configuration search)
# ---------------------------------------------------------------------------

def _h1(a: np.ndarray, b: float) -> np.ndarray:
    """KL divergence between Bernoulli(a) and Bernoulli(b)."""
    a = np.clip(a, 1e-12, 1 - 1e-12)
    b = float(np.clip(b, 1e-12, 1 - 1e-12))
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hoeffding_bentkus_pvalue(r_hat: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Valid p-value for H0: R(lambda) > alpha, for losses bounded in [0, 1].

    The Hoeffding-Bentkus p-value is the minimum of two concentration bounds
    (Bates et al. 2021): Hoeffding is tight in the bulk, Bentkus/binomial in the
    tail. The minimum of two valid p-values is itself valid.
    """
    from scipy.stats import binom

    r_hat = np.asarray(r_hat, dtype=float)
    hoeffding = np.exp(-n * _h1(np.minimum(r_hat, alpha), alpha))
    bentkus = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)
    return np.minimum(1.0, np.minimum(hoeffding, bentkus))


def learn_then_test(
    losses: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    correction: str = "bonferroni",
) -> np.ndarray:
    """Indices of configurations certified to have risk <= alpha.

    Unlike CRC this makes no monotonicity assumption, so the configuration grid
    may be multi-dimensional and arbitrarily ordered - exactly what the
    two-threshold triage policy needs.

    Parameters
    ----------
    losses : (n, m) per-example loss for each of m configurations.
    alpha : target risk.
    delta : allowed probability of any false certification (family-wise).
    correction : "bonferroni" (valid, conservative) or "fixed_sequence"
        (more powerful, valid only for a pre-specified order).
    """
    losses = np.asarray(losses, dtype=float)
    n, m = losses.shape
    r_hat = losses.mean(axis=0)
    p = hoeffding_bentkus_pvalue(r_hat, n, alpha)

    if correction == "bonferroni":
        return np.flatnonzero(p <= delta / m)

    if correction == "fixed_sequence":
        order = np.argsort(r_hat)
        keep = []
        for j in order:
            if p[j] <= delta:
                keep.append(int(j))
            else:
                break
        return np.array(sorted(keep), dtype=int)

    raise ValueError(f"unknown correction {correction!r}")


# ---------------------------------------------------------------------------
# 3. Mondrian (group-conditional) risk control
# ---------------------------------------------------------------------------

def mondrian_risk_control(
    losses: np.ndarray,
    lambdas: np.ndarray,
    groups: np.ndarray,
    alpha_by_group: dict,
    B: float = 1.0,
) -> dict:
    """Per-group CRC: one calibrated threshold per group, each with its own alpha.

    Lets a safety-critical defect class be held to a tighter escape bound than a
    cosmetic one. Exchangeability now only needs to hold *within* each group,
    which is a weaker and more realistic assumption on a production line than
    global exchangeability across defect types.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    out = {}
    for g, alpha_g in alpha_by_group.items():
        mask = groups == g
        n_g = int(mask.sum())
        if n_g == 0:
            out[g] = {"lambda": None, "n": 0, "error": "no calibration data"}
            continue
        try:
            lam = conformal_risk_control(losses[mask], lambdas, alpha_g, B=B)
            # exact lookup: lambdas is descending, so searchsorted would be wrong
            j = int(np.flatnonzero(lambdas == lam)[0])
            out[g] = {
                "lambda": lam,
                "n": n_g,
                "alpha": alpha_g,
                "empirical_risk": float(losses[mask][:, j].mean()),
            }
        except RiskNotAchievable as e:
            out[g] = {"lambda": None, "n": n_g, "alpha": alpha_g, "error": str(e)}
    return out


# ---------------------------------------------------------------------------
# Loss definitions for detection
# ---------------------------------------------------------------------------

def escape_threshold_grid(n: int = 201, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Descending score-threshold grid, the ordering CRC needs for escape loss.

    Descending because escape loss increases with the threshold; see the
    `lambdas` note in `conformal_risk_control`.
    """
    return np.linspace(hi, lo, n)


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4), both xyxy. Returns (N, M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def escape_loss_curve(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    lambdas: np.ndarray,
    iou_thresh: float = 0.5,
) -> np.ndarray:
    """Per-image defect-escape loss as a function of the score threshold.

        L(lambda) = #{GT defects with no prediction of score >= lambda
                      at IoU >= iou_thresh} / #{GT defects}

    Bounded in [0, 1] and non-increasing as lambda decreases, as CRC requires.
    Defect-free images contribute 0 - they cannot produce an escape, and
    including them would dilute the risk estimate, so `build_calibration_losses`
    filters them out by default.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    m = lambdas.shape[0]
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    if gt_boxes.shape[0] == 0:
        return np.zeros(m, dtype=float)
    pred_boxes = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
    if pred_boxes.shape[0] == 0:
        return np.ones(m, dtype=float)

    hit = box_iou_matrix(gt_boxes, pred_boxes) >= iou_thresh   # (n_gt, n_pred)
    scores = np.asarray(pred_scores, dtype=float).reshape(-1)

    out = np.empty(m, dtype=float)
    for j, lam in enumerate(lambdas):
        kept = scores >= lam
        if not kept.any():
            out[j] = 1.0
        else:
            out[j] = 1.0 - hit[:, kept].any(axis=1).mean()
    return out


def build_calibration_losses(
    records: Sequence[dict],
    lambdas: np.ndarray,
    iou_thresh: float = 0.5,
    drop_defect_free: bool = True,
):
    """Assemble the (n, m) calibration loss matrix from per-image records.

    Each record needs `gt_boxes` (N,4), `pred_boxes` (M,4), `pred_scores` (M,),
    and optionally `gt_labels` for Mondrian grouping.

    Returns (losses, groups, kept_indices).
    """
    lambdas = np.asarray(lambdas, dtype=float)
    losses, groups, keep = [], [], []
    for i, r in enumerate(records):
        gt = np.asarray(r["gt_boxes"], dtype=float).reshape(-1, 4)
        if drop_defect_free and gt.shape[0] == 0:
            continue
        losses.append(
            escape_loss_curve(gt, r["pred_boxes"], r["pred_scores"], lambdas, iou_thresh)
        )
        labels = r.get("gt_labels")
        groups.append(int(labels[0]) if labels is not None and len(labels) else -1)
        keep.append(i)
    if not losses:
        raise ValueError("no usable calibration images (all defect-free?)")
    return np.stack(losses), np.asarray(groups), np.asarray(keep)
