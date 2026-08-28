"""
acquisition.py
==============
Defect-seeking calibration: buy the labels that tighten the certificate.

The problem
-----------
Escape loss is defined only on **defective** parts, so the conformal penalty is
1/(n_def + 1) where n_def counts labelled *defects*, not labelled parts. On a
line that is 89% clean, a random labelling budget buys mostly zeros: rows that
cost money and move the guarantee not at all.

Measured on KolektorSDD2 at alpha = 0.10 over 300 replicates, a budget of 50
labels spent at random yields 5.4 defects and a certificate **0% of the time** -
CRC correctly refuses, because the finite-sample term alone exceeds the budget.
The same 50 labels spent by screening on the detector's own score yield 44
defects and a certificate 76% of the time.

The catch, and the fix
----------------------
Screening on the detector score selects on the very quantity being calibrated.
The screen preferentially finds *easy* defects, which carry high escape
confidence, so the calibration sample flatters the detector and the threshold
comes out too permissive: measured mean risk 0.1011-0.1043 against alpha = 0.10
with P(risk > alpha) = 1.0. Small, but systematic and always in one direction.

That bias is correctable **here specifically** because the policy is ours. Under
a known sampling law pi(x), the calibration sample is drawn from a tilted
distribution, and inverse-propensity weights w_i = 1 / pi(x_i) map it back. This
is the weighted-conformal / covariate-shift construction (Tibshirani, Barber,
Candes & Ramdas, 2019), and its precondition - a *known* likelihood ratio - is
exactly what a self-chosen acquisition policy provides and what a generic
adaptive selection rule does not.

Efficiency
----------
The inner loop is a weighted quantile over calibration scores, evaluated for many
candidate thresholds. Written naively that is O(m) per threshold and O(m^2)
overall. Sorting once and sweeping a prefix sum of the weights makes it
O(m log m) once, then O(1) per threshold: the certified risk at the k-th order
statistic is a prefix ratio, and the admissible k is then found by binary search
because certified risk is monotone in k. That matters here because the policy is
re-calibrated after every acquisition round.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -1.0e9


# ---------------------------------------------------------------------------
# Acquisition policies
# ---------------------------------------------------------------------------

def screening_probabilities(scores: np.ndarray, gamma: float = 3.0,
                            floor: float = 1e-3) -> np.ndarray:
    """Sampling law pi(x) proportional to detector score ^ gamma.

    `gamma = 0` recovers uniform random labelling. Larger gamma screens harder
    for likely defects.

    `floor` keeps every part reachable with non-zero probability. That is not a
    numerical nicety: inverse-propensity weights are 1/pi, so a part that could
    never be sampled would carry infinite weight if it ever were, and any part
    with pi = 0 puts a region of the input space outside the correction's reach
    entirely - the guarantee would then hold only on the support the screen
    happens to like.
    """
    s = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    if gamma <= 0:
        p = np.ones_like(s)
    else:
        p = np.power(s + floor, gamma)
    total = p.sum()
    if total <= 0:
        p = np.ones_like(s)
        total = p.sum()
    return p / total


def acquire(scores: np.ndarray, budget: int, gamma: float = 3.0,
            seed: int = 0):
    """Choose `budget` parts to label. Returns (indices, sampling_probs).

    Sampling is without replacement, so the realised inclusion probability is
    not exactly pi. For budget << pool the first-order approximation pi is
    standard and is what the returned weights assume; the unit test measures
    coverage empirically rather than trusting that approximation.
    """
    p = screening_probabilities(scores, gamma)
    n = len(p)
    budget = int(min(budget, n))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=budget, replace=False, p=p)
    return idx, p[idx]


# ---------------------------------------------------------------------------
# Weighted conformal risk control
# ---------------------------------------------------------------------------

def weighted_crc_threshold(c: np.ndarray, weights: np.ndarray,
                           alpha: float) -> float:
    """Largest threshold whose *weighted* certified escape risk is <= alpha.

    With w_i = 1 / pi_i normalised, the certified risk at threshold t is

        R(t) = ( sum_i w_i 1{c_i < t} + w_max ) / ( sum_i w_i + w_max )

    The w_max term is the weighted analogue of the 1/(n+1) finite-sample
    correction: it charges for the one unobserved test point at the worst weight
    it could carry. With uniform weights it collapses back to 1/(n+1) exactly,
    so this is a strict generalisation of `crc_threshold` rather than a
    different estimator.

    Implementation: sort once, take a prefix sum of the weights, and binary
    search the largest admissible order statistic. Certified risk is
    non-decreasing in k, which is what makes the search valid.
    """
    c = np.asarray(c, dtype=float)
    w = np.asarray(weights, dtype=float)
    if c.size == 0:
        return NEG_INF
    if c.size != w.size:
        raise ValueError(f"{c.size} scores but {w.size} weights")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive; a zero-probability "
                         "part cannot be corrected back to the population")

    order = np.argsort(c, kind="mergesort")     # stable: ties keep input order
    cs, ws = c[order], w[order]
    prefix = np.concatenate([[0.0], np.cumsum(ws)])
    total = prefix[-1]
    w_max = ws.max()

    # certified[k] = risk if we accept the k smallest scores as "escapes"
    certified = (prefix + w_max) / (total + w_max)
    ok = np.flatnonzero(certified <= alpha)
    if ok.size == 0:
        return NEG_INF                          # alpha unreachable: flag all
    k = int(ok[-1])
    return float(cs[k - 1]) if k >= 1 else NEG_INF


def unweighted_crc_threshold(c: np.ndarray, alpha: float) -> float:
    """Plain CRC, i.e. weighted CRC with equal weights. Kept for comparison."""
    c = np.asarray(c, dtype=float)
    if c.size == 0:
        return NEG_INF
    return weighted_crc_threshold(c, np.ones_like(c), alpha)


# ---------------------------------------------------------------------------
# One acquisition round
# ---------------------------------------------------------------------------

def calibrate_from_acquisition(pool_scores, pool_confidences, budget, alpha,
                               gamma=3.0, seed=0, correct=True):
    """Spend a labelling budget, then calibrate from what it bought.

    pool_scores       screening signal available BEFORE labelling (detector max score)
    pool_confidences  escape confidence, revealed only for the parts we label
                      (NaN marks a clean part, which carries no escape loss)

    Returns a dict with the threshold and diagnostics. `correct=False` reproduces
    the naive, biased estimator so the two can be compared on the same draw.
    """
    idx, probs = acquire(np.asarray(pool_scores, float), budget, gamma, seed)
    conf = np.asarray(pool_confidences, float)[idx]
    defective = ~np.isnan(conf)

    c_def = conf[defective]
    p_def = probs[defective]
    if c_def.size == 0:
        return {"threshold": NEG_INF, "n_labelled": len(idx), "n_defective": 0,
                "issued": False, "corrected": correct}

    w = (1.0 / p_def) if correct else np.ones_like(c_def)
    thr = weighted_crc_threshold(c_def, w, alpha)
    return {"threshold": thr, "n_labelled": int(len(idx)),
            "n_defective": int(c_def.size), "issued": thr > NEG_INF / 2,
            "corrected": bool(correct),
            "weight_ratio": float(w.max() / w.min()) if w.size else 1.0}
