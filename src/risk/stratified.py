"""
stratified.py
=============
Stratified defect-seeking calibration - bounded weights by construction.

Why this exists
---------------
Defect-seeking acquisition works: on KolektorSDD2 a budget of 50 labels spent by
screening on the detector score buys 44 defects instead of 5, and turns "no
certificate at any budget" into "a certificate 80% of the time". It is also
biased, because screening on the detector score selects on the very quantity
being calibrated: the screen finds *easy* defects, which flatter the detector
(measured escape 0.1011-0.1043 against alpha = 0.10).

The textbook correction - inverse-propensity weighting - was implemented in
`acquisition.py` and **fails here**. Weighted CRC must charge the unobserved test
point at the worst weight it could carry:

    R(t) = ( sum_i w_i 1{c_i < t} + w_max ) / ( sum_i w_i + w_max )

Under a gamma = 3 screen the ratio w_max / w_min is enormous, so w_max alone
dominates both sums and no useful threshold is ever certified: measured 0%
issuance at every gamma and every clipping level. The screen buys 5x the defects
and the correction hands all of it back.

The fix
-------
The problem is not weighting; it is *unbounded* weights. Under proportional
sampling the propensity of the rarest part can be orders of magnitude below the
most likely one, and w = 1/pi inherits that spread.

Stratified sampling removes the spread by construction. Partition the pool into K
strata by the screening score, then draw a **fixed, chosen** number n_k from
stratum k. The design weight of a part in stratum k is

    w_k = N_k / n_k

which depends only on counts the sampler chose, not on any observed score. The
weight ratio is therefore bounded by the design - typically single digits - and
the w_max term stops dominating. The strata boundaries and the allocation are
both fixed before any label is revealed, so nothing here is data-dependent in a
way that could void exchangeability *within* a stratum.

This also makes the estimator interpretable to a plant: "we sampled 20 parts from
the top-scoring decile and 5 from each lower decile" is an auditable sampling
plan, which "we sampled proportional to score^3" is not.
"""

from __future__ import annotations

import numpy as np

NEG_INF = -1.0e9


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

def make_strata(scores: np.ndarray, k: int = 5) -> np.ndarray:
    """Assign each part to one of k strata by score quantile.

    Quantile edges rather than fixed score cuts: score distributions differ
    wildly between datasets (KolektorSDD2 piles almost everything near zero),
    and fixed cuts would leave strata empty on some datasets and merged on
    others.
    """
    s = np.asarray(scores, dtype=float)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(edges, s, side="right")


def allocate(strata: np.ndarray, budget: int, k: int = 5,
             tilt: float = 2.0, min_per_stratum: int = 3) -> np.ndarray:
    """How many labels to spend in each stratum.

    `tilt` biases the allocation toward high-score strata, which is where the
    defects are; `tilt = 1` is proportional allocation (no defect seeking).
    `min_per_stratum` keeps every stratum represented, which is what bounds the
    weight ratio: a stratum sampled zero times would have infinite design weight
    and put part of the population outside the estimator entirely.
    """
    sizes = np.array([(strata == j).sum() for j in range(k)], dtype=float)
    pref = sizes * (np.arange(1, k + 1) ** tilt)
    pref = np.where(sizes > 0, pref, 0.0)
    if pref.sum() <= 0:
        pref = np.where(sizes > 0, 1.0, 0.0)
    n = np.floor(budget * pref / pref.sum()).astype(int)
    n = np.minimum(n, sizes.astype(int))
    # every non-empty stratum gets a floor, then spend any remainder greedily
    n = np.where((sizes > 0) & (n < min_per_stratum),
                 np.minimum(min_per_stratum, sizes.astype(int)), n)
    while n.sum() > budget:
        j = int(np.argmax(np.where(n > min_per_stratum, n, -1)))
        if n[j] <= min_per_stratum:
            break
        n[j] -= 1
    while n.sum() < budget:
        room = sizes.astype(int) - n
        if room.max() <= 0:
            break
        j = int(np.argmax(room * (np.arange(1, k + 1) ** tilt)))
        n[j] += 1
    return n


def stratified_acquire(scores, budget, k=5, tilt=2.0, seed=0):
    """Draw a stratified sample. Returns (indices, design_weights).

    Design weight N_k / n_k is fixed by the sampling plan, so it is known before
    any label is revealed and its spread is bounded by the allocation rather than
    by the score distribution.
    """
    s = np.asarray(scores, dtype=float)
    strata = make_strata(s, k)
    n_k = allocate(strata, budget, k, tilt)
    rng = np.random.default_rng(seed)

    idx, w = [], []
    for j in range(k):
        pool = np.flatnonzero(strata == j)
        if pool.size == 0 or n_k[j] == 0:
            continue
        take = rng.choice(pool, size=int(min(n_k[j], pool.size)), replace=False)
        idx.append(take)
        w.append(np.full(take.size, pool.size / float(take.size)))
    if not idx:
        return np.array([], dtype=int), np.array([])
    return np.concatenate(idx), np.concatenate(w)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def stratified_crc_threshold(c: np.ndarray, w: np.ndarray, alpha: float) -> float:
    """Weighted CRC with design weights.

    Identical estimator to `acquisition.weighted_crc_threshold`; kept separate so
    the two can be compared side by side in the notebook without either being
    silently swapped for the other. Sort once, prefix-sum the weights, binary
    search the largest admissible order statistic - O(m log m) then O(1) per
    candidate threshold.
    """
    c = np.asarray(c, dtype=float)
    w = np.asarray(w, dtype=float)
    if c.size == 0:
        return NEG_INF
    order = np.argsort(c, kind="mergesort")
    cs, ws = c[order], w[order]
    prefix = np.concatenate([[0.0], np.cumsum(ws)])
    certified = (prefix + ws.max()) / (prefix[-1] + ws.max())
    ok = np.flatnonzero(certified <= alpha)
    if ok.size == 0:
        return NEG_INF
    kk = int(ok[-1])
    return float(cs[kk - 1]) if kk >= 1 else NEG_INF


def weight_ratio(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.max() / w.min()) if w.size else 1.0
