"""
spi_exact.py
============
The exact Synthetic-Powered Predictive Inference (SPI) machinery of

    Bashari, Lotan, Lee, Dobriban & Romano,
    "Synthetic-Powered Predictive Inference", NeurIPS 2025 (arXiv:2505.13432),
    reference code https://github.com/Meshiba/spi

re-implemented from the paper's equations (checked against the paper's HTML on
2026-09-23 and again on 2026-09-27). Nothing here is specific to defects: it takes
real and synthetic NONCONFORMITY scores ``s`` (large = hard) and returns SPI's
prediction rule, its bounds and its beta selection. ``src/risk/sperc.py`` applies
it to detector escape scores (``s = -c``).

Notation (paper, Section 3)
---------------------------
m real calibration scores S_1..S_m, N synthetic scores S~_1..S~_N, sorted
S~_(1) <= ... <= S~_(N), and S~_(N+1) = +inf.

* Rank pmf of the r-th real order statistic among the synthetic ones, r in [m+1]:

      p_{m,N,r}(k) = C(k+r-2, r-1) C(N+m-k-r+2, m-r+1) / C(N+m+1, m+1),  k = 1..N+1
      F(t) = sum_{k<=t} p_{m,N,r}(k)

* Rank windows at tolerance beta in (0, 1):

      R_r^- = max{t in [N+1] : F(t-1) <= beta/2},  R_r^+ = min{t in [N+1] : F(t) >= 1-beta/2}
      I_m(r) = [L_m(r), U_m(r)] = [S~_(R_r^-), S~_(R_r^+)]

* Score transporter, with r_eta = 1 + #{i : S_i < eta} (rank of eta among real + eta):

      T(eta) = U_m(r)              if eta >= U_m(r)
             = NN^-_m(r, eta)      if L_m(r) <= eta < U_m(r)
             = L_m(r)              if eta <  L_m(r)
      NN^-_m(r, eta) = max_{R_r^- <= j <= R_r^+} {S~_(j) : S~_(j) <= eta}

* Prediction set C = {y : T(s(x, y)) <= Q~},  Q~ = S~_(ceil((1-alpha)(N+1))).

Guarantees (coverage = P(Y_{m+1} in C))
--------------------------------------
* Theorem 3.3:  1-alpha-beta-eps <= coverage <= 1-alpha+beta+eps+1/(N+1),
  eps = (1/(m+1)) sum_i d_TV(P_(i)^{m+1}, Q_(i)^{m+1}), the mean total-variation
  distance between the laws of the i-th order statistics of m+1 real and m+1
  synthetic scores. eps is NOT observable.
* Theorem 3.5 (any synthetic data):
      #{j : R_j^+ <= ceil((1-alpha)(N+1))}/(m+1) <= coverage
                                   <= #{j : R_j^- <= ceil((1-alpha)(N+1))}/(m+1)
* Corollary 3.6: Theorem 3.5 still holds when the synthetic scores are built from
  the real calibration data themselves.
* Algorithm 4: smallest beta on a step-``eps`` grid whose Theorem 3.5 lower bound
  reaches a desired coverage floor L.

Ties: the theory assumes continuous scores (or ties broken at random). Callers
with discrete scores should break ties first (``src/risk/escape.break_ties``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln

NEG_INF = float("-inf")
POS_INF = float("inf")


# ------------------------------------------------------------------ rank law
def _log_comb(n, k):
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def rank_pmf(m: int, N: int, r: int) -> np.ndarray:
    """p_{m,N,r}(k) for k = 1..N+1 (array of length N+1). Sums to 1."""
    if m < 1 or N < 1:
        raise ValueError(f"need m >= 1 and N >= 1, got m={m}, N={N}")
    if not (1 <= r <= m + 1):
        raise ValueError(f"r must be in 1..m+1, got r={r}, m={m}")
    k = np.arange(1, N + 2, dtype=float)
    lp = (_log_comb(k + r - 2, r - 1) + _log_comb(N + m - k - r + 2, m - r + 1)
          - _log_comb(N + m + 1, m + 1))
    return np.exp(lp)


@lru_cache(maxsize=8192)
def rank_windows(m: int, N: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """(R^-, R^+) for r = 1..m+1: 1-indexed synthetic order-statistic indices.

    Depends only on (m, N, beta), never on score values. Arrays are read-only
    (they are cached).
    """
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    lo = np.empty(m + 1, dtype=int)
    hi = np.empty(m + 1, dtype=int)
    t = np.arange(1, N + 2)
    for r in range(1, m + 2):
        F = np.concatenate([[0.0], np.cumsum(rank_pmf(m, N, r))])   # F[t], t = 0..N+1
        lo[r - 1] = t[F[t - 1] <= beta / 2 + 1e-12].max()
        hi[r - 1] = t[F[t] >= 1 - beta / 2 - 1e-12].min()
    lo.setflags(write=False)
    hi.setflags(write=False)
    return lo, hi


def synthetic_quantile_index(N: int, alpha: float) -> int:
    """ceil((1-alpha)(N+1)), the 1-indexed position of Q~ (may exceed N)."""
    return int(np.ceil((1.0 - alpha) * (N + 1) - 1e-9))


# ------------------------------------------------------------------ bounds
def coverage_lower_worst_case(m: int, N: int, alpha: float, beta: float) -> float:
    """SPI Theorem 3.5 lower bound on coverage, valid for ANY synthetic data."""
    _, hi = rank_windows(m, N, beta)
    return float(np.sum(hi <= synthetic_quantile_index(N, alpha)) / (m + 1))


def coverage_upper_worst_case(m: int, N: int, alpha: float, beta: float) -> float:
    """SPI Theorem 3.5 upper bound on coverage, valid for ANY synthetic data."""
    lo, _ = rank_windows(m, N, beta)
    return float(np.sum(lo <= synthetic_quantile_index(N, alpha)) / (m + 1))


def tier_h_cap(m: int, N: int, alpha: float, beta: float) -> float:
    """Worst-case ESCAPE (non-coverage) upper bound = 1 - Theorem 3.5 lower bound."""
    return 1.0 - coverage_lower_worst_case(m, N, alpha, beta)


def tier_h_floor(m: int, N: int, alpha: float, beta: float) -> float:
    """Worst-case ESCAPE lower bound = 1 - Theorem 3.5 upper bound."""
    return 1.0 - coverage_upper_worst_case(m, N, alpha, beta)


def tier_n_interval(alpha: float, beta: float, N: int, eps: float = 0.0) -> Tuple[float, float]:
    """Theorem 3.3 in escape units: [alpha-beta-eps-1/(N+1), alpha+beta+eps]."""
    return float(alpha - beta - eps - 1.0 / (N + 1)), float(alpha + beta + eps)


def tier_n_bound(alpha: float, beta: float) -> float:
    """Upper end of Tier N without the unobservable eps: alpha + beta."""
    return float(alpha + beta)


# ------------------------------------------------------------------ beta selection
def select_beta_alg4(m: int, N: int, alpha: float, coverage_floor: float,
                     step: float = 0.005, beta_max: float = 0.995) -> Optional[float]:
    """SPI Algorithm 4: beta <- step; while the Thm 3.5 lower bound < L, beta += step.

    Returns the first beta (on the step grid) whose worst-case coverage floor is at
    least ``coverage_floor``, or None if none up to ``beta_max`` reaches it. Uses
    only (m, N, alpha), never data, so selecting beta costs no validity.
    """
    if not 0 < step < 1:
        raise ValueError("step must be in (0, 1)")
    k = 1
    while True:
        beta = round(k * step, 10)
        if beta > beta_max:
            return None
        if coverage_lower_worst_case(m, N, alpha, beta) >= coverage_floor - 1e-12:
            return float(beta)
        k += 1


def select_beta_grid(m: int, N: int, alpha: float, coverage_floor: float,
                     grid: Sequence[float]) -> Optional[float]:
    """Smallest beta in ``grid`` whose worst-case coverage floor reaches the target."""
    for b in sorted(grid):
        if coverage_lower_worst_case(m, N, alpha, b) >= coverage_floor - 1e-12:
            return float(b)
    return None


# ------------------------------------------------------------------ transporter
def transport(eta: float, s_real: np.ndarray, s_syn: np.ndarray, beta: float) -> float:
    """T(eta) exactly as in the paper (s_real, s_syn need not be sorted)."""
    s_real = np.sort(np.asarray(s_real, dtype=float))
    s_syn = np.sort(np.asarray(s_syn, dtype=float))
    m, N = s_real.size, s_syn.size
    lo, hi = rank_windows(m, N, beta)
    ext = np.concatenate([s_syn, [POS_INF]])
    r = 1 + int(np.searchsorted(s_real, eta, side="left"))          # 1 + #{S_i < eta}
    L, U = ext[lo[r - 1] - 1], ext[hi[r - 1] - 1]
    if eta >= U:
        return float(U)
    if eta < L:
        return float(L)
    j = int(np.searchsorted(ext, eta, side="right"))                # #{S~_(j) <= eta}
    j = min(max(j, lo[r - 1]), hi[r - 1])
    return float(ext[j - 1])


def spi_covers(eta, s_real: np.ndarray, s_syn: np.ndarray, alpha: float, beta: float):
    """Whether SPI's set contains each test score (bool or bool array)."""
    s_syn_sorted = np.sort(np.asarray(s_syn, dtype=float))
    N = s_syn_sorted.size
    Q = synthetic_quantile_index(N, alpha)
    q = POS_INF if Q > N else s_syn_sorted[Q - 1]
    etas = np.atleast_1d(np.asarray(eta, dtype=float))
    out = np.array([transport(e, s_real, s_syn_sorted, beta) <= q for e in etas])
    return bool(out[0]) if np.ndim(eta) == 0 else out


def spi_score_threshold(s_real: np.ndarray, s_syn: np.ndarray, alpha: float,
                        beta: float) -> float:
    """t = sup{eta : T(eta) <= Q~}: the one-sided set {eta <= t} contains SPI's set.

    T is non-decreasing (windows are non-decreasing in r), so SPI's covered set is
    a lower interval; the only possible difference from {eta <= t} is the single
    boundary point eta = t, which the threshold rule covers and SPI may not. So the
    threshold rule covers at least what SPI covers: coverage lower bounds (Tier H,
    Tier N lower) carry over exactly. Returns +inf if Q~ does not exist (Q > N):
    SPI then covers every score.

    Within the rank-r interval (s_(r-1), s_(r)]:
        U(r) <= Q~        whole interval covered;
        L(r) >  Q~        none of it;
        otherwise         covered iff eta < S~_(Q'), Q' the first window index
                          whose synthetic score exceeds Q~.
    """
    s_real = np.sort(np.asarray(s_real, dtype=float))
    s_syn = np.sort(np.asarray(s_syn, dtype=float))
    m, N = s_real.size, s_syn.size
    if m == 0 or N == 0:
        raise ValueError("need at least one real and one synthetic score")
    Q = synthetic_quantile_index(N, alpha)
    if Q > N:
        return POS_INF
    q_tilde = s_syn[Q - 1]
    lo, hi = rank_windows(m, N, beta)
    s_ext = np.concatenate([s_syn, [POS_INF]])
    edges = np.concatenate([[NEG_INF], s_real, [POS_INF]])
    best = NEG_INF
    for r in range(1, m + 2):
        a, b = edges[r - 1], edges[r]
        if not a < b:                                   # empty interval (tied reals)
            continue
        L, U = s_ext[lo[r - 1] - 1], s_ext[hi[r - 1] - 1]
        if U <= q_tilde:
            best = max(best, b)
        elif L > q_tilde:
            continue
        else:
            win = s_ext[lo[r - 1] - 1: hi[r - 1]]
            nxt = win[win > q_tilde][0]
            if nxt > a:
                best = max(best, min(b, nxt))
    return float(best)
