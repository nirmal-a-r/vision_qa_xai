"""
sperc.py
========
SPERC - Synthetic-Powered Escape-Risk Certification.

Certifies a frozen detector's defect-escape rate from a FEW real defective
calibration images plus MANY synthetic (generated) defective images, by applying
the exact SPI transporter (``src/risk/spi_exact.py``; Bashari et al., NeurIPS 2025)
to the scalar escape scores of ``src/risk/escape.py``.

What is guaranteed (read this before using the numbers)
-------------------------------------------------------
With nonconformity s = -c (large s = hard-to-catch defect), escape is exactly
SPI's non-coverage (project document, Section 2.3), so SPI's theorems translate
directly:

* **Tier N (near-nominal, SPI Theorem 3.3).**
      alpha - beta - eps - 1/(N+1)  <=  P(escape)  <=  alpha + beta + eps
  eps = mean total-variation distance between the order-statistic laws of real
  and synthetic escape scores. Tight when the generator is good; eps is NOT
  observable (the KS distance in the sweep only estimates alignment).

* **Tier H (hard, SPI Theorem 3.5), for ANY synthetic data:**
      P(escape) <= 1 - #{j in [m+1] : R_j^+ <= ceil((1-alpha)(N+1))} / (m+1)
  It depends only on (m, N, alpha, beta), so it can be computed before any data
  is collected (``commissioning_table``). It can be far above alpha, e.g. 0.1875
  at m=15, N=1000, alpha=0.05, beta=0.05. Synthetic data may even depend on the
  real calibration data (SPI Corollary 3.6).

* **Refusal.** If no beta makes the Tier H cap at most the plant's ``alpha_max``,
  SPERC refuses (threshold -inf: every part goes to review), exactly as CRC
  refuses when 1/(m+1) > alpha.

Both statements average over the calibration draw, like every conformal bound;
neither is a per-part guarantee.

Deployment rule: auto-accept a part iff its escape score c < threshold (for part
escape: iff no detection scores at least the threshold).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from src.risk.escape import NEG_INF, crc_escape_threshold
from src.risk.spi_exact import (POS_INF, coverage_lower_worst_case,  # noqa: F401  (re-exported)
                                rank_pmf, rank_windows, select_beta_alg4,
                                select_beta_grid, spi_covers, spi_score_threshold,
                                tier_h_cap, tier_h_floor, tier_n_bound,
                                tier_n_interval, transport)

DEFAULT_BETA_GRID = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def select_beta(m: int, N: int, alpha: float, alpha_max: float,
                grid: Optional[Sequence[float]] = DEFAULT_BETA_GRID,
                step: Optional[float] = None) -> Optional[float]:
    """Smallest beta whose Tier H cap is <= alpha_max (SPI Algorithm 4).

    ``step`` given: Algorithm 4 exactly as in the paper (beta = step, 2 step, ...).
    Otherwise: the same search restricted to ``grid``. Smallest, because beta also
    loosens Tier N by the same amount. None = no beta meets the cap (refuse).
    Uses only (m, N, alpha, alpha_max): selecting beta does not touch the data.
    """
    floor = 1.0 - alpha_max
    if step is not None:
        return select_beta_alg4(m, N, alpha, floor, step=step)
    return select_beta_grid(m, N, alpha, floor, grid or DEFAULT_BETA_GRID)


@dataclass
class Certificate:
    """Everything a plant needs to know about one calibration."""
    method: str
    threshold: float          # escape-score threshold: accept a part iff c < threshold
    issued: bool              # False = refusal: review every part
    alpha: float
    m: int
    N: int = 0
    beta: Optional[float] = None
    tier_h_cap: Optional[float] = None   # hard, distribution-free escape bound
    tier_n_bound: Optional[float] = None # alpha + beta (+ unobservable eps)
    note: str = ""

    def to_dict(self):
        return asdict(self)


def certify_crc(c_real: np.ndarray, alpha: float) -> Certificate:
    """Real-data-only baseline: exact conformal risk control on 0/1 escape."""
    c_real = np.asarray(c_real, dtype=float)
    lam = crc_escape_threshold(c_real, alpha)
    issued = bool(np.isfinite(lam))
    return Certificate("crc", lam, issued, alpha, int(c_real.size),
                       tier_h_cap=alpha if issued else 0.0,
                       note="" if issued else "1/(m+1) > alpha: no certifiable threshold")


def certify_sperc(c_real: np.ndarray, c_syn: np.ndarray, alpha: float,
                  beta: Optional[float] = None, alpha_max: Optional[float] = None,
                  beta_grid: Optional[Sequence[float]] = None,
                  beta_step: Optional[float] = None) -> Certificate:
    """Two-tier SPERC certificate from real and synthetic ESCAPE SCORES (c, not s).

    Parameters
    ----------
    c_real, c_syn : escape scores of real / synthetic defective images
        (``escape.escape_scores``). ``-inf`` entries are allowed; break ties first
        (``escape.break_ties``) when scores are discrete.
    alpha : SPI level in (0, 1). Tier N is alpha + beta, so to target escape ``a``
        under alignment run with alpha = a - beta.
    beta : rank-window tolerance. If None, chosen by ``select_beta`` to meet
        ``alpha_max`` (which is then required); ``beta_step`` switches that search
        to SPI's Algorithm 4 on a step grid.
    alpha_max : maximum tolerable worst-case escape. If given and the Tier H cap
        exceeds it, the certificate is a refusal.
    """
    c_real = np.asarray(c_real, dtype=float)
    c_syn = np.asarray(c_syn, dtype=float)
    m, N = int(c_real.size), int(c_syn.size)
    if not 0.0 < alpha < 1.0:
        return Certificate("sperc", NEG_INF, False, float(alpha), m, N, beta,
                           note=f"alpha={alpha:.4f} outside (0, 1): nothing to certify; refuse")
    if m == 0 or N == 0:
        return Certificate("sperc", NEG_INF, False, alpha, m, N, beta,
                           note="need at least one real and one synthetic score; refuse")
    if beta is None:
        if alpha_max is None:
            raise ValueError("give beta, or alpha_max so that beta can be selected")
        beta = select_beta(m, N, alpha, alpha_max, grid=beta_grid or DEFAULT_BETA_GRID,
                           step=beta_step)
        if beta is None:
            return Certificate("sperc", NEG_INF, False, alpha, m, N,
                               note=f"no beta gives Tier H cap <= {alpha_max}; refuse")
    cap = tier_h_cap(m, N, alpha, beta)
    if alpha_max is not None and cap > alpha_max + 1e-12:
        return Certificate("sperc", NEG_INF, False, alpha, m, N, beta, cap,
                           tier_n_bound(alpha, beta),
                           note=f"Tier H cap {cap:.4f} > alpha_max {alpha_max}; refuse")
    t = spi_score_threshold(-c_real, -c_syn, alpha, beta)     # s = -c
    lam = -t                                                   # caught iff c >= lam
    issued = bool(np.isfinite(lam))
    return Certificate("sperc", float(lam), issued, alpha, m, N, beta, cap,
                       tier_n_bound(alpha, beta),
                       note="" if issued else "synthetic grid cannot resolve alpha; refuse")


def commissioning_table(ms: Sequence[int], N: int, alpha: float,
                        betas: Sequence[float]):
    """Rows of (m, beta, Tier H cap, Tier N bound, CRC issuable) - no data needed.

    This is the project document's Section 8.4 table: what a plant can be promised
    before it has collected a single defect image.
    """
    rows = []
    for m in ms:
        j = int(np.floor(alpha * (m + 1) + 1e-9))
        for b in betas:
            rows.append({"m": int(m), "N": int(N), "alpha": alpha, "beta": b,
                         "tier_h_cap": tier_h_cap(m, N, alpha, b),
                         "tier_h_floor": tier_h_floor(m, N, alpha, b),
                         "tier_n_bound": tier_n_bound(alpha, b),
                         "crc_issuable_at_alpha": bool(j >= 1),
                         "crc_certifiable_order_statistic": j})
    return rows
