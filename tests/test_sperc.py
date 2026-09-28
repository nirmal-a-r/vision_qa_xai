"""
Tests for src/risk/sperc.py - exact SPI rank windows, both certificate tiers,
beta selection and refusal. Monte-Carlo checks use Beta-distributed scores so the
true escape probability of a threshold is known in closed form.
"""

import os
import sys

import numpy as np
from scipy.stats import beta as Beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.sperc import (certify_crc, certify_sperc, rank_pmf, rank_windows,  # noqa: E402
                            select_beta, spi_score_threshold, tier_h_cap)
from src.risk.baselines import naive_pooled_crc_threshold  # noqa: E402

REAL = (5, 2)


def _p_escape(lam):
    return 0.0 if not np.isfinite(lam) else float(Beta.cdf(lam, *REAL))


def test_rank_pmf_is_a_distribution():
    for m, N in [(5, 50), (15, 1000), (40, 300)]:
        for r in (1, (m + 1) // 2, m + 1):
            p = rank_pmf(m, N, r)
            assert p.shape == (N + 1,) and abs(p.sum() - 1) < 1e-9 and (p >= 0).all()
    print("  p_{m,N,r} sums to 1")


def test_windows_valid_and_monotone():
    lo, hi = rank_windows(15, 1000, 0.05)
    assert (1 <= lo).all() and (hi <= 1001).all() and (lo <= hi).all()
    assert (np.diff(lo) >= 0).all() and (np.diff(hi) >= 0).all()
    print("  1 <= R^- <= R^+ <= N+1, non-decreasing in r")


def test_guardrail_matches_paper_example():
    # SPI: at m=15, N=1000, alpha=0.05 a beta exists with coverage floor >= 0.80
    assert abs((1 - tier_h_cap(15, 1000, 0.05, 0.05)) - 13 / 16) < 1e-12
    assert 1 - tier_h_cap(15, 1000, 0.05, 0.05) >= 0.80
    print("  coverage floor 0.8125 at (15, 1000, 0.05, 0.05)")


def test_threshold_covers_every_score_spi_covers():
    """Brute force: apply SPI's transporter to many test scores directly."""
    rng = np.random.default_rng(7)
    m, N, alpha, beta = 12, 300, 0.1, 0.1
    s_real, s_syn = np.sort(rng.normal(size=m)), np.sort(rng.normal(0.3, 1.2, N))
    t = spi_score_threshold(s_real, s_syn, alpha, beta)
    lo, hi = rank_windows(m, N, beta)
    ext = np.concatenate([s_syn, [np.inf]])
    q = ext[int(np.ceil((1 - alpha) * (N + 1))) - 1]
    for s in np.linspace(-4, 5, 4001):
        r = 1 + int(np.sum(s_real < s))
        L, U = ext[lo[r - 1] - 1], ext[hi[r - 1] - 1]
        win = ext[lo[r - 1] - 1: hi[r - 1]]
        T = U if s >= U else (L if s < L else win[win <= s].max())
        if T <= q:
            assert s <= t + 1e-12, (s, t)
    print("  threshold rule is a superset of SPI's covered set")


def test_tier_n_when_aligned():
    rng = np.random.default_rng(1)
    m, N, alpha, beta = 15, 1000, 0.05, 0.05
    esc = [_p_escape(certify_sperc(rng.beta(*REAL, m), rng.beta(*REAL, N), alpha, beta).threshold)
           for _ in range(700)]
    mean = float(np.mean(esc))
    assert alpha - beta - 1 / (N + 1) - 0.01 <= mean <= alpha + beta + 0.01, mean
    print(f"  aligned: mean escape {mean:.4f} inside Tier N [{alpha - beta:.3f}, {alpha + beta:.3f}]")


def test_tier_h_under_adversarial_synthetic():
    rng = np.random.default_rng(2)
    m, N, alpha, beta = 15, 1000, 0.05, 0.05
    cap = tier_h_cap(m, N, alpha, beta)
    esc, naive = [], []
    for _ in range(700):
        cr, cs = rng.beta(*REAL, m), rng.beta(12, 1.5, N)     # synthetic far too easy
        esc.append(_p_escape(certify_sperc(cr, cs, alpha, beta).threshold))
        naive.append(_p_escape(naive_pooled_crc_threshold(cr, cs, 0.10)))
    mean = float(np.mean(esc))
    assert mean <= cap + 0.01, (mean, cap)
    assert np.mean(naive) > 0.3, "naive pooling should visibly break"
    print(f"  adversarial: mean escape {mean:.4f} <= Tier H cap {cap:.4f}; naive pooling {np.mean(naive):.3f}")


def test_refusal_and_beta_selection():
    assert not certify_crc(np.random.default_rng(0).random(15), 0.05).issued   # 1/16 > 0.05
    b = select_beta(15, 1000, 0.05, alpha_max=0.15)
    assert b is not None and tier_h_cap(15, 1000, 0.05, b) <= 0.15
    smaller = [g for g in (0.01, 0.02, 0.05, 0.10) if g < b]
    assert all(tier_h_cap(15, 1000, 0.05, g) > 0.15 for g in smaller), "must pick the smallest beta"
    cert = certify_sperc(np.random.default_rng(0).random(5), np.random.default_rng(1).random(1000),
                         0.05, alpha_max=0.01)
    assert not cert.issued and "refuse" in cert.note
    print(f"  CRC refuses at m=15, alpha=0.05; beta={b} selected for alpha_max=0.15; impossible cap refused")


def test_tiny_synthetic_pool_refuses_cleanly():
    # (1-alpha)(N+1) > N: the synthetic grid cannot resolve alpha, SPI covers
    # every score, so SPERC must refuse (review everything) and the cap is 0.
    assert tier_h_cap(25, 5, 0.05, 0.05) == 0.0
    cert = certify_sperc(np.random.default_rng(0).random(25), np.random.default_rng(1).random(5), 0.05, beta=0.05)
    assert not cert.issued and cert.threshold == -np.inf
    print("  N too small for alpha -> refusal, not a crash")


def test_minus_infinity_scores_are_handled():
    rng = np.random.default_rng(4)
    cr = np.concatenate([rng.beta(*REAL, 14), [-np.inf]])
    cs = np.concatenate([rng.beta(*REAL, 995), [-np.inf] * 5])
    cert = certify_sperc(cr, cs, 0.05, beta=0.05)
    assert cert.issued and np.isfinite(cert.threshold)
    print("  missed defects (-inf) in real and synthetic pools handled")


if __name__ == "__main__":
    from _runner import run
    run(globals())
