"""
Stage 0 of the experimental protocol (project document, Section 12):
``src/risk/spi_exact.py`` must reproduce SPI's guarantees on synthetic Beta scores,
and adversarial synthetic data must stay under the Tier H cap. Stop rule: realised
coverage outside SPI's Theorem 3.3 / 3.5 bounds = implementation bug.

Scores are Beta-distributed so the true escape probability of any threshold is
known in closed form (no test-set noise): for part escape score c ~ Beta(5, 2),
P(escape at lambda) = P(c < lambda) = BetaCDF(lambda). Monte-Carlo checks allow
4 standard errors of the draw-to-draw noise.
"""

import os
import sys

import numpy as np
from scipy.stats import beta as Beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.escape import break_ties  # noqa: E402
from src.risk.spi_exact import (coverage_lower_worst_case, coverage_upper_worst_case,  # noqa: E402
                                rank_pmf, rank_windows, select_beta_alg4, spi_covers,
                                spi_score_threshold, tier_h_cap, tier_h_floor,
                                tier_n_interval, transport)
from src.risk.sperc import certify_sperc, commissioning_table  # noqa: E402

REAL = (5, 2)


def _p_escape(lam):
    if np.isnan(lam):
        raise AssertionError("nan threshold")
    if lam == -np.inf:
        return 0.0
    if lam == np.inf:
        return 1.0
    return float(Beta.cdf(lam, *REAL))


def _mc(esc):
    esc = np.asarray(esc, float)
    return float(esc.mean()), float(esc.std(ddof=1) / np.sqrt(esc.size))


# ------------------------------------------------------------------ the rank law
def test_rank_pmf_matches_simulation():
    """p_{m,N,r} is the law of the r-th real order statistic's position among N synthetic."""
    rng = np.random.default_rng(0)
    m, N, reps = 6, 20, 40000
    for r in (1, 3, m + 1):
        pos = np.empty(reps, int)
        for i in range(reps):
            x = rng.random(m + 1)             # m real calibration + 1 test = m+1 real
            y = rng.random(N)
            xr = np.sort(x)[r - 1]
            pos[i] = 1 + np.sum(y < xr)       # k in 1..N+1
        emp = np.bincount(pos, minlength=N + 2)[1:] / reps
        p = rank_pmf(m, N, r)
        assert abs(p.sum() - 1) < 1e-10 and (p >= 0).all()
        assert np.abs(emp - p).max() < 0.012, (r, np.abs(emp - p).max())
    print("  p_{m,N,r} sums to 1 and matches a 40,000-draw simulation")


def test_windows_valid_monotone_and_data_free():
    for m, N, b in [(5, 1000, 0.05), (15, 1000, 0.05), (40, 300, 0.2), (3, 7, 0.5)]:
        lo, hi = rank_windows(m, N, b)
        assert lo.shape == hi.shape == (m + 1,)
        assert (1 <= lo).all() and (hi <= N + 1).all() and (lo <= hi).all()
        assert (np.diff(lo) >= 0).all() and (np.diff(hi) >= 0).all()
    # larger beta = narrower windows = smaller (or equal) Tier H cap
    caps = [tier_h_cap(15, 1000, 0.05, b) for b in np.arange(0.01, 0.99, 0.01)]
    assert all(a >= b - 1e-12 for a, b in zip(caps, caps[1:]))
    print("  1 <= R^- <= R^+ <= N+1, non-decreasing in r; Tier H cap non-increasing in beta")


# ------------------------------------------------------------------ Section 8.4
DOC_TABLE = {   # m: (alpha .05 beta .05, alpha .05 beta .20, alpha .10 beta .05), N = 1000
    5: (0.333, 0.167, 0.333), 10: (0.182, 0.182, 0.273), 15: (0.188, 0.125, 0.250),
    25: (0.154, 0.115, 0.231), 40: (0.122, 0.098, 0.195)}


def test_commissioning_table_matches_project_document():
    for m, want in DOC_TABLE.items():
        got = (tier_h_cap(m, 1000, 0.05, 0.05), tier_h_cap(m, 1000, 0.05, 0.20),
               tier_h_cap(m, 1000, 0.10, 0.05))
        assert tuple(round(g, 3) for g in got) == want, (m, got, want)
    # the cross-check quoted in Section 8.4: coverage floor 0.812 at m=15, a=b=0.05
    assert abs(coverage_lower_worst_case(15, 1000, 0.05, 0.05) - 13 / 16) < 1e-12
    rows = commissioning_table([5, 10, 15, 25, 40], 1000, 0.10, [0.05])
    assert [r["crc_certifiable_order_statistic"] for r in rows] == [0, 1, 1, 2, 4]   # Section 8.4 CRC column
    print("  every Tier H cap in the document's Section 8.4 table reproduced to 3 decimals")


# ------------------------------------------------------------------ transporter
def test_threshold_equals_spi_set_brute_force():
    """The one-sided threshold rule and SPI's transporter agree on every test score."""
    rng = np.random.default_rng(7)
    for trial in range(150):
        m, N = int(rng.integers(1, 25)), int(rng.integers(1, 250))
        a, b = float(rng.uniform(0.02, 0.4)), float(rng.choice([0.02, 0.05, 0.1, 0.3]))
        sr = np.round(rng.normal(size=m), 1)                # heavy ties on purpose
        ss = np.round(rng.normal(0.3, 1.2, N), 1)
        t = spi_score_threshold(sr, ss, a, b)
        etas = np.concatenate([np.linspace(-5, 5, 201), sr, ss])
        cov = spi_covers(etas, sr, ss, a, b)
        assert np.all(etas[cov] <= t + 1e-12), trial        # rule covers all SPI covers
        assert np.all(cov[etas < t - 1e-12]), trial         # and nothing below t is missed
    print("  threshold rule == SPI prediction set (up to the boundary point) on 150 tie-heavy cases")


def test_transport_definition():
    sr, ss = np.array([0.0, 1.0, 2.0]), np.linspace(-3, 3, 61)
    lo, hi = rank_windows(3, 61, 0.1)
    ext = np.concatenate([np.sort(ss), [np.inf]])
    for eta in np.linspace(-4, 4, 97):
        T = transport(eta, sr, ss, 0.1)
        r = 1 + int(np.sum(sr < eta))
        L, U = ext[lo[r - 1] - 1], ext[hi[r - 1] - 1]
        assert L <= T <= U                                  # T lands in the rank window
        if L <= eta < U:
            assert T <= eta and T in ext                    # lower nearest neighbour
    print("  T(eta) stays in its rank window and is the lower nearest synthetic neighbour")


# ------------------------------------------------------------------ Theorem 3.3
def test_tier_n_bounds_when_aligned():
    """Aligned synthetic data (same law as real): escape inside [a-b-1/(N+1), a+b]."""
    rng = np.random.default_rng(1)
    for m, N, a, b in [(15, 1000, 0.05, 0.05), (5, 500, 0.10, 0.05), (40, 1000, 0.10, 0.20)]:
        esc = [_p_escape(certify_sperc(rng.beta(*REAL, m), rng.beta(*REAL, N), a, beta=b).threshold)
               for _ in range(800)]
        mean, se = _mc(esc)
        lo, hi = tier_n_interval(a, b, N)
        assert lo - 4 * se <= mean <= hi + 4 * se, (m, N, a, b, mean, lo, hi)
        print(f"  m={m:>2} N={N} a={a} b={b}: mean escape {mean:.4f} in Tier N [{max(lo, 0):.3f}, {hi:.3f}]")


# ------------------------------------------------------------------ Theorem 3.5
def _adversaries(rng, m, N):
    return {
        "far too easy": (rng.beta(*REAL, m), lambda cr: rng.beta(12, 1.5, N)),
        "far too hard": (rng.beta(*REAL, m), lambda cr: rng.beta(1.5, 12, N)),
        "uniform noise": (rng.beta(*REAL, m), lambda cr: rng.random(N)),
        # Corollary 3.6: synthetic data built FROM the real calibration defects
        "copies of real, shifted easy": (rng.beta(*REAL, m),
                                        lambda cr: np.clip(np.repeat(cr, N // m + 1)[:N] + 0.2, 0, 1)),
    }


def test_tier_h_holds_for_any_synthetic_data():
    m, N, a, b = 15, 1000, 0.05, 0.05
    cap, floor = tier_h_cap(m, N, a, b), tier_h_floor(m, N, a, b)
    for name in _adversaries(np.random.default_rng(0), m, N):
        rng = np.random.default_rng(abs(hash(name)) % 2**32)
        esc = []
        for _ in range(800):
            cr, make = _adversaries(rng, m, N)[name]
            esc.append(_p_escape(certify_sperc(cr, make(cr), a, beta=b).threshold))
        mean, se = _mc(esc)
        assert floor - 4 * se <= mean <= cap + 4 * se, (name, mean, floor, cap)
        print(f"  {name:>30s}: mean escape {mean:.4f} within Tier H [{floor:.3f}, {cap:.3f}]")


def test_worst_case_bounds_are_ordered():
    for m in (5, 15, 40):
        for a in (0.05, 0.10):
            for b in (0.05, 0.2):
                lo = coverage_lower_worst_case(m, 1000, a, b)
                hi = coverage_upper_worst_case(m, 1000, a, b)
                assert 0 <= lo <= 1 - a + 1e-9 <= hi + 1e-9 <= 1 + 1e-9 or lo <= hi
                assert lo <= hi
    print("  Theorem 3.5 lower bound <= upper bound everywhere")


# ------------------------------------------------------------------ Algorithm 4
def test_algorithm4_picks_smallest_feasible_beta():
    m, N, a = 15, 1000, 0.05
    for amax in (0.15, 0.20, 0.30):
        b = select_beta_alg4(m, N, a, 1 - amax, step=0.005)
        assert b is not None and tier_h_cap(m, N, a, b) <= amax + 1e-12
        prev = round(b - 0.005, 10)
        if prev > 0:
            assert tier_h_cap(m, N, a, prev) > amax, (amax, b)
    # A near-1 floor is only reachable by the TRIVIAL set: when every R_j^+ <= Q, every
    # score is covered, so the certificate must be a refusal (review everything).
    b = select_beta_alg4(5, 1000, 0.05, 1 - 0.01, step=0.01)
    assert b is not None and tier_h_cap(5, 1000, 0.05, b) == 0.0
    rng = np.random.default_rng(3)
    assert not certify_sperc(rng.random(5), rng.random(1000), 0.05, beta=b).issued
    assert select_beta_alg4(5, 1000, 0.05, 1 - 0.01, step=0.01, beta_max=0.3) is None
    print("  Algorithm 4 returns the first step-grid beta meeting the floor; a cap of 0 means refusal")


# ------------------------------------------------------------------ ties
def test_random_tie_breaking_keeps_guarantee():
    """Discrete scores (3 decimals) + missed defects (-inf), ties broken at random."""
    rng = np.random.default_rng(5)
    m, N, a, b = 15, 1000, 0.05, 0.05
    cap = tier_h_cap(m, N, a, b)
    esc = []
    for _ in range(800):
        cr = np.round(rng.beta(*REAL, m), 2)
        cs = np.round(rng.beta(*REAL, N), 2)
        cr[rng.random(m) < 0.05] = -np.inf
        cs[rng.random(N) < 0.05] = -np.inf
        cert = certify_sperc(break_ties(cr, rng), break_ties(cs, rng), a, beta=b)
        # true escape of the tie-broken rule on a fresh part from the same discrete law
        lam = cert.threshold
        grid = np.round(np.linspace(0, 1, 101), 2)
        pk = np.diff(Beta.cdf(np.concatenate([[-np.inf], (grid[:-1] + grid[1:]) / 2, [np.inf]]), *REAL))
        p = 0.05 + 0.95 * float(np.sum(pk[grid < lam]))     # -inf always escapes (lam > -1 here)
        p += 0.95 * float(np.sum(pk[np.isclose(grid, np.floor(lam * 100) / 100)])) * 0.5
        esc.append(p if np.isfinite(lam) else 0.0)
    mean, se = _mc(esc)
    assert mean <= cap + 4 * se, (mean, cap)
    assert mean <= a + b + 0.03, mean       # aligned: near Tier N
    print(f"  tie-heavy scores with missed defects: mean escape {mean:.4f} (Tier N {a + b:.2f}, Tier H {cap:.3f})")


if __name__ == "__main__":
    from _runner import run
    run(globals())
