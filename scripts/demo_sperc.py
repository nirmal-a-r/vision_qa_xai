"""
demo_sperc.py - see SPERC work in under a minute, with no data and no GPU.

Simulates a new inspection line with only a handful of real defect images and a
large set of synthetic ones, then compares three ways of certifying the escape
rate:

  CRC      conformal risk control on the real defects only (the standard method)
  SPERC    this project: real defects + synthetic defects, two-tier certificate
  naive    real + synthetic pooled into CRC (looks attractive, is NOT valid)

for a GOOD generator (synthetic scores look like real ones) and a BAD one
(synthetic defects are far too easy to detect).

    python scripts/demo_sperc.py
"""

import os
import sys

import numpy as np
from scipy.stats import beta as Beta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.baselines import naive_pooled_crc_threshold   # noqa: E402
from src.risk.sperc import certify_crc, certify_sperc       # noqa: E402

REAL = (5, 2)                   # escape scores of real defects ~ Beta(5, 2)
GENERATORS = {"good generator": (5, 2), "bad generator": (12, 1.5)}
M, N, TARGET, BETA, TRIALS = 15, 1000, 0.10, 0.05, 1000


def p_escape(lam):
    """True escape probability of a threshold (known here because data is simulated)."""
    return 0.0 if not np.isfinite(lam) else float(Beta.cdf(lam, *REAL))


def main():
    rng = np.random.default_rng(0)
    print(f"A new line with m = {M} real defect images and N = {N} synthetic ones.")
    print(f"Target escape rate {TARGET:.2f}. SPERC runs at alpha = {TARGET - BETA:.2f}, beta = {BETA}.\n")
    for gname, syn in GENERATORS.items():
        crc, sp, nv, issued_crc, cap = [], [], [], 0, None
        for _ in range(TRIALS):
            cr, cs = rng.beta(*REAL, M), rng.beta(*syn, N)
            c = certify_crc(cr, TARGET)
            s = certify_sperc(cr, cs, TARGET - BETA, beta=BETA)
            crc.append(p_escape(c.threshold)); issued_crc += c.issued
            sp.append(p_escape(s.threshold)); cap = s.tier_h_cap
            nv.append(p_escape(naive_pooled_crc_threshold(cr, cs, TARGET)))
        print(f"--- {gname} ---")
        print(f"  CRC   (real only)  mean escape {np.mean(crc):.3f}   guaranteed <= {TARGET:.3f}"
              f"   (issued {issued_crc / TRIALS:.0%}; only the lowest real score is certifiable)")
        print(f"  SPERC (two-tier)   mean escape {np.mean(sp):.3f}   Tier N <= {TARGET:.3f} + eps,"
              f" Tier H <= {cap:.3f}")
        print(f"  naive pooling      mean escape {np.mean(nv):.3f}   no valid guarantee\n")
    print("Reading: with a good generator SPERC operates near its target; with a bad one it")
    print("degrades to, but not beyond, its hard Tier H cap. Naive pooling breaks badly.")


if __name__ == "__main__":
    main()
