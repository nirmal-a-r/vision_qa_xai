"""
baselines.py
============
Two calibration rules kept ONLY as baselines, so the paper can show why they are
not used. Neither carries a valid synthetic-data guarantee.

* ``naive_pooled_crc_threshold`` - stack real and synthetic scores into one CRC
  calibration set. Invalid: it treats synthetic points as exchangeable with the
  real line. Measured earlier in this project: escape 0.12-0.16 against a 0.10
  target.
* ``legacy_nearest_rank_threshold`` - the rule in the old ``src/risk/spi.py``
  (removed 2026-09). It snaps each real score down to the synthetic grid and keeps
  CRC's budget over the m real points, so its threshold can never exceed the CRC
  threshold: valid, but it cannot gain anything from synthetic data.
"""

from __future__ import annotations

import numpy as np

from src.risk.escape import NEG_INF, crc_escape_threshold


def naive_pooled_crc_threshold(c_real, c_syn, alpha: float) -> float:
    pooled = np.concatenate([np.asarray(c_real, float), np.asarray(c_syn, float)])
    return crc_escape_threshold(pooled, alpha)


def legacy_nearest_rank_threshold(c_real, c_syn, alpha: float) -> float:
    c_real = np.sort(np.asarray(c_real, dtype=float))
    c_syn = np.sort(np.asarray(c_syn, dtype=float))
    m, N = c_real.size, c_syn.size
    if m == 0:
        return NEG_INF
    if N == 0:
        return crc_escape_threshold(c_real, alpha)
    transported = np.clip(np.searchsorted(c_syn, c_real, side="left"), 1, N)
    transported.sort()
    k = np.arange(m + 1)
    ok = np.flatnonzero(k / (m + 1.0) + 1.0 / (m + 1.0) <= alpha)
    if ok.size == 0 or int(ok[-1]) < 1:
        return NEG_INF
    return float(c_syn[int(transported[int(ok[-1]) - 1]) - 1])
