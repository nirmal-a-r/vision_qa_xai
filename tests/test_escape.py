"""
Tests for src/risk/escape.py - the two escape events and the exact CRC threshold.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.escape import (NEG_INF, crc_escape_threshold, empirical_escape,  # noqa: E402
                             escape_scores, localized_escape_score, part_escape_score)
from src.risk.conformal import conformal_risk_control  # noqa: E402

REC_CLEAN = {"gt_boxes": [], "pred_boxes": [[0, 0, 5, 5]], "pred_scores": [0.9]}
REC_NONE = {"gt_boxes": [[0, 0, 10, 10]], "pred_boxes": [], "pred_scores": []}
REC_TWO = {"gt_boxes": [[0, 0, 10, 10], [50, 50, 60, 60]],
           "pred_boxes": [[0, 0, 10, 10], [1, 1, 11, 11], [200, 200, 210, 210]],
           "pred_scores": [0.3, 0.8, 0.95]}


def test_scores_on_hand_built_records():
    assert np.isnan(part_escape_score(REC_CLEAN)) and np.isnan(localized_escape_score(REC_CLEAN))
    assert part_escape_score(REC_NONE) == NEG_INF and localized_escape_score(REC_NONE) == NEG_INF
    # part: any detection counts -> 0.95; localised: box 2 is never matched -> -inf
    assert part_escape_score(REC_TWO) == 0.95
    assert localized_escape_score(REC_TWO) == NEG_INF
    rec = dict(REC_TWO, pred_boxes=REC_TWO["pred_boxes"] + [[50, 50, 60, 60]],
               pred_scores=REC_TWO["pred_scores"] + [0.4])
    # box 1 best match 0.8, box 2 best match 0.4 -> weakest-covered = 0.4
    assert localized_escape_score(rec) == 0.4
    s = escape_scores([REC_CLEAN, REC_NONE, rec], "localized")
    assert s.size == 2, "clean images must be dropped"
    print("  part / localised scores, clean-image exclusion correct")


def test_crc_threshold_is_exact_order_statistic():
    c = np.arange(1, 20, dtype=float)                 # m = 19
    assert crc_escape_threshold(c, 0.05) == 1.0       # j* = floor(0.05*20) = 1
    assert crc_escape_threshold(c, 0.10) == 2.0       # j* = 2
    assert crc_escape_threshold(c[:18], 0.05) == NEG_INF   # m = 18: 1/19 > 0.05 -> refuse
    print("  j* = floor(alpha (m+1)); refusal when 1/(m+1) > alpha")


def test_crc_threshold_agrees_with_generic_crc():
    rng = np.random.default_rng(3)
    for m, alpha in [(19, 0.05), (40, 0.10), (100, 0.2)]:
        c = rng.random(m)
        lam = np.concatenate([np.sort(c)[::-1], [-1.0]])          # descending grid
        losses = (c[:, None] < lam[None, :]).astype(float)
        generic = conformal_risk_control(losses, lam, alpha)
        assert abs(generic - crc_escape_threshold(c, alpha)) < 1e-12, (m, alpha)
    print("  quantile form matches conformal_risk_control on the 0/1 loss")


def test_crc_validity_monte_carlo():
    rng = np.random.default_rng(0)
    for m, alpha in [(9, 0.10), (19, 0.05), (30, 0.10)]:
        # uniform scores: P(escape at c_(j)) = c_(j) exactly
        risk = [crc_escape_threshold(rng.random(m), alpha) for _ in range(6000)]
        risk = np.where(np.isfinite(risk), risk, 0.0)
        j = int(np.floor(alpha * (m + 1) + 1e-9))
        assert abs(np.mean(risk) - j / (m + 1)) < 0.006, (m, np.mean(risk))
        assert np.mean(risk) <= alpha + 0.006
    print("  mean escape = j*/(m+1) <= alpha")


def test_empirical_escape():
    assert empirical_escape(np.array([0.1, 0.5, NEG_INF]), 0.2) == 2 / 3
    print("  -inf always escapes")


if __name__ == "__main__":
    from _runner import run
    run(globals())
