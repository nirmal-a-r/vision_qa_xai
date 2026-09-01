"""Tests for reporting diagnostics; these must never be mistaken for certificates."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.rigor import operating_metrics, threshold_stability, wilson_interval


def record(defect, score, matched=True):
    gt = [[0, 0, 10, 10]] if defect else []
    box = [[0, 0, 10, 10]] if matched else [[20, 20, 30, 30]]
    return {"gt_boxes": gt, "pred_boxes": box, "pred_scores": [score]}


def test_wilson_interval_contains_observed_proportion():
    lo, hi = wilson_interval(3, 20)
    assert lo <= 0.15 <= hi
    print("Wilson interval contains observed proportion")


def test_operating_metrics_routing_is_conservative():
    # The unmatched high-score defect is flagged (reviewed), never accepted.
    rows = [record(True, .1, True), record(True, .9, False), record(False, .2)]
    m = operating_metrics(rows, .5)
    assert m["auto_accept_rate"] == 2 / 3
    # Both defective parts are accepted by the low threshold; one was matched
    # only below it and one has no matched detection at all.
    assert m["escape_count"] == 2
    assert m["review_burden"] == 1 / 3
    print("routing and matched-detection escape are reported separately")


def test_stability_keeps_refusals_explicit():
    d = threshold_stability(np.array([.1, .2, .3]), .05, n_boot=100, seed=1)
    assert d["issue_rate"] == 0.0 and d["threshold_median"] is None
    print("bootstrap refusals are not fabricated into thresholds")


if __name__ == "__main__":
    for fn in [v for k, v in globals().items() if k.startswith("test_")]:
        fn()
    print("3/3 passed")
