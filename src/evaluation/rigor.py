"""Auditable reporting metrics for conformal defect inspection.

This module deliberately separates three quantities that are often conflated:

* the *CRC guarantee* (an expectation over a fresh exchangeable part);
* realised held-out performance (a noisy estimate, reported with an interval);
* operational cost (the fraction of parts routed away from auto-accept).

The functions do not create another certificate.  They make the certificate
auditable: a paper reader can see how sensitive the issued threshold is to the
calibration draw and to the IoU definition of a matched defect.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np

from src.risk.escape import NEG_INF, crc_escape_threshold as crc_threshold, empirical_escape as empirical_risk
from src.risk.escape import escape_scores


def escape_confidences(records, iou_thresh=0.5):
    """Localised escape scores of the defective images (see src/risk/escape.py)."""
    return escape_scores(records, "localized", iou_thresh)


def max_detection_score(record: dict) -> float:
    """Detector score used by the deployment routing rule."""
    scores = np.asarray(record.get("pred_scores", []), dtype=float)
    return float(scores.max()) if scores.size else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """Two-sided Wilson interval for a binomial proportion.

    This describes sampling uncertainty in a held-out measurement only.  It is
    not substituted for, or combined with, the distribution-free CRC bound.
    """
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    den = 1.0 + z * z / total
    ctr = (p + z * z / (2.0 * total)) / den
    rad = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return (float(max(0.0, ctr - rad)), float(min(1.0, ctr + rad)))


def threshold_stability(confidences: np.ndarray, alpha: float, n_boot: int = 1000,
                        seed: int = 0) -> dict:
    """Non-parametric bootstrap diagnostic for the calibration threshold.

    Refusals are retained as a separate outcome rather than coerced into a
    numeric threshold.  The bootstrap is a stability diagnostic, not a new
    validity proof (resampling is not the CRC calibration procedure).
    """
    c = np.asarray(confidences, dtype=float)
    if c.size == 0:
        return {"n_boot": n_boot, "issue_rate": 0.0, "threshold_q025": None,
                "threshold_median": None, "threshold_q975": None}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        threshold = crc_threshold(c[rng.integers(0, c.size, size=c.size)], alpha)
        if threshold > NEG_INF / 2:
            draws.append(threshold)
    if not draws:
        return {"n_boot": n_boot, "issue_rate": 0.0, "threshold_q025": None,
                "threshold_median": None, "threshold_q975": None}
    q = np.quantile(draws, [0.025, 0.5, 0.975])
    return {"n_boot": n_boot, "issue_rate": float(len(draws) / n_boot),
            "threshold_q025": float(q[0]), "threshold_median": float(q[1]),
            "threshold_q975": float(q[2])}


def operating_metrics(test_records: Iterable[dict], threshold: float,
                      iou_thresh: float = 0.5) -> dict:
    """Held-out risk and workflow burden for a fixed, already-calibrated rule.

    Routing is conservative: a part is auto-accepted only when *no* detector
    score reaches the CRC threshold.  Therefore every accepted defective part
    is included in the matched-detection escape loss used for calibration; the
    CRC loss upper-bounds the deployed escape event.
    """
    records = list(test_records)
    defect = np.asarray([bool(r.get("gt_boxes")) for r in records], dtype=bool)
    scores = np.asarray([max_detection_score(r) for r in records], dtype=float)
    flagged = scores >= threshold
    c = escape_confidences(records, iou_thresh)
    escaped = int(np.sum(c < threshold))
    n_def = int(c.size)
    lo, hi = wilson_interval(escaped, n_def)
    clean = ~defect
    return {
        "n_parts": int(len(records)), "n_defective": n_def,
        "escape_risk": empirical_risk(c, threshold),
        "escape_count": escaped, "escape_wilson95_low": lo,
        "escape_wilson95_high": hi,
        "review_burden": float(flagged.mean()) if flagged.size else 0.0,
        "auto_accept_rate": float((~flagged).mean()) if flagged.size else 0.0,
        "clean_review_rate": float(flagged[clean].mean()) if clean.any() else None,
        "defect_flag_rate": float(flagged[defect].mean()) if defect.any() else None,
    }


def audit_pair(cal_records: list[dict], test_records: list[dict], alphas=(0.05, 0.10, 0.20),
               ious=(0.30, 0.50, 0.75), n_boot: int = 1000, seed: int = 0) -> dict:
    """Audit one fixed cal/test split across targets and IoU definitions."""
    report = {"n_cal_parts": len(cal_records), "n_test_parts": len(test_records),
              "ious": list(ious), "alphas": list(alphas), "results": {}}
    for iou in ious:
        cal_c = escape_confidences(cal_records, iou)
        report["results"][str(iou)] = {"n_cal_defective": int(cal_c.size),
                                         "alpha_floor": float(1.0 / (cal_c.size + 1)),
                                         "targets": {}}
        for alpha in alphas:
            threshold = crc_threshold(cal_c, alpha)
            issued = threshold > NEG_INF / 2
            row = {"issued": issued, "threshold": float(threshold) if issued else None}
            if issued:
                row.update(operating_metrics(test_records, threshold, iou))
                row["threshold_stability"] = threshold_stability(
                    cal_c, alpha, n_boot=n_boot, seed=seed + int(iou * 1000) + int(alpha * 10000))
            report["results"][str(iou)]["targets"][str(alpha)] = row
    return report


def audit_prediction_directory(pred_dir: str, out_path: str, **kwargs) -> dict:
    """Run the audit over every cached ``*_cal.json`` / ``*_test.json`` pair."""
    report = {"protocol": "fixed-split certificate-quality audit", "pairs": {}}
    for name in sorted(os.listdir(pred_dir)):
        if not name.endswith("_cal.json"):
            continue
        test_name = name.replace("_cal.json", "_test.json")
        test_path = os.path.join(pred_dir, test_name)
        if not os.path.exists(test_path):
            continue
        with open(os.path.join(pred_dir, name), encoding="utf-8") as f:
            cal = json.load(f)["records"]
        with open(test_path, encoding="utf-8") as f:
            test = json.load(f)["records"]
        report["pairs"][name.replace("_cal.json", "")] = audit_pair(cal, test, **kwargs)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
