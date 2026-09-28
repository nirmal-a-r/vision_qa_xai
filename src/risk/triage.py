"""
triage.py
=========
Risk-controlled selective inspection.

A production QA line does not want a detector's raw output. It wants a policy
that sorts every part into one of three bins:

    AUTO_ACCEPT   ship it, no human looks at it
    AUTO_REJECT   pull it, no human looks at it
    HUMAN_REVIEW  send it to an inspector

Only one of those errors is expensive in the way that matters: an AUTO_ACCEPT
on a part that really is defective. That is a **defect escape** - it reaches the
customer. A false AUTO_REJECT costs a part; an escape costs a recall.

So the policy is chosen by constrained optimisation:

    minimise    E[review load]
    subject to  E[escape rate] <= alpha,  certified with confidence 1 - delta

The certification comes from Learn-then-Test over the whole configuration grid.
That choice is deliberate and load-bearing: because LTT controls the family-wise
error rate across *all* configurations simultaneously, we may afterwards pick
whichever certified configuration we like - including the arg-min of review load
- without paying any further multiplicity penalty. Selecting on the same data
used for a per-configuration test would be exactly the winner's-curse mistake
that invalidates the guarantee; LTT is what makes the selection honest.

The third gate, `phi`, is an explanation-faithfulness floor. An image is only
allowed to skip human review if the model's explanation for its decision is
itself trustworthy. This turns XAI from a post-hoc visualisation into a
load-bearing part of the control loop - see `src/xai/faithfulness.py`.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from itertools import product

from .conformal import learn_then_test


ACCEPT, REVIEW, REJECT = 0, 1, 2
_DECISION_NAMES = {ACCEPT: "AUTO_ACCEPT", REVIEW: "HUMAN_REVIEW", REJECT: "AUTO_REJECT"}


@dataclass(frozen=True)
class TriageConfig:
    """One operating point of the policy."""
    lambda_lo: float   # below this max-score, declare the part clean
    lambda_hi: float   # at/above this max-score, reject outright
    phi: float         # minimum explanation faithfulness to skip review

    def decide(self, max_score: float, faithfulness: float) -> int:
        if max_score < self.lambda_lo:
            return ACCEPT
        if max_score >= self.lambda_hi and faithfulness >= self.phi:
            return REJECT
        return REVIEW


def decisions_for_grid(
    max_scores: np.ndarray,
    faithfulness: np.ndarray,
    configs: list[TriageConfig],
) -> np.ndarray:
    """(n, m) matrix of decisions, one column per configuration."""
    n, m = len(max_scores), len(configs)
    out = np.empty((n, m), dtype=np.int8)
    for j, c in enumerate(configs):
        below = max_scores < c.lambda_lo
        reject = (max_scores >= c.lambda_hi) & (faithfulness >= c.phi)
        out[:, j] = np.where(below, ACCEPT, np.where(reject, REJECT, REVIEW))
    return out


def escape_indicator(decisions: np.ndarray, has_defect: np.ndarray) -> np.ndarray:
    """1 when a genuinely defective part was auto-accepted. This is the risk."""
    return ((decisions == ACCEPT) & has_defect[:, None]).astype(float)


def review_indicator(decisions: np.ndarray) -> np.ndarray:
    return (decisions == REVIEW).astype(float)


def false_scrap_indicator(decisions: np.ndarray, has_defect: np.ndarray) -> np.ndarray:
    """1 when a clean part was auto-rejected. Costs a part, not a recall."""
    return ((decisions == REJECT) & ~has_defect[:, None]).astype(float)


def build_grid(
    lambda_lo_vals: np.ndarray,
    lambda_hi_vals: np.ndarray,
    phi_vals: np.ndarray,
) -> list[TriageConfig]:
    """All (lo, hi, phi) combinations with lo <= hi."""
    return [
        TriageConfig(float(lo), float(hi), float(p))
        for lo, hi, p in product(lambda_lo_vals, lambda_hi_vals, phi_vals)
        if lo <= hi
    ]


@dataclass
class TriageSolution:
    config: TriageConfig
    alpha: float
    delta: float
    n_calib: int
    n_configs: int
    n_certified: int
    escape_rate_cal: float
    review_load_cal: float
    false_scrap_cal: float

    def to_dict(self):
        d = asdict(self)
        d["config"] = asdict(self.config)
        return d


class NoCertifiableConfig(RuntimeError):
    """No operating point in the grid can certify the requested escape risk."""


def fit_triage_policy(
    max_scores: np.ndarray,
    faithfulness: np.ndarray,
    has_defect: np.ndarray,
    configs: list[TriageConfig],
    alpha: float,
    delta: float = 0.10,
    max_false_scrap: float | None = None,
    per_defective: bool = True,
) -> TriageSolution:
    """Certify escape risk over the grid, then take the cheapest certified point.

    Parameters
    ----------
    max_scores : (n,) highest detection score per calibration image.
    faithfulness : (n,) explanation-faithfulness score per image, in [0, 1].
    has_defect : (n,) bool, whether the image truly contains >= 1 defect.
    alpha : maximum tolerated escape rate.
    delta : family-wise error budget for the certification.
    max_false_scrap : optional cap on the auto-reject-a-clean-part rate. Applied
        as a *filter on already-certified* configurations, so it never weakens
        the escape guarantee.
    per_defective : certify escape per DEFECTIVE part (default), the event the
        whole project uses (project document, Section 2.3; part escape of
        src/risk/escape.py). False reproduces the older per-shipped-part rate,
        which is that number times the line's defect prevalence.

    Returns the selected operating point plus its calibration-set statistics.
    """
    max_scores = np.asarray(max_scores, dtype=float)
    faithfulness = np.asarray(faithfulness, dtype=float)
    has_defect = np.asarray(has_defect, dtype=bool)
    if not (len(max_scores) == len(faithfulness) == len(has_defect)):
        raise ValueError("max_scores, faithfulness and has_defect must be the same length")

    dec = decisions_for_grid(max_scores, faithfulness, configs)
    escapes = escape_indicator(dec, has_defect)
    reviews = review_indicator(dec)
    scraps = false_scrap_indicator(dec, has_defect)

    risk_rows = escapes[has_defect] if per_defective else escapes
    if risk_rows.shape[0] == 0:
        raise NoCertifiableConfig("no defective calibration images: escape is undefined")
    certified = learn_then_test(risk_rows, alpha, delta, correction="bonferroni")
    if certified.size == 0:
        best = int(np.argmin(risk_rows.mean(axis=0)))
        raise NoCertifiableConfig(
            f"no configuration certifies escape rate <= {alpha} at delta={delta} "
            f"over {len(configs)} configs and n={risk_rows.shape[0]} calibration images. "
            f"Best empirical escape rate is {risk_rows[:, best].mean():.4f}. "
            "Loosen alpha, collect more calibration data, or improve recall."
        )

    pool = certified
    if max_false_scrap is not None:
        ok = pool[scraps[:, pool].mean(axis=0) <= max_false_scrap]
        # If the scrap cap eliminates everything, keep the escape guarantee and
        # report the violation rather than silently dropping the constraint.
        if ok.size:
            pool = ok

    # Among certified configurations, minimise review load. Valid without any
    # further correction precisely because LTT certified them all at once.
    j = int(pool[np.argmin(reviews[:, pool].mean(axis=0))])

    return TriageSolution(
        config=configs[j],
        alpha=alpha,
        delta=delta,
        n_calib=len(max_scores),
        n_configs=len(configs),
        n_certified=int(certified.size),
        escape_rate_cal=float(risk_rows[:, j].mean()),
        review_load_cal=float(reviews[:, j].mean()),
        false_scrap_cal=float(scraps[:, j].mean()),
    )


def evaluate_policy(
    config: TriageConfig,
    max_scores: np.ndarray,
    faithfulness: np.ndarray,
    has_defect: np.ndarray,
) -> dict:
    """Apply a fitted policy to held-out data and report the operating stats."""
    dec = decisions_for_grid(np.asarray(max_scores, float),
                             np.asarray(faithfulness, float), [config])[:, 0]
    has_defect = np.asarray(has_defect, dtype=bool)
    n = len(dec)
    counts = {name: int((dec == k).sum()) for k, name in _DECISION_NAMES.items()}
    n_defect = int(has_defect.sum())
    return {
        "n": n,
        "counts": counts,
        "escape_rate": float(((dec == ACCEPT) & has_defect).sum() / max(n_defect, 1)),
        "escape_rate_per_image": float(((dec == ACCEPT) & has_defect).mean()),
        "review_load": float((dec == REVIEW).mean()),
        "false_scrap_rate": float(((dec == REJECT) & ~has_defect).sum()
                                  / max(int((~has_defect).sum()), 1)),
        "auto_decision_rate": float((dec != REVIEW).mean()),
    }
