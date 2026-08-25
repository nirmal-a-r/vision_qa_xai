"""
inspector.py
============
The closed-loop inspection agent.

This is deliberately a control loop, not a language-model wrapper. Every decision
it makes is one a statistical certificate can license or refuse, and an LLM in
that position would supply confident prose with no guarantee attached - the
opposite of what this paper argues an inspection system should do. "Agentic"
here means the loop perceives, maintains state, decides, monitors itself, and
chooses corrective actions autonomously, under a risk budget it cannot exceed.

The loop, per part:

    perceive   -> detector scores + explanation faithfulness
    decide     -> AUTO_ACCEPT / HUMAN_REVIEW / AUTO_REJECT under the current
                  certified policy
    monitor    -> feed the score to the drift detector; feed realised loss
                  (when a reviewed part yields ground truth) to the adaptive
                  controller
    act        -> on drift: recalibrate if enough fresh labels exist, else
                  escalate to full review; and queue the most informative parts
                  for labelling

The part that is new is the acquisition rule. Standard active learning buys
labels that most reduce *model* uncertainty. Here labels are bought for whatever
most tightens the *certificate*, which is a different objective. The certified
bound is

    (n/(n+1)) * R_hat_n(lambda) + B/(n+1)

so its slack has two distinct sources: the finite-sample penalty B/(n+1), which
only shrinks with more labelled *defective* parts, and the empirical risk term,
which sharpens with labels near the decision boundary. A clean part carries
almost no certificate value however uncertain the model is about it - which is
exactly the case standard uncertainty sampling gets wrong here, since
low-confidence clean parts look maximally informative to it.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..risk.conformal import (
    conformal_risk_control,
    escape_threshold_grid,
    build_calibration_losses,
    escape_loss_curve,
    RiskNotAchievable,
)
from ..risk.adaptive import AdaptiveRiskController, DriftMonitor
from ..risk.triage import TriageConfig, ACCEPT, REVIEW, REJECT


ESCALATED = "escalated_full_review"


@dataclass
class AgentState:
    lam: float
    certified: bool
    escalated: bool = False
    n_parts: int = 0
    n_accept: int = 0
    n_review: int = 0
    n_reject: int = 0
    n_escapes: int = 0
    n_defective_seen: int = 0
    recalibrations: int = 0
    drift_alarms: int = 0
    labels_requested: int = 0

    def as_dict(self):
        d = dict(self.__dict__)
        n = max(self.n_parts, 1)
        d["review_load"] = self.n_review / n
        d["auto_rate"] = (self.n_accept + self.n_reject) / n
        d["escape_rate"] = self.n_escapes / max(self.n_defective_seen, 1)
        return d


class InspectionAgent:
    """Autonomous risk-controlled inspector with self-monitoring and label acquisition."""

    def __init__(self, cal_records, alpha, grid=None, gamma=0.02,
                 faithfulness_floor=0.0, label_budget_per_alarm=50,
                 iou_thresh=0.5, drift_window=200, max_false_scrap=0.05):
        self.alpha = alpha
        # Upper threshold budget. Without it the policy has ONE threshold, the
        # review band is empty by construction, and every part that is not
        # certifiably clean is scrapped - on KolektorSDD2 that meant rejecting
        # 834 of 834 parts, i.e. scrapping all of production. A second
        # threshold is what separates "not provably clean" from "confidently
        # defective"; only the gap between them is what a human ever sees.
        self.max_false_scrap = max_false_scrap
        self.lam_hi = 1.0
        self.grid = grid if grid is not None else escape_threshold_grid(201)
        self.iou_thresh = iou_thresh
        self.phi = faithfulness_floor
        self.label_budget = label_budget_per_alarm

        self.cal_records = list(cal_records)
        self.state = AgentState(lam=1.0, certified=False)
        self._calibrate()

        ref = self._cal_scores()
        self.drift = DriftMonitor(ref, window=drift_window) if len(ref) >= 20 else None
        self.adaptive = AdaptiveRiskController(alpha=alpha, gamma=gamma,
                                               lambda_init=self.state.lam)
        self.label_queue = []
        self.log = []

    # -- calibration ------------------------------------------------------
    def _cal_scores(self):
        # Must use exactly the rule `step` uses, including the 0.0 for parts
        # with no detection at all. Skipping those here instead (the natural
        # way to write it) silently compares two different quantities: the
        # reference then has no mass at zero while the live buffer does, and on
        # KolektorSDD2 - where ~89% of parts are clean and often produce no
        # detection - that fabricated mismatch fired the drift alarm on 586 of
        # 834 exchangeable parts. The distributions actually agree
        # (KS p = 0.48 over the full split).
        return np.asarray([self._max_score(r) for r in self.cal_records], dtype=float)

    @staticmethod
    def _max_score(record):
        s = np.asarray(record.get("pred_scores", []), dtype=float)
        return float(s.max()) if s.size else 0.0

    def _calibrate(self):
        try:
            losses, _, _ = build_calibration_losses(self.cal_records, self.grid,
                                                    self.iou_thresh)
            self.state.lam = conformal_risk_control(losses, self.grid, self.alpha)
            self.state.certified = True
            self.state.escalated = False
        except (RiskNotAchievable, ValueError):
            # No certifiable operating point: escalate rather than guess. This
            # is the branch that makes the agent safe - it is allowed to say
            # "I cannot guarantee this" and hand the line back to people.
            self.state.certified = False
            self.state.escalated = True
            self.state.lam = 0.0
        self._calibrate_upper()
        self.state.recalibrations += 1
        return self.state.lam

    def _calibrate_upper(self):
        """Lowest threshold at which auto-rejecting is still precise enough.

        Scanning downwards, keep the smallest t for which the parts scoring >= t
        on the calibration set are defective at least (1 - max_false_scrap) of
        the time. Lower t means more parts auto-rejected and fewer sent to a
        human, so taking the smallest admissible t minimises review load subject
        to the scrap budget - the same objective the offline triage policy
        optimises, computed online from the same held-out block.

        Falls back to 1.0 (auto-reject disabled, everything above lam goes to
        review) when no threshold is precise enough. That is the safe direction:
        it costs inspector time, never a wrongly scrapped part.
        """
        scores = self._cal_scores()
        defect = np.asarray([bool(r.get("gt_boxes")) for r in self.cal_records])
        if scores.size == 0 or not defect.any():
            self.lam_hi = 1.0
            return self.lam_hi
        best = 1.0
        for t in np.unique(np.round(scores, 4))[::-1]:
            sel = scores >= t
            if sel.sum() < 5:                      # too few to estimate precision
                continue
            false_scrap = float((~defect[sel]).mean())
            if false_scrap <= self.max_false_scrap:
                best = float(t)
            else:
                break                              # precision only degrades further down
        self.lam_hi = max(best, self.state.lam)    # never invert the two thresholds
        return self.lam_hi

    # -- per-part loop ----------------------------------------------------
    def step(self, record, faithfulness=None, ground_truth_available=False):
        st = self.state
        st.n_parts += 1
        smax = self._max_score(record)
        faith = 1.0 if faithfulness is None else float(faithfulness)
        has_defect = bool(record.get("gt_boxes"))
        if has_defect:
            st.n_defective_seen += 1

        # ---- decide ----
        if st.escalated:
            decision = REVIEW
        elif smax < st.lam:
            decision = ACCEPT                      # certified clean
        elif smax >= self.lam_hi and faith >= self.phi:
            decision = REJECT                      # confidently defective AND explainably so
        else:
            decision = REVIEW                      # the band a human actually sees

        if decision == ACCEPT:
            st.n_accept += 1
            if has_defect:
                st.n_escapes += 1
        elif decision == REJECT:
            st.n_reject += 1
        else:
            st.n_review += 1

        # ---- monitor ----
        alarm = self.drift.push(smax) if self.drift is not None else None
        if alarm:
            st.drift_alarms += 1

        # A reviewed part yields ground truth for free - an inspector looked at
        # it. That is the only loss signal available online, and it is exactly
        # the signal the adaptive controller needs.
        if ground_truth_available or decision == REVIEW:
            L = float(escape_loss_curve(record.get("gt_boxes", []),
                                        record.get("pred_boxes", []),
                                        record.get("pred_scores", []),
                                        np.array([st.lam]), self.iou_thresh)[0]) \
                if has_defect else 0.0
            new_lam = self.adaptive.update(L)
            if not st.escalated:
                st.lam = new_lam

        # ---- act ----
        if alarm:
            self._on_drift(record)

        self.log.append({"t": st.n_parts, "lambda": st.lam, "lambda_hi": self.lam_hi,
                         "decision": decision,
                         "score": smax, "faith": faith, "defect": has_defect,
                         "alarm": bool(alarm), "escalated": st.escalated})
        return decision

    def _on_drift(self, record):
        """Drift detected: buy the labels that most tighten the certificate."""
        self.state.escalated = True          # safe until re-certified
        self.state.labels_requested += self.label_budget

    # -- risk-aware active learning ---------------------------------------
    def acquisition_scores(self, candidates, faithfulness=None):
        """Value of labelling each candidate, in units of certificate slack.

        Three terms, each tied to a term of the bound rather than to model
        uncertainty in the abstract:

        1. `defect_prior` - only parts that turn out defective enter the
           calibration set at all, so they alone shrink the B/(n+1) penalty.
           Detector score is the available proxy for that probability.
        2. `boundary` - parts whose score sits near the current threshold are
           the ones whose loss actually moves R_hat, so they sharpen the
           empirical term.
        3. `explanation_doubt` - low faithfulness marks parts the model handles
           for reasons that do not survive audit; those are where the score is
           least trustworthy and a human label is worth most.
        """
        lam = self.state.lam
        out = []
        for i, r in enumerate(candidates):
            s = np.asarray(r.get("pred_scores", []), dtype=float)
            smax = float(s.max()) if s.size else 0.0
            defect_prior = smax
            boundary = float(np.exp(-((smax - lam) ** 2) / (2 * 0.10 ** 2)))
            doubt = 1.0 - (1.0 if faithfulness is None else float(faithfulness[i]))
            out.append(0.5 * defect_prior + 0.35 * boundary + 0.15 * doubt)
        return np.asarray(out)

    def select_for_labeling(self, candidates, budget, faithfulness=None):
        sc = self.acquisition_scores(candidates, faithfulness)
        return list(np.argsort(-sc)[:budget])

    def add_labels(self, new_records):
        """Fold freshly labelled parts into calibration and re-certify."""
        self.cal_records.extend(new_records)
        lam = self._calibrate()
        if self.state.certified:
            self.adaptive.lam = lam
            if self.drift is not None:
                self.drift = DriftMonitor(self._cal_scores(),
                                          window=self.drift.window)
        return lam

    def summary(self):
        d = self.state.as_dict()
        d["alpha"] = self.alpha
        d["adaptive"] = self.adaptive.summary()
        return d


# ---------------------------------------------------------------------------
# Episode runner used by the experiment and the notebook
# ---------------------------------------------------------------------------

def run_episode(agent, stream, faithfulness=None, label_pool=None,
                relabel_every_alarm=True):
    """Drive the agent over a stream of parts, optionally with a label pool.

    Returns the agent summary plus the per-part log.
    """
    for i, rec in enumerate(stream):
        f = None if faithfulness is None else faithfulness[i]
        agent.step(rec, faithfulness=f)
        if (relabel_every_alarm and agent.state.escalated
                and label_pool and agent.state.labels_requested > 0):
            k = min(agent.label_budget, len(label_pool))
            idx = agent.select_for_labeling(label_pool, k)
            agent.add_labels([label_pool[j] for j in idx])
            for j in sorted(idx, reverse=True):
                label_pool.pop(j)
            agent.state.labels_requested = 0
    return agent.summary(), agent.log
