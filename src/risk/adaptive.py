"""
adaptive.py
===========
Risk control when the production line drifts.

Split-conformal guarantees (`conformal.py`) rest on exchangeability between the
calibration block and the test point. A real inspection line violates that
constantly: a new steel coil, a lamp that dims, a re-tooled press, a seasonal
change in raw material. Once the score distribution moves, a threshold
calibrated last month is no longer certified, and - worse - nothing in the
static procedure tells you it has stopped being certified.

Two mechanisms here, addressing the open problem stated as future work in
Conformal Segmentation in Industrial Surface Defect Detection (arXiv:2504.17721,
2025): "dynamic calibration mechanisms to address time-varying distribution
shifts in production line data".

1. `AdaptiveRiskController` - online threshold adaptation in the style of
   Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift"
   (NeurIPS 2021), transposed from coverage to *risk*. The update is

       lambda_{t+1} = clip( lambda_t + gamma * (alpha - L_t) )

   where L_t is the realised escape loss on the part just inspected. If parts
   start escaping (L_t > alpha) the threshold falls and the detector becomes
   more permissive; if nothing escapes it rises again and review load drops.

   The guarantee is different in kind from split conformal and worth stating
   precisely, because it is easy to overclaim. It is a *long-run* bound on the
   realised average loss, and it holds for an ARBITRARY - even adversarial -
   sequence, with no exchangeability assumption at all:

       | (1/T) sum_t L_t  -  alpha |  <=  (lambda_max - lambda_min + gamma) / (gamma * T)

   So the time-average risk converges to alpha at rate O(1/T). What it does NOT
   give is a guarantee for any individual part, which split CRC does give under
   exchangeability. The two are complements: CRC for the certified static
   operating point, ACI to keep it honest as the line moves.

2. `DriftMonitor` - a two-sample test on the score distribution that raises an
   alarm before the escape rate degrades, so recalibration can be triggered on
   evidence rather than on a fixed schedule.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Online adaptive risk control
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveRiskController:
    """Online threshold adaptation with a long-run risk guarantee.

    Parameters
    ----------
    alpha : target long-run risk (e.g. 0.05 escape rate).
    gamma : step size. Larger adapts faster to abrupt drift but leaves more
        residual oscillation; the O(1/(gamma*T)) bound tightens with larger
        gamma while the variance of lambda_t grows with it, so this is a genuine
        bias/variance trade-off rather than a free knob.
    lambda_init, lambda_min, lambda_max : threshold and its clip range.
    """

    alpha: float
    gamma: float = 0.02
    lambda_init: float = 0.5
    lambda_min: float = 0.0
    lambda_max: float = 1.0

    lam: float = field(init=False)
    history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        self.lam = float(np.clip(self.lambda_init, self.lambda_min, self.lambda_max))

    def update(self, realized_loss: float) -> float:
        """Feed back the loss observed at the current threshold, get the next one."""
        loss = float(realized_loss)
        self.history.append({"lambda": self.lam, "loss": loss})
        self.lam = float(np.clip(self.lam + self.gamma * (self.alpha - loss),
                                 self.lambda_min, self.lambda_max))
        return self.lam

    # -- diagnostics ------------------------------------------------------
    @property
    def realized_risk(self) -> float:
        return float(np.mean([h["loss"] for h in self.history])) if self.history else 0.0

    def regret_bound(self) -> float:
        """The theoretical bound on |time-average risk - alpha| after T steps."""
        T = max(len(self.history), 1)
        return (self.lambda_max - self.lambda_min + self.gamma) / (self.gamma * T)

    def summary(self) -> dict:
        return {
            "steps": len(self.history),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda_final": self.lam,
            "realized_risk": self.realized_risk,
            "abs_gap": abs(self.realized_risk - self.alpha),
            "regret_bound": self.regret_bound(),
            "within_bound": abs(self.realized_risk - self.alpha) <= self.regret_bound(),
        }


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

@dataclass
class DriftMonitor:
    """Sliding-window two-sample test against the calibration score distribution.

    Operates on detector scores rather than on the realised loss, which is the
    point: scores are available on every part immediately, whereas the loss
    needs ground truth that on a real line only arrives later (or never). This
    is what lets the alarm fire *before* escapes accumulate.

    The defaults for `p_threshold` and `persistence` are set from measured
    failure modes, not convention. Testing a shift-free stream of 1500 parts
    against a 500-sample reference produced a sustained false alarm ~100 parts
    *before* the injected shift, from two compounding causes:

      * The reference is one finite sample and can simply be atypical. The draw
        used in testing sat 2.6 sigma low, which biases every subsequent
        comparison toward rejection. No amount of persistence fixes a reference
        that is off-centre, so the threshold has to carry the slack: at
        p < 0.001 that run produced zero false alarms (its smallest pre-shift
        p-value was 2.9e-3), where p < 0.01 produced 26.

      * Consecutive windows overlap in 199 of 200 samples, so they are nowhere
        near independent tests. The longest unbroken run of pre-shift
        rejections was 18, which defeats any short persistence requirement.
        Hence persistence defaults to a quarter of the window rather than a
        small constant - it has to span enough fresh samples to be a genuinely
        new test.

    Together these cost some detection latency, which is the right trade on an
    inspection line: an alarm that fires spuriously trains operators to ignore
    it, and a monitor that is ignored is worth nothing.
    """

    reference_scores: np.ndarray
    window: int = 200
    p_threshold: float = 0.001
    persistence: int = None   # defaults to window // 4

    buffer: list = field(default_factory=list, init=False)
    alarms: list = field(default_factory=list, init=False)
    _t: int = field(default=0, init=False)
    _streak: int = field(default=0, init=False)
    _fired: bool = field(default=False, init=False)

    def __post_init__(self):
        self.reference_scores = np.asarray(self.reference_scores, dtype=float).reshape(-1)
        if self.reference_scores.size < 20:
            raise ValueError("need at least 20 reference scores for a meaningful test")
        if self.persistence is None:
            self.persistence = max(10, self.window // 4)

    def push(self, score: float) -> dict | None:
        """Add one part's score; returns an alarm record when drift is detected."""
        from scipy.stats import ks_2samp

        self._t += 1
        self.buffer.append(float(score))
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        if len(self.buffer) < self.window:
            return None

        stat, p = ks_2samp(self.reference_scores, np.asarray(self.buffer))
        if p < self.p_threshold:
            self._streak += 1
        else:
            self._streak = 0
            return None

        if self._streak < self.persistence:
            return None

        # Latch: reset the streak so a single drift event produces one alarm
        # rather than one per part for the rest of the run. Without this the
        # caller sees hundreds of duplicate alarms for the same event and any
        # alarm-triggered action fires repeatedly.
        self._streak = 0
        rec = {"t": self._t, "ks_stat": float(stat), "p_value": float(p),
               "window_mean": float(np.mean(self.buffer)),
               "reference_mean": float(self.reference_scores.mean())}
        self.alarms.append(rec)
        return rec

    def summary(self) -> dict:
        return {"steps": self._t, "n_alarms": len(self.alarms),
                "persistence": self.persistence,
                "first_alarm_t": self.alarms[0]["t"] if self.alarms else None}


# ---------------------------------------------------------------------------
# Simulation harness (used by the drift experiment and the paper figure)
# ---------------------------------------------------------------------------

def run_adaptive_experiment(loss_at_lambda, n_steps, alpha, gamma=0.02,
                            lambda_grid=None, seed=0):
    """Drive an AdaptiveRiskController through a (possibly drifting) stream.

    `loss_at_lambda(t, lam, rng) -> float` supplies the realised loss for the
    part at time t when the threshold is lam, and is where a drift schedule is
    injected.
    """
    rng = np.random.default_rng(seed)
    lambda_grid = lambda_grid if lambda_grid is not None else (0.0, 1.0)
    ctrl = AdaptiveRiskController(alpha=alpha, gamma=gamma,
                                  lambda_min=lambda_grid[0], lambda_max=lambda_grid[1])
    lams, losses = [], []
    for t in range(n_steps):
        lam = ctrl.lam
        L = loss_at_lambda(t, lam, rng)
        lams.append(lam)
        losses.append(L)
        ctrl.update(L)
    return {"lambdas": np.asarray(lams), "losses": np.asarray(losses),
            "controller": ctrl, "summary": ctrl.summary()}


def simulated_line_loss(quality_schedule):
    """Build an escape-loss function for a drifting inspection line.

    `quality_schedule(t) -> q in (0, 1]` is the detector's effective quality at
    time t, where 1.0 means "as trained". Scores for true defects are drawn from
    a Beta whose mass slides toward zero as q falls, which is what a drifting
    line looks like from the detector's side: defects get harder to see, scores
    sag, and a frozen threshold starts letting them through.

    Lives here rather than in the test file so the drift experiment, the paper
    figure and the notebook all exercise the same generator.
    """
    def loss_at_lambda(t, lam, rng):
        q = float(quality_schedule(t))
        n_def = 1 + rng.poisson(1.5)
        scores = rng.beta(6.0 * q, 3.0, size=n_def) * q
        return float((scores < lam).mean())
    return loss_at_lambda
