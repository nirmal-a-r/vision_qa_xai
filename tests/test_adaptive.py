"""
Validation of drift-aware risk control.

The claim being tested is specifically the long-run one: under an arbitrary
drifting sequence the *time-average* realised risk converges to alpha. A static
split-conformal threshold has no such property once exchangeability breaks, and
the contrast between the two under identical drift is the experiment that
justifies the adaptive layer existing at all.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.adaptive import (
    AdaptiveRiskController,
    DriftMonitor,
    run_adaptive_experiment,
    simulated_line_loss as make_loss_fn,
)


def test_adaptive_converges_no_drift():
    """With a stationary stream the time-average risk should sit at alpha."""
    alpha = 0.10
    res = run_adaptive_experiment(make_loss_fn(lambda t: 1.0),
                                  n_steps=4000, alpha=alpha, gamma=0.05, seed=1)
    s = res["summary"]
    print(f"  stationary: realized={s['realized_risk']:.4f} alpha={alpha} "
          f"gap={s['abs_gap']:.4f} bound={s['regret_bound']:.4f}")
    assert s["abs_gap"] < 0.02, f"did not converge: gap {s['abs_gap']:.4f}"


def test_adaptive_tracks_gradual_drift():
    """Under a slow degradation the adaptive threshold must hold risk at alpha."""
    alpha = 0.10
    # detector quality decays from 1.0 to 0.6 over the run
    res = run_adaptive_experiment(make_loss_fn(lambda t: 1.0 - 0.4 * min(t / 3000, 1.0)),
                                  n_steps=4000, alpha=alpha, gamma=0.05, seed=2)
    s = res["summary"]
    print(f"  gradual drift: realized={s['realized_risk']:.4f} alpha={alpha} "
          f"gap={s['abs_gap']:.4f}  lambda {res['lambdas'][0]:.3f} -> {res['lambdas'][-1]:.3f}")
    assert s["abs_gap"] < 0.03, f"failed to track drift: gap {s['abs_gap']:.4f}"
    # the threshold must actually have moved down to compensate
    assert res["lambdas"][-1] < res["lambdas"][0], "threshold did not adapt downward"


def test_adaptive_survives_abrupt_shift():
    """A sudden regime change (new coil / new tooling) must be absorbed."""
    alpha = 0.10
    res = run_adaptive_experiment(make_loss_fn(lambda t: 1.0 if t < 2000 else 0.55),
                                  n_steps=5000, alpha=alpha, gamma=0.05, seed=3)
    s = res["summary"]
    post = res["losses"][2500:]      # after the controller has re-settled
    print(f"  abrupt shift: overall={s['realized_risk']:.4f} "
          f"post-shift={post.mean():.4f} alpha={alpha}")
    assert s["abs_gap"] < 0.03, f"overall risk off target: {s['abs_gap']:.4f}"
    assert abs(post.mean() - alpha) < 0.04, f"did not re-settle: {post.mean():.4f}"


def test_static_threshold_fails_under_drift():
    """The motivating negative result: a fixed threshold silently degrades.

    Without this contrast the adaptive layer has no justification, so it is
    tested as explicitly as the positive claims.
    """
    alpha = 0.10
    loss_fn = make_loss_fn(lambda t: 1.0 if t < 2000 else 0.55)

    # calibrate a static threshold on the pre-shift regime, then freeze it
    rng = np.random.default_rng(4)
    grid = np.linspace(1.0, 0.0, 101)
    cal = np.array([[loss_fn(t, lam, rng) for lam in grid] for t in range(400)])
    mean_loss = cal.mean(axis=0)
    static_lam = float(grid[np.flatnonzero(
        (400 / 401) * mean_loss + 1 / 401 <= alpha)[0]])

    rng2 = np.random.default_rng(5)
    post = np.array([loss_fn(t, static_lam, rng2) for t in range(2000, 5000)])
    adaptive = run_adaptive_experiment(loss_fn, 5000, alpha, gamma=0.05, seed=3)
    adaptive_post = adaptive["losses"][2500:]

    print(f"  static lambda={static_lam:.3f} -> post-shift risk {post.mean():.4f} "
          f"({post.mean()/alpha:.1f}x alpha)")
    print(f"  adaptive              -> post-shift risk {adaptive_post.mean():.4f}")
    assert post.mean() > alpha * 1.5, (
        "static threshold did not degrade, so the drift scenario is too weak "
        "to motivate the adaptive layer")
    assert adaptive_post.mean() < post.mean(), "adaptive was no better than static"


def test_regret_bound_holds():
    """The realised gap must sit inside the theoretical O(1/(gamma*T)) bound."""
    for gamma in (0.01, 0.05, 0.1):
        res = run_adaptive_experiment(make_loss_fn(lambda t: 1.0),
                                      n_steps=3000, alpha=0.10, gamma=gamma, seed=7)
        s = res["summary"]
        print(f"  gamma={gamma}: gap={s['abs_gap']:.5f} bound={s['regret_bound']:.5f} "
              f"{'OK' if s['within_bound'] else 'VIOLATED'}")
        assert s["within_bound"], f"regret bound violated at gamma={gamma}"


def test_drift_monitor_fires_on_shift_and_not_before():
    """The monitor must alarm after a real shift and stay quiet on a stable line."""
    rng = np.random.default_rng(11)
    ref = rng.beta(6, 3, size=500)

    quiet = DriftMonitor(ref, window=200, p_threshold=0.01)
    fired_quiet = sum(1 for _ in range(1500) if quiet.push(rng.beta(6, 3)) is not None)

    shifting = DriftMonitor(ref, window=200, p_threshold=0.01)
    first = None
    for t in range(1500):
        s = rng.beta(6, 3) if t < 700 else rng.beta(3, 4)   # regime change at 700
        rec = shifting.push(s)
        if rec and first is None:
            first = rec["t"]

    print(f"  stable line: {fired_quiet} alarms in 1500 parts")
    print(f"  shifted line: first alarm at t={first} (shift at t=700)")
    assert fired_quiet <= 30, f"too many false alarms on a stable line: {fired_quiet}"
    assert first is not None, "missed a clear distribution shift"
    assert 700 <= first <= 1000, f"alarm at t={first} not promptly after the shift"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'=' * 60}\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
