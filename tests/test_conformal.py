"""
Validation of the risk-control machinery.

These are correctness tests, not demos: each one checks a property the theory
promises, on synthetic data where ground truth is known. If any of these fail,
every downstream number in the paper is meaningless.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk.conformal import (
    conformal_risk_control,
    escape_threshold_grid,
    learn_then_test,
    hoeffding_bentkus_pvalue,
    escape_loss_curve,
    box_iou_matrix,
    RiskNotAchievable,
    localization_robust_risk_control,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Synthetic detector: score of a matched GT ~ Beta(a,b), misses are score 0
# ---------------------------------------------------------------------------

def simulate_image(rng, n_gt_lam=2.0, detect_p=0.85, a=2.0, b=3.0):
    """One image: per-GT detection score, or -1 if the detector never fired.

    -1 rather than 0.0 matters: a defect the detector missed outright must count
    as escaped at *every* threshold, including the lowest one. Scoring it 0.0
    makes `0.0 < 0.0` false, so the lowest threshold appears to catch
    everything and the risk floor collapses to zero.
    """
    n_gt = 1 + rng.poisson(n_gt_lam)
    detected = rng.random(n_gt) < detect_p
    return np.where(detected, rng.beta(a, b, size=n_gt), -1.0)


def losses_from_scores(score_list, lambdas):
    """L_i(lambda) = fraction of that image's GT defects with score < lambda."""
    out = np.empty((len(score_list), len(lambdas)))
    for i, s in enumerate(score_list):
        for j, lam in enumerate(lambdas):
            out[i, j] = float((s < lam).mean())
    return out


def test_crc_guarantee_holds():
    """The headline claim: E[loss on a fresh image] <= alpha.

    Repeat the whole calibrate-then-test cycle many times and check the mean
    test risk respects alpha. This is the empirical validation of the theorem.
    """
    lambdas = escape_threshold_grid(101)
    alpha = 0.10
    n_cal, n_test, n_trials = 300, 300, 400

    test_risks = []
    for t in range(n_trials):
        rng = np.random.default_rng(1000 + t)
        # detect_p=0.97 -> risk floor ~0.03, comfortably under alpha=0.10
        cal = [simulate_image(rng, detect_p=0.97) for _ in range(n_cal)]
        tst = [simulate_image(rng, detect_p=0.97) for _ in range(n_test)]
        lam = conformal_risk_control(losses_from_scores(cal, lambdas), lambdas, alpha)
        j = int(np.flatnonzero(lambdas == lam)[0])
        test_risks.append(losses_from_scores(tst, lambdas)[:, j].mean())

    mean_risk = float(np.mean(test_risks))
    print(f"  CRC: alpha={alpha}  mean test risk={mean_risk:.4f}  "
          f"P(risk>alpha)={np.mean(np.array(test_risks) > alpha):.3f}")
    assert mean_risk <= alpha + 1e-3, f"guarantee VIOLATED: {mean_risk:.4f} > {alpha}"
    # Should not be absurdly conservative either, or the method is useless.
    assert mean_risk > alpha * 0.5, f"far too conservative: {mean_risk:.4f} << {alpha}"


def test_crc_tighter_alpha_gives_lower_threshold():
    """Monotone behaviour: demanding less risk must not raise the threshold."""
    lambdas = escape_threshold_grid(101)
    rng = np.random.default_rng(7)
    cal = [simulate_image(rng, detect_p=0.99) for _ in range(500)]
    L = losses_from_scores(cal, lambdas)
    lam_loose = conformal_risk_control(L, lambdas, 0.20)
    lam_tight = conformal_risk_control(L, lambdas, 0.05)
    print(f"  alpha=0.20 -> lambda={lam_loose:.3f}   alpha=0.05 -> lambda={lam_tight:.3f}")
    assert lam_tight <= lam_loose


def test_crc_rejects_non_monotone_losses():
    """A silently wrong loss must raise, not return a bogus guarantee."""
    lambdas = escape_threshold_grid(5)
    bad = np.tile(np.array([0.5, 0.1, 0.4, 0.2, 0.3]), (20, 1))
    try:
        conformal_risk_control(bad, lambdas, 0.2)
    except ValueError as e:
        assert "non-increasing" in str(e)
        print("  correctly rejected non-monotone loss")
        return
    raise AssertionError("accepted a non-monotone loss -- guarantee would be void")


def test_crc_raises_when_alpha_unreachable():
    """If even the loosest threshold cannot hit alpha, say so loudly."""
    lambdas = escape_threshold_grid(51)
    # Detector misses 40% of defects outright: risk can never go below 0.4.
    rng = np.random.default_rng(3)
    cal = [simulate_image(rng, detect_p=0.60) for _ in range(200)]
    try:
        conformal_risk_control(losses_from_scores(cal, lambdas), lambdas, 0.05)
    except RiskNotAchievable as e:
        print(f"  correctly refused: {str(e)[:70]}...")
        return
    raise AssertionError("claimed alpha=0.05 was achievable by a 60%-recall detector")


def test_ltt_fwer_control():
    """Learn-then-Test must rarely certify a configuration that is truly bad."""
    alpha, delta, n, n_trials = 0.10, 0.10, 400, 300
    # All 20 configs have TRUE risk just above alpha -> every certification is
    # an error. The family-wise error rate must stay <= delta.
    true_risk = 0.13
    false_certifications = 0
    for t in range(n_trials):
        rng = np.random.default_rng(5000 + t)
        losses = (rng.random((n, 20)) < true_risk).astype(float)
        if learn_then_test(losses, alpha, delta).size > 0:
            false_certifications += 1
    fwer = false_certifications / n_trials
    print(f"  LTT: true risk={true_risk} > alpha={alpha}; "
          f"empirical FWER={fwer:.3f} (budget {delta})")
    assert fwer <= delta, f"FWER {fwer:.3f} exceeds budget {delta}"


def test_ltt_has_power():
    """It must still certify configurations that are genuinely safe."""
    alpha, delta, n = 0.10, 0.10, 400
    rng = np.random.default_rng(11)
    losses = (rng.random((n, 20)) < 0.02).astype(float)   # true risk well under alpha
    certified = learn_then_test(losses, alpha, delta)
    print(f"  LTT certified {certified.size}/20 genuinely-safe configs")
    assert certified.size >= 18


def test_hb_pvalue_is_valid():
    """P(p <= u) <= u under H0 -- the defining property of a valid p-value."""
    alpha, n, n_trials = 0.10, 200, 4000
    rng = np.random.default_rng(2)
    # Under H0 the true risk equals alpha (boundary case, hardest).
    r_hat = (rng.random((n_trials, n)) < alpha).mean(axis=1)
    p = hoeffding_bentkus_pvalue(r_hat, n, alpha)
    for u in (0.01, 0.05, 0.10, 0.20):
        emp = float((p <= u).mean())
        print(f"  P(p<={u:.2f}) = {emp:.4f}")
        assert emp <= u + 0.01, f"invalid p-value: P(p<={u}) = {emp:.4f}"


def test_escape_loss_is_monotone_and_bounded():
    """The detection loss must satisfy CRC's preconditions on real box geometry."""
    rng = np.random.default_rng(4)
    lambdas = escape_threshold_grid(51)
    for _ in range(200):
        n_gt, n_pred = rng.integers(1, 5), rng.integers(0, 8)
        gt = rng.random((n_gt, 2)) * 100
        gt = np.hstack([gt, gt + 10 + rng.random((n_gt, 2)) * 40])
        pr = rng.random((n_pred, 2)) * 100
        pr = np.hstack([pr, pr + 10 + rng.random((n_pred, 2)) * 40])
        sc = rng.random(n_pred)
        L = escape_loss_curve(gt, pr, sc, lambdas)
        assert L.min() >= 0.0 and L.max() <= 1.0, "loss left [0,1]"
        assert np.all(np.diff(L) <= 1e-9), "loss not non-increasing along descending grid"
    print("  escape loss bounded in [0,1] and monotone over 200 random layouts")


def test_iou_matches_hand_computation():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 5.0, 15.0, 15.0],     # inter 25, union 175 -> 1/7
                  [0.0, 0.0, 10.0, 10.0],     # identical -> 1
                  [20.0, 20.0, 30.0, 30.0]])  # disjoint -> 0
    got = box_iou_matrix(a, b)[0]
    exp = np.array([25 / 175, 1.0, 0.0])
    assert np.allclose(got, exp), f"{got} != {exp}"
    print("  IoU matches hand computation")


def test_localization_robust_threshold_is_most_conservative():
    """The multi-IoU deployment threshold must be <= every individual one."""
    grid = escape_threshold_grid(51)
    # First defect is only a loose-localisation hit; second is a strict hit.
    records = []
    for i in range(120):
        if i % 3:
            records.append({"gt_boxes": [[0, 0, 10, 10]],
                            "pred_boxes": [[0, 0, 10, 10]], "pred_scores": [.4]})
        else:
            records.append({"gt_boxes": [[0, 0, 10, 10]],
                            "pred_boxes": [[2, 2, 12, 12]], "pred_scores": [.4]})
    res = localization_robust_risk_control(records, grid, .40, (0.30, 0.50))
    assert res["issued"], res
    individual = [v["threshold"] for v in res["per_iou"].values()]
    assert res["threshold"] == min(individual)
    print("  multi-IoU threshold is the pointwise-safe, most conservative threshold")


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
    print(f"\n{'='*60}\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
