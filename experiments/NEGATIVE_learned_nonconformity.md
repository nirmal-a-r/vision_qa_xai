# Negative result: a learned nonconformity score does not beat raw confidence

## Motivation

Every prior conformal-detection work (Shen & Liu 2025; SeqCRC 2025) thresholds the
**raw maximum detection confidence** as its nonconformity score. That score is known
to be poorly calibrated, so the obvious contribution is to learn a better one: fuse
confidence statistics, box geometry, detection count and detector self-agreement,
then run conformal risk control on the fused score. Better ranking of escape risk
should buy a higher auto-accept rate at the same certified escape budget - which is
the quantity a plant actually pays for.

## Protocol

Calibration block split in half: the score is fitted on half A, the conformal
threshold is calibrated on half B, and the test block is untouched. Splitting is
mandatory - fitting and calibrating on the same block double-dips and voids the
guarantee. (The first version of this experiment made exactly that mistake; the
numbers below are from the corrected protocol.)

Features: max score, mean score, runner-up score, detection count, largest box area
fraction, mean pairwise IoU between detections, score std.

## Result 1 - ranking does improve

AUC for predicting per-image escape, fit on cal, evaluated on test:

| dataset | AUC max-score | AUC multi-signal | gain |
|---|---:|---:|---:|
| neu | 0.6497 | 0.7065 | +0.057 |
| gc10 | 0.7249 | 0.7714 | +0.047 |
| magnetic_tile | 0.7337 | 0.7663 | +0.033 |
| kolektor | 0.8758 | 0.8930 | +0.017 |
| pcb | 0.7874 | 0.7716 | -0.016 |

## Result 2 - it does NOT convert into less human review

Auto-accept rate at a certified escape budget of alpha = 0.10:

| dataset | max-score | learned | verdict |
|---|---:|---:|---|
| neu | 14.9% | 11.3% | max-score wins |
| gc10 | 14.9% | 17.0% | learned wins |
| pcb | 21.8% | 16.7% | max-score wins |
| kolektor | **88.1%** | **0.7%** | max-score wins decisively |

## Why

AUC measures ranking; the certified operating point depends on the score's
behaviour in the *lower tail*, which is a different property. Raw confidence has a
structure the learned score destroys: on an imbalanced line most clean parts sit at
essentially zero confidence, so a low threshold clears a large clean majority while
risking very few defects. KolektorSDD2 is 89% clean and this is worth 88% auto-accept.
A logistic score fitted with balanced class weights spreads that mass across the
range and the separation is gone.

Splitting the calibration block also doubles the finite-sample penalty B/(n+1),
a cost the learned score must overcome before it even breaks even.

## Conclusion

**Raw maximum confidence is already close to optimal as a nonconformity score for
escape control on imbalanced industrial data.** Recombining the detector's own
outputs does not beat it. Any improvement has to come from information the detector
does not already encode - test-time-augmentation consistency, feature-space novelty
relative to the training distribution, or explanation faithfulness - none of which
is tested here.

This result is reported rather than discarded because it removes an obvious
reviewer objection ("why not learn the score?") with evidence.
