# The acquisition frontier: four corrections, four failures, one honest recommendation

KolektorSDD2, alpha = 0.10, budget 100 labels, 300 replicates each.

## The setting

Escape loss is defined only on defective parts, so the certificate is bought with
labelled **defects** and the penalty is 1/(n_def + 1). On a line that is 89% clean
a random budget buys almost nothing: 10.3 defects, and a certificate **0% of the
time**. Screening on the detector score buys 50.3 defects and certifies **82.7%**
of the time - but selects on the quantity being calibrated, so it is biased
(realised 0.1011 against 0.10).

Four ways to remove that bias were implemented and measured.

## 1. Inverse-propensity weighting - REFUTED

| budget | weight ratio w_max/w_min | issued |
|---:|---:|---:|
| 50 | 1,953 | 0% |
| 100 | **2,103,520** | 0% |

Weighted CRC charges the unobserved test point at `w_max`. Under proportional
screening the propensity of the rarest part is millions of times below the most
likely one, `w_max` dominates both sums, and no threshold is ever certified.
Clipping enough to issue reintroduces the bias the weights existed to remove
(0.1204 at a 5x clip).

## 2. Stratified sampling with design weights - REFUTED

Fixed the weight explosion exactly as intended: ratio 1.00-2.78 instead of
millions. But bounding the weights means spreading the budget across strata, and
the defect-free strata consume labels that buy nothing.

Sweep over k in {2,3,5}, tilt in {2,4,8}:

| design | defects | weight ratio | issued | risk |
|---|---:|---:|---:|---:|
| k=3, tilt=2 | 16.9 | 1.00 | 0.5% | 0.1011 |
| **k=3, tilt=4 (best valid)** | 23.2 | 1.00 | **6.0%** | 0.1049 |
| k=5, tilt=8 | 41.9 | 2.78 | 25.0% | 0.1155 (biased) |

The best *valid* stratified design issues 6% of the time. The naive screen issues
82.7% at a **lower** realised risk (0.1011 vs 0.1049). The correction is worse on
both axes at once.

## 3. Deflated alpha - REFUTED, and it explains the others

If the bias is a known ~1% relative overshoot, calibrate at a tighter alpha' and
let the bias absorb it. It does not work:

| alpha' | issued | test risk |
|---:|---:|---:|
| 0.100 | 82.7% | 0.1011 |
| 0.095 | **0.0%** | - |
| 0.090 | 0.0% | - |

A cliff, not a slope. With ~50 defective calibration points the achievable
operating grid is the order statistics themselves - steps of roughly 1/51 ~ 0.02.
There is no threshold between alpha = 0.10 and alpha = 0.095, so shaving the
budget by 0.005 does not tighten the operating point, it removes the only one
that existed.

**This is the mechanism behind all three failures.** At n_def ~ 50 the operating
grid is coarse. Every correction is an attempt to make a fine adjustment on a
grid whose spacing is larger than the adjustment.

## 4. What is left

The naive screen: 82.7% issuance, realised 0.1011 against a 0.10 budget - a 1.1%
relative overshoot, systematic and in one direction.

The honest recommendation is to **report it, not correct it**. Every correction
measured here costs more than the bias it removes, and the reason is structural
rather than a tuning failure: the grid is too coarse for the size of the
adjustment. A practitioner is better served by "this operating point overshoots
its budget by about 1%, here is the measurement" than by a machine that refuses
to issue any operating point at all.

## Why this is a result rather than a dead end

It maps a frontier that nobody has mapped for detection risk control, with a
mechanism for each failure:

* acquisition efficiency is real and large (0% -> 83% issuance on the same spend);
* the induced bias is real and small (1.1% relative);
* three standard corrections all fail, for one shared and identifiable reason;
* the binding constraint is the **discreteness of the operating grid at small
  n_def**, which is exactly the cold-start regime this project set out to study.

That last point is the transferable finding: at small calibration sizes,
conformal operating points are quantised, and any method that assumes a
continuous alpha knob is making an assumption the data cannot support.
