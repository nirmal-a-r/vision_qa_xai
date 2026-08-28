# Validated: the reduction, and how much headroom CRC leaves on the table

Two results here. The first is solid and reusable. The second quantifies the
opportunity that motivates the whole cold-start direction. The third section
records what I could NOT get working, so nobody repeats it.

## 1. The reduction (validated)

For a defective image i, define its **escape confidence**

    c_i = min over ground-truth boxes g of  max{ score of a detection matching g }

with c_i = -inf when some ground-truth box is never matched. Then

    L_i(lambda) = 1{ lambda > c_i }        and       R_hat_n(lambda) = F_hat_n(lambda)

so conformal risk control on the image-level escape loss is exactly a **quantile
of the c_i**. The loss curve collapses to one scalar per image.

This matters because it converts a detection risk-control problem into the
scalar-score setting that the conformal literature (including SPI) is written
for - no re-derivation needed.

**Verified empirically.** CRC applied to the reduced score, 600 trials, n=60
calibration, Beta(2,3) scores:

| alpha | mean test escape | holds |
|---|---:|---|
| 0.05 | 0.0322 | yes |
| 0.10 | 0.0818 | yes |
| 0.20 | 0.1794 | yes |

## 2. How much CRC wastes at small n (the opportunity)

With m = 25 real defective calibration images:

| alpha | CRC threshold | CRC realised risk | budget used |
|---|---:|---:|---:|
| 0.05 | **refuses** (flags everything) | 0.0000 | 0% |
| 0.10 | 0.077 | 0.0403 | 40% |
| 0.20 | 0.178 | 0.1540 | 77% |

At alpha = 0.05 the method **cannot issue an operating point at all** from 25
examples. At alpha = 0.10 it spends less than half its allowed budget, because
the finite-sample term 1/(m+1) = 0.038 and the coarse quantile grid (steps of
1/26) force it to round down hard.

That unused budget is the prize. It is not detector error and no amount of
training recovers it - it is the price of having few labelled defects. Kolektor
has 53; a newly commissioned line has fewer.

## 3. What did NOT work: reconstructing SPI from the paper description

Two reconstructions were built and tested (400 trials each, m=25, N=500).

**Attempt A - quantile at synthetic resolution, rank-window clipped**

| synthetic | alpha | SPI risk | verdict |
|---|---:|---:|---|
| aligned | 0.05 | 0.0505 | valid **and tighter** (CRC refuses here) |
| aligned | 0.10 | 0.1003 | valid and tighter (uses the full budget) |
| optimistic +0.25 | 0.10 | 0.5121 | **COVERAGE VIOLATED** |
| wrong shape | 0.10 | 0.7589 | **COVERAGE VIOLATED** |

**Attempt B - transport real values into synthetic rank space**

| synthetic | alpha | SPI risk | verdict |
|---|---:|---:|---|
| aligned | 0.10 | 0.0387 | valid, **no tighter than CRC** |
| optimistic +0.25 | 0.10 | 0.2887 | **COVERAGE VIOLATED** |
| uniform junk | 0.10 | 0.0390 | valid, not tighter |

**Diagnosis.** The returned threshold is always a synthetic order statistic. If
every synthetic value is shifted upward, the synthetic support contains nothing
low enough, and no rank-based correction can recover a valid threshold. SPI must
handle this through the window construction (its Equations 6-8, the R_r^+/-
definitions and the role of beta), which is exactly the part I could not obtain:
OpenReview gates on browser verification and the arXiv PDF exceeds the fetch
size limit.

**Conclusion.** The direction is sound and the headroom is real and measured,
but it depends on a faithful implementation of SPI's transporter, which I could
not reproduce from the abstract and a partial description. Getting the paper PDF
or the authors' reference code is a prerequisite, not an optimisation.

Attempts A and B are both recorded rather than deleted because each fails in an
informative direction, and because "just clip the ranks" and "just map the
values" are the two obvious things a reader would try.
