# Defect-seeking calibration: an agentic contribution with measured evidence

## The question

Does an "agentic" component add credibility, or is it decoration? Answer depends
entirely on which kind. Measured evidence below for the kind that is not
decoration.

## Why a labelling policy is the right agentic object here

Escape loss is defined **only on defective parts**. The conformal penalty is
1/(n_def + 1). So the certificate is bought with *labelled defects*, and on a line
that is 89% clean a random labelling budget buys mostly zeros - labels that cost
money and move the guarantee not at all.

That makes "which part do I pay to label next" a genuine sequential decision
problem with a measurable objective, rather than a wrapper around a chatbot.

## Result 1 - guided labelling buys defects far faster

Detector max-score used as a pre-label screening signal:

| dataset | budget | random -> defectives | guided -> defectives | ratio |
|---|---:|---:|---:|---:|
| kolektor | 50 | 5.5 | **44** | **8.1x** |
| kolektor | 100 | 10.8 | 50 | 4.7x |
| kolektor | 200 | 21.3 | 53 | 2.5x |
| magnetic_tile | 50 | 14.5 | **48** | **3.3x** |
| magnetic_tile | 100 | 29.2 | 58 | 2.0x |

## Result 2 - random labelling cannot certify at all

Stochastic policy, sample with probability proportional to max-score^gamma;
gamma = 0 is uniform random. 300 replicates, KolektorSDD2, alpha = 0.10:

| gamma | budget | defectives obtained | **certificate issued** | mean realised risk |
|---:|---:|---:|---:|---:|
| 0 (random) | 50 | 5.4 | **0.0%** | never issues |
| 0 (random) | 100 | 11.0 | **0.0%** | never issues |
| 1 | 50 | 40.6 | 53.3% | 0.1043 |
| 1 | 100 | 50.1 | 63.3% | 0.1011 |
| 3 | 50 | 44.1 | 76.3% | 0.1011 |
| 3 | 100 | 50.3 | **84.3%** | 0.1011 |

**Random labelling at a realistic budget never produces a usable guarantee.** Not
"a loose one" - none at all, because CRC correctly refuses when the finite-sample
term alone exceeds the budget. Guided labelling issues one 53-84% of the time
from the same number of labels.

## Result 3 - and it is systematically, slightly invalid

Mean realised risk under the guided policy is 0.1011-0.1043 against alpha = 0.10,
and P(risk > alpha) = 1.000 across replicates. The overshoot is small (1-4%
relative) but it is **systematic and always in the same direction**.

The cause is not subtle: selecting on the detector score selects on the very
quantity being calibrated. Screening for high-scoring parts preferentially finds
*easy* defects, which carry high escape confidence, so the calibration sample
flatters the detector and the threshold comes out too permissive.

## Why this is the contribution rather than a problem

The three results compose into one statement:

> A defect-seeking labelling policy is the difference between **no certificate at
> all** and a certificate 84% of the time on the same budget - but it is biased by
> construction, and the bias must be corrected rather than ignored.

The correction is tractable precisely because **the policy is ours**: the
selection probabilities are known by construction, which is the setting weighted
conformal prediction (Tibshirani et al., covariate shift) is built for. Standard
adaptive calibration selection is invalid because the selection rule is unknown or
data-dependent in an uncontrolled way; here it is neither.

## What is NOT claimed

The weighted correction is **not yet implemented or tested**. Results 1-3 are
measured; the fix is motivated but unproven. It is written down here so the claim
and the evidence stay distinguishable.

## Relation to existing work

Conformal-guided active learning exists (e.g. Conformal Cross-Modal Active
Learning, CVPR 2026) but labels to improve the **model**. The object here is the
**certificate**: labels are bought to shrink 1/(n_def+1) and tighten a risk bound,
which is a different objective and a different validity problem. Searches for
active *calibration* selection surfaced the obstacle - adaptive selection breaks
exchangeability - but no method that exploits a known selection rule to keep
validity while seeking defects.

---

## Result 4 (new) - the obvious correction cancels the benefit

Inverse-propensity weighting is the textbook fix for a known sampling law, and it
was implemented (`src/risk/acquisition.py`) and tested. It does not work here.

KolektorSDD2, budget 100, alpha = 0.10, 300 replicates:

| gamma | weight clip | defects bought | certificate issued | realised risk |
|---:|---:|---:|---:|---:|
| 0 (random) | none | 10.3 | 0.0% | - |
| 1 | none | 49.3 | 0.0% | - |
| 3 | none | 50.3 | 0.0% | - |
| 1 | 5x | 49.3 | 3.3% | 0.1191 |
| 3 | 5x | 50.3 | 0.0% | - |
| 3 | 20x | 50.3 | 0.0% | - |

**Why.** Weighted CRC charges the unobserved test point at the worst weight it
could carry:

    R(t) = ( sum_i w_i 1{c_i < t} + w_max ) / ( sum_i w_i + w_max )

With uniform weights that term is 1/(n+1) and behaves. Under a gamma = 3 screen
the weight ratio w_max / w_min is enormous, so w_max alone dominates both sums
and the certified risk never falls below alpha at any useful threshold. The
screen buys 5x more defects and the correction gives all of it back - and then
some, because the naive estimator at least issued 83% of the time.

Weight clipping does not rescue it: clipping enough to issue (5x at gamma = 1)
reintroduces exactly the bias the weights existed to remove (0.1191 against
alpha = 0.10).

## Where this leaves the agentic direction

Honest state:

* the **efficiency gain is real and large** - 8x more defects per label, and the
  difference between a certificate 0% and 83% of the time;
* the **bias is real but small** - 0.1011-0.1043 against 0.10;
* the **textbook correction is unusable** - it destroys the effective sample size
  and issues nothing.

So the contribution is not "screen and reweight". Something with *bounded* weights
by construction is needed - stratified acquisition with known stratum counts is
the obvious candidate, since a stratified estimator's weights are fixed by design
rather than by 1/pi. That is **untested** and is written here as the next
experiment, not as a claim.

Reporting all four results together is the honest framing: an acquisition policy
that changes "no certificate" into "a certificate", a measured bias, and a
demonstration that the obvious fix costs more than the bias it removes.
