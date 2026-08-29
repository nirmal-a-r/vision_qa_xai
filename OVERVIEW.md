# Project overview — certified defect inspection under scarce labels

State as audited from disk. Every number below was recomputed from the repository
at the time of writing, not quoted from memory.

---

## 1. The thesis in one paragraph

An inspection system should not report a confidence. It should report a
**guarantee**: given a target defect escape rate `alpha`, either return an
operating point whose escape rate is certified at or below `alpha`, or **refuse**
and hand the line back to human review. The refusal branch carries as much weight
as the first — a method that always answers is a method that sometimes lies.

The contribution is not the detector. It is the finding that the guarantee is
bought with **labelled defects** rather than labelled parts, that this makes
cold-start the binding constraint on a real line, and a measured account of which
acquisition strategies buy their way out of it and which do not.

---

## 2. What is established

### 2.1 The reduction — VALIDATED

For a defective image *i*, define the **escape confidence**

```
c_i = min over ground-truth boxes g of  max{ score of a detection matching g }
```

with `c_i = -inf` when any box is never matched. Then the image-level escape loss
is `L_i(lambda) = 1{lambda > c_i}`, so the empirical risk is exactly the empirical
CDF of the `c_i`, and CRC's rule becomes **a quantile of the `c_i`**.

The loss curve collapses to one scalar per image. That puts detection risk control
into the plain scalar setting the conformal literature is written for, with no
set-valued machinery required.

*Validated:* CRC on the reduced score holds coverage over 400–600 trials
(0.032 / 0.082 / 0.179 against alpha 0.05 / 0.10 / 0.20).

### 2.2 The guarantee on real detectors — VALIDATED

**20/20 operating points hold**, across 5 datasets × 4 alpha levels, over 60
repeated calibration splits each.

Two behaviours that look like failures and are not:

* **Refusal.** Every dataset refuses at `alpha = 0.01`. The finite-sample term
  `1/(n+1)` alone exceeds the budget, so no threshold is certifiable and the
  honest answer is to say so.
* **CRC bounds the mean, not every draw.** Individual splits exceed `alpha`; the
  theorem is about the expectation. Reporting only the *issued* splits is a
  selection on the calibration draw and makes a satisfied guarantee look violated
  — kolektor at `alpha = 0.10` reads 0.1730 conditional but **0.0288
  unconditional**.

### 2.3 The binding constraint — MEASURED

The penalty is `1/(n_def + 1)` over images where the loss is *defined*, i.e.
defective ones. A clean part contributes an identically-zero row.

| dataset | cal images | defective | alpha floor |
|---|---:|---:|---:|
| kolektor | 500 | **53** | **0.0185** |
| magnetic_tile | 203 | **59** | **0.0167** |
| pcb | 102 | 102 | 0.0097 |
| neu | 270 | 270 | 0.0037 |
| gc10 | 342 | 342 | 0.0029 |

No detector, however good, can certify below its own floor. The two datasets that
carry the triage argument have the worst floors, because their defects are rare —
which is also what makes them realistic. A newly commissioned line has fewer.

*Mitigation implemented:* defect-aware split allocation (defective images get
45/30/25, clean keep 60/15/25) roughly halves every floor — kolektor 0.0185 →
0.0093. Clean images move for free because they never entered the risk estimate.
**Requires a retrain to take effect.**

### 2.4 Defect-seeking acquisition — MEASURED

KolektorSDD2, `alpha = 0.10`, 300 replicates:

| policy | budget | defects bought | **certificate issued** | realised risk |
|---|---:|---:|---:|---:|
| random | 50 | 5.4 | **0.0%** | never |
| random | 100 | 10.3 | **0.0%** | never |
| defect-seeking | 50 | 44.1 | 76–80% | 0.1011 |
| defect-seeking | 100 | 50.3 | **82.7%** | 0.1011 |

Random labelling at a realistic budget produces **no usable guarantee at all**.
Screening on the detector score buys defects ~8× faster and turns that into a
certificate most of the time on the same spend.

It is also **biased by 1.1% relative** (0.1011 against 0.10), because screening on
the detector score selects on the quantity being calibrated and finds *easy*
defects that flatter the detector.

### 2.5 The correction frontier — three failures, one shared cause

| correction | result | why |
|---|---|---|
| inverse-propensity weighting | **0% issuance** | weight ratio reaches **2,103,520**; `w_max` dominates the estimator |
| stratified design weights | best *valid* design issues **6.0%** at risk 0.1049 | bounds weights to 1.0–2.8 as intended, but the budget spreads into defect-free strata |
| deflated alpha | 82.7% at `alpha'=0.100` → **0.0% at 0.095** | a cliff, not a slope |

The third explains the other two. With ~50 defective calibration points the
achievable operating grid **is** the order statistics — spacing ≈ 1/51 ≈ 0.02.
Every correction attempts a fine adjustment on a grid coarser than the adjustment
itself.

**This is the transferable finding:** conformal operating points are *quantised*
at small `n_def`, so any method assuming a continuous `alpha` knob assumes
something the data cannot support — precisely the cold-start regime of interest.

**Recommendation:** report the 1.1% overshoot rather than correct it. Every
correction measured costs more than the bias it removes, structurally rather than
through mis-tuning.

---

## 3. What was refuted — kept deliberately

A reader with no view of what failed has no way to calibrate what succeeded.

| idea | outcome | evidence |
|---|---|---|
| learned nonconformity score | **REFUTED** | ranking improves (AUC +0.017…+0.057) but auto-accept at certified `alpha=0.10` gets *worse* on 3/4 datasets; kolektor 88.1% → 0.7% |
| CCE preprocessing (raw \| CLAHE \| HF residual) | **NEUTRAL** | separability +23…+352% but mAP **−0.76pp** mean over 3 valid pairs |
| faithfulness-gated triage | **VACUOUS AS BUILT** | optimiser selects `phi=0`, disabling its own gate; faithfulness covered 5.3% of test images and was never passed into the fit |
| inverse-propensity correction | **REFUTED** | see 2.5 |
| stratified acquisition | **REFUTED** | see 2.5 |
| deflated-alpha calibration | **REFUTED** | see 2.5 |
| SPI transporter reconstruction | **FAILED** | two reconstructions: one tighter but coverage-violating (0.51, 0.76), one valid but never tighter. Window equations unobtainable — OpenReview gates on browser verification, arXiv PDF exceeds fetch limit |

Eight additions measured, seven failed. The one that worked is 2.4.

---

## 4. Assets

| | |
|---|---|
| datasets | 5 industrial, 9,468 images, 11,543 boxes, 28 classes, geometry-verified |
| detectors | **10 of 20** configurations trained (5 datasets × 2 models × 2 encodings) |
| tests | `test_conformal` 10/10 · `test_adaptive` 6/6 · `test_rigor` 3/3 |
| source | 20 modules |
| notebooks | 4 (2 deliverable, 2 executed copies) |
| figures | 14 PNG + PDF |
| write-ups | 4 experiment records in `experiments/` |
| commits | 10 |

**Detection accuracy** (valid runs only):

| model | dataset | mAP@0.5 |
|---|---|---:|
| YOLOv8s | pcb | 0.9718 |
| RT-DETR | magnetic_tile | 0.9451 |
| YOLOv8s | magnetic_tile (CCE) | 0.9009 |
| YOLOv8s | neu | 0.7331 |
| RT-DETR | neu | 0.7052 |
| YOLOv8s | kolektor | 0.6790 |
| YOLOv8s | gc10 | 0.6268 |

Published SOTA on NEU is **0.831** (SH-DETR, PLOS One 2025) / **0.8314** (DSAT,
Sci. Reports 2025). Ours is 0.733. That gap is real and must be reported; the
claim is denominated in labelling effort, not mAP.

---

## 5. Literature position

**Base paper (verified by reading):** Bashari, Lotan, Lee, Dobriban & Romano,
*Synthetic-Powered Predictive Inference*, **NeurIPS 2025**, arXiv:2505.13432.

| work | year | task | guarantee | acquisition | cold-start |
|---|---|---|---|---|---|
| Shen & Liu (arXiv 2504.17721) | 2025 | segmentation | pixel FDR/FNR | — | — |
| SeqCRC (arXiv 2505.24038) | 2025 | detection, general | risk control | — | — |
| Selective CRC (arXiv 2512.12844) | 2025 | generic ML | risk control | — | — |
| SPI (NeurIPS) | 2025 | classification, regression | coverage | — | scarcity |
| SH-DETR (PLOS One) | 2025 | steel detection | none | — | — |
| DSAT (Sci. Reports) | 2025 | steel detection | none | — | — |
| **this work** | — | **detection, 5 domains** | **instance escape** | **measured frontier** | **the subject** |

Nobody occupies the last row. The guarantee people work on pixels or generic ML;
the detection people optimise mAP; nobody treats *labelled-defect scarcity* as the
object of study.

**Caveat:** that is a claim about my searches. Elsevier, MDPI and Nature returned
403/login walls, so paywalled and un-indexed work is invisible here. Run the
intersection terms in Scopus and Web of Science before submitting.

---

## 6. Honest gaps

1. **Single seed everywhere.** 10/10 configurations trained once. No error bars.
   This is the first thing a reviewer asks for; it is pure GPU time.
2. **10 of 20 detector runs incomplete.** GC10+CCE, PCB+CCE and all four
   RT-DETR+CCE runs are missing — killed when another project took the GPU.
3. **CCE ablation rests on 3 of 5 pairs** and reads neutral.
4. **Drift is synthetic.** Shift is induced by biasing the test block, not
   observed on a line over time.
5. **Defect-aware splits are implemented but not applied** — doing so invalidates
   every cached model and needs a full retrain.
6. **The acquisition bias is unresolved**, and Section 2.5 argues it should be
   reported rather than corrected. A reviewer may disagree; the evidence for the
   position is there either way.

---

## 7. What I would do next, in order

1. **Multi-seed retrain with defect-aware splits.** Closes gaps 1, 2, 3 and 5 in
   one overnight job, and halves every alpha floor. Highest value per GPU-hour by
   a wide margin.
2. **Faithfulness at full coverage on the two primary datasets.** Makes the
   faithfulness claim falsifiable instead of decorative — and if the gate still
   never fires, drop it and say so.
3. **Obtain the SPI paper PDF or reference code.** The synthetic-calibration
   direction is blocked on equations I could not retrieve, not on ideas.
4. **Scopus/WoS search** to convert the novelty claim in Section 5 from
   "my web searches found nothing" into something defensible in a rebuttal.
