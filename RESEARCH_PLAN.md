# Research plan v2 — cold-start risk control for defect inspection

Written after auditing v1 as a hostile reviewer and killing the parts that did not
survive. Every claim below is marked with how it was verified.

---

## 1. Why v1 needed replacing

| v1 contribution | verdict | evidence |
|---|---|---|
| C1 escape risk + per-class budgets | **keep** | 15/15 operating points validated over 60 repeated splits |
| C2 faithfulness-gated triage | **dead** | never tested; the optimiser selects φ=0, i.e. disables its own gate; faithfulness computed for 125/2369 test images (5.3%); `run_all_experiments.py` never passes faithfulness into the triage fit at all |
| C3 drift-aware recalibration | **weak** | synthetic covariate shift only, no real production drift |
| C4 CCE preprocessing | **neutral** | −0.76pp mean over 3 valid pairs (the −18pp figure came from two interrupted runs recorded as finished) |
| C5 learned nonconformity score | **refuted** | tested and written up in `experiments/NEGATIVE_learned_nonconformity.md` — ranking improves (AUC +0.017…+0.057) but auto-accept at certified α=0.10 gets *worse* on 3/4 datasets, catastrophically on KolektorSDD2 (88.1% → 0.7%) |

One survivor out of five. v1 was a wrapper around an off-the-shelf detector, and
its novel-sounding parts were untested.

---

## 2. The real problem, found by measurement not intuition

The conformal penalty is `B/(n+1)` where `n` counts calibration points on which
the loss is **defined**. Escape loss is only defined on defective images — a clean
part cannot escape. So the guarantee's tightness is set by the number of
**defective** calibration images:

| dataset | cal images | defective | α floor |
|---|---:|---:|---:|
| kolektor | 500 | **53** | **0.0185** |
| magnetic_tile | 203 | **59** | **0.0167** |
| pcb | 102 | 102 | 0.0097 |
| neu | 270 | 270 | 0.0037 |
| gc10 | 342 | 342 | 0.0029 |

α = 0.01 was refused on every dataset. Not because the detector is weak — because
**a perfect detector still cannot certify 1% escape from 53 defective examples.**

This is the industrial bottleneck, and it gets worse in the field, not better:
a newly commissioned line has *fewer* labelled defects than KolektorSDD2, not more.
A guarantee that needs hundreds of labelled failures is a guarantee you cannot have
when you most need it — at commissioning, before the line has produced enough scrap
to learn from.

**No prior work addresses this.** Shen & Liu (2025) and SeqCRC (2025) both assume
abundant real calibration data.

---

## 3. Base paper

**Synthetic-Powered Predictive Inference (SPI)** — Bashari, Lotan, Lee, Dobriban &
Romano · NeurIPS 2025 · [arXiv:2505.13432](https://arxiv.org/abs/2505.13432)

*Verified: abstract and algorithm read directly.*

SPI solves exactly the scarcity problem, in general ML. Its core is a **score
transporter**: an empirical quantile mapping from real scores onto synthetic ones.
The prediction set uses a quantile of the **synthetic** scores, with the real scores
entering only through *ranks*. Validity survives because the window indices
`R_r^±` depend only on `m, N, β` — never on score values — so the method cannot
inflate its effective sample size. Theorem 3.5 gives worst-case coverage
"regardless of the relationship between Q and P", i.e. **even if the synthetic data
is bad, the guarantee holds; only tightness suffers.**

That last property is what makes it deployable: a plant is not asked to trust a
generative model, only to benefit from it when it happens to be good.

**Evaluated on image classification (diffusion-generated images) and tabular
regression. Not object detection. Not industrial inspection.**

### A caution recorded honestly

My first attempt to apply SPI transported *synthetic → real* and stacked the result
into the calibration set. That **broke coverage** — test risk 0.12–0.16 against
α=0.10 — because stacking lets CRC treat `n=170` transported rows as independent
when the effective sample size is still ~20. The failure is instructive: it is
precisely the error SPI's rank-coupling construction exists to prevent, and it
confirms the mechanism is load-bearing rather than cosmetic.

---

## 4. Proposed contribution

**Synthetic-powered escape-risk control for cold-start defect inspection.**

Extend SPI from scalar classification scores to the **monotone escape-loss curve**
of object detection, using diffusion-synthesised defects as the synthetic source,
so a line can carry a certified escape bound from day one.

Three things make this a technical extension rather than an application note:

**(a) The loss is a curve, not a scalar.** SPI transports one nonconformity score
per calibration point. CRC for detection operates on `L_i(λ)`, a monotone
non-increasing function of the threshold, and takes its mean across `i`. Transporting
a *curve* while preserving monotonicity — a precondition of the CRC theorem, and one
this codebase already enforces with a hard check — is not addressed by SPI.

**(b) The support is conditional.** Escape loss exists only on defective images, so
the synthetic source must generate *defects*, and the exchangeability argument is
over the defective sub-population rather than all parts. On KolektorSDD2 that is 53
of 500 calibration images.

**(c) The synthetic source is physically constrained.** Diffusion defect synthesis
must respect the substrate: a generated scratch on steel has to obey the same
illumination and texture statistics as a real one, or the transporter's alignment —
and therefore all of the tightness — collapses. This connects the generative model
back to the photometric analysis already in the repo.

### What would be claimed

> At a fixed certified escape budget, synthetic-powered calibration reduces the
> number of **real labelled defects** required to reach a usable operating point by
> a factor of X, with coverage preserved.

This is an honest claim because it does not require beating anyone's mAP. It is
also the claim a plant manager can act on: it is denominated in labelling effort.

---

## 5. Test before building — the premise that must hold first

The one experiment that decides whether this is a paper:

> Does an SPI-transported synthetic defect set tighten the certified escape
> threshold at fixed α, **without** violating coverage on held-out real data?

Cheap protocol, no diffusion model and no GPU needed for the first pass: take GC10
(342 defective calibration images), hold out `n_real ∈ {20, 30, 50}` as the
"trusted" set and use the remainder as a stand-in for synthetic data. This is the
*best case* for alignment — real data standing in for synthetic. If SPI does not
help here, it will not help with a diffusion model, and the idea dies cheaply.

Only if that passes is it worth training a defect diffusion model.

### Order of work

1. Implement SPI's transporter correctly (rank-coupled windows, real→synthetic,
   quantile of synthetic scores) and unit-test coverage on synthetic data, exactly
   as `tests/test_conformal.py` already does for CRC and LTT.
2. Best-case test above. **Go / no-go.**
3. Misalignment stress test: use another dataset's defects as the surrogate, i.e.
   deliberately bad synthetic data. Confirm coverage holds and measure how much
   tightness degrades — this is the honest robustness result.
4. Only then: diffusion defect synthesis, conditioned on the substrate.
5. Multi-seed + defect-aware splits, one retrain.

---

## 6. Datasets — reframed

Not "five datasets". Two roles:

* **Primary — KolektorSDD2, Magnetic-Tile.** The only sets with clean parts, so the
  only ones where auto-accept, review load and triage are evaluable at all. Also the
  scarcest in defective calibration data (53, 59), which is exactly the regime the
  contribution targets. The weakness of v1 becomes the subject of v2.
* **Replication — NEU-DET, GC10-DET, PCB.** Escape-risk control only. Every image
  contains a defect, so auto-accept is vacuous there by construction, and any paper
  reporting a triage number on them is reporting an artefact.

---

## 7. Honest risk register

| risk | severity | mitigation |
|---|---|---|
| SPI does not tighten for a monotone loss curve | **fatal** | step 2 is a cheap go/no-go before any model is trained |
| Diffusion defects too unrealistic to align | high | SPI stays *valid* under misalignment; the paper then reports a null tightening, which is still a result |
| Detector mAP below SOTA (0.733 vs 0.831) | medium | the claim is denominated in labelling effort, not mAP; report the gap plainly |
| Single seed everywhere (12/12 configs) | medium | GPU time only |
| Reviewer: "this is SPI applied to detection" | high | (a)–(c) above must be argued and *demonstrated*, not asserted |
