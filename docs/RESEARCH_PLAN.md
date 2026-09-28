# Research plan v3 — certified cold-start defect inspection with synthetic defects

As of 2026-09-27. Replaces plan v2 (the former `RESEARCH_PLAN.md`). The full write-up with
the literature survey is the "Complete Project Document" (`docs/Project_Document.pdf`); this file
is the short version kept next to the code, with the same section and stage numbering.

---

## 1. Problem

Certify, before a line has produced enough scrap to learn from, that a frozen object
detector lets through at most a chosen fraction alpha of defective parts, and refuse when
that cannot be certified.

For a defective image, an escape reduces to one scalar (`src/risk/escape.py`):

- **part escape** (primary): `c = max detection score`; the part is auto-accepted when `c < lambda`;
- **localised escape**: `c = min over ground-truth boxes of the best matching score`.

Escape at threshold `lambda` is `1{c < lambda}`, so conformal risk control on escape is
split-conformal on `c`. With `m` real defective calibration images it can certify only
multiples of `1/(m+1)`: it refuses below `m = 19` at alpha = 0.05, and at `m = 15`,
alpha = 0.10, its only certifiable threshold is the lowest real score.

## 2. Gap (verified 2026-09-23)

| Work | Guarantee | Detection | Scarce real defects | Synthetic defects |
|---|---|---|---|---|
| Conformal Risk Control (ICLR 2024); Learn then Test (Ann. Appl. Stat. 2025) | yes | general | no | no |
| Andéol et al. (COPA 2023); SeqCRC (arXiv 2025) | yes | yes | no | no |
| Shen & Liu (arXiv 2025; MDPI Mathematics) | yes | segmentation | no | no |
| Yuan et al. (Reliab. Eng. Syst. Saf. 2026) | yes | no (time series) | no | no |
| SPI, Bashari et al. (NeurIPS 2025) | yes | no | yes | yes |
| AnomalyDiffusion (AAAI 2024); training-free generation (ICCV 2025); MaCoDiff, JGTDiff (J. Intell. Manuf. 2026); NPI synthesis (arXiv 2026) | no | anomaly / detection | yes | yes |
| **This project** | **two-tier** | **yes** | **yes** | **yes** |

No published method certifies a detector's escape rate from a small real calibration set
using synthetic defects, or states what still holds when those synthetic defects are poor.

## 3. Method: SPERC

Apply the exact rank-coupled transporter of SPI (`src/risk/spi_exact.py`) to real and synthetic
escape scores (`src/risk/sperc.py`) and report both of SPI's statements:

- **Tier N** (SPI Theorem 3.3): `alpha - beta - eps - 1/(N+1) <= P(escape) <= alpha + beta + eps`,
  `eps` = total-variation distance between real and synthetic order-statistic laws (not observable).
- **Tier H** (SPI Theorem 3.5): `P(escape) <= 1 - #{j : R_j^+ <= ceil((1-alpha)(N+1))}/(m+1)`,
  for any synthetic data, computable from `(m, N, alpha, beta)` alone.
- **Refusal** when no beta gives a Tier H cap at or below the plant's `alpha_max`
  (`select_beta`: SPI Algorithm 4, on its step grid or on a fixed grid).

Contributions:

1. **C1** the exact scalar reduction that makes SPI apply to detection as written;
2. **C2** the two-tier industrial certificate with refusal;
3. **C3** the resolution-versus-validity result: Tier H stays quantised at `1/(m+1)`, only
   Tier N gains synthetic resolution. Measured in simulation (`docs/PROJECT_STATUS.md`,
   Section 4): SPERC operates at its target at every m, but at an equal hard guarantee
   CRC is at least as efficient;
4. **C4** generator quality measured in certification terms (training-free vs fine-tuned
   generator; does real-vs-synthetic KS distance predict the operating gap?).

No new theorem is claimed.

## 4. Rules that protect the guarantee

1. The detector never sees calibration images: not for training, early stopping or
   checkpoint selection (fixed 2026-09-23: `val` is now carved from train).
2. Generator reference defects and clean backgrounds come from the train split.
3. Real calibration defects must be exchangeable with future defective parts; drift is
   watched by `src/risk/adaptive.py`.
4. Refusals are reported as refusals, never dropped from averages.

## 5. Data and protocols

| Dataset | Role |
|---|---|
| KolektorSDD2, Magnetic-Tile | primary (clean parts, so review load is measurable) |
| NEU-DET, GC10-DET, PCB | escape-validity checks only (GC10 also carries Stage 1) |
| MVTec AD | bridge to SPI's native image-level setting (Stage 2) |
| DAGM 2007 | not in the plan's five; converter kept for an optional extra check (artificial textures) |

Splits: train / val / cal / test, `val` carved from train for the detector's own model selection;
defect-aware allocation (defective 45/30/25, clean 60/15/25) on every dataset (`configs/config.yaml`).

- **P1, calibration cold start** - subsample real defective calibration images to
  m in {5, 10, 15, 25, 40, full}, 500 draws (`scripts/run_coldstart_sweep.py`), every model x seed.
- **P2, full cold start** - detector trained on 5 real + synthetic defects + all clean images;
  a disjoint half of the synthetic set is held out as SPERC's calibration data; primary datasets.

## 6. Experiments, in order (each has a stop rule; document Section 12)

| Stage | What | Where | Stop rule |
|---|---|---|---|
| 0 | SPI on synthetic Beta scores; adversarial synthetic data | `tests/test_spi_exact.py` | realised coverage outside Thm 3.3 / 3.5 = bug. **Passes.** |
| 1 | best case: held-out real GC10 defects play the synthetic role, m in {5, 10, 15, 25} | `scripts/run_stage1.py` | no review-load gain over CRC at the same hard-cap constraint = fallback paper |
| 2 | MVTec AD bridge, image-level anomaly scores | `scripts/run_mvtec_bridge.py` | transporter must behave as SPI reports |
| 3 | retrain with defect-aware splits, 3 seeds, primary datasets first | `scripts/run_pipeline.py` | needed regardless |
| 4 | P1 sweep on the primary datasets with both generators | `run_pipeline.py --generate ...` | headline figure |
| 5 | misalignment stress: `uniform_noise`, `cross:<dataset>`, under-trained generator | every sweep | realised escape above Tier H beyond 3 SE = claim fails |
| 6 | P2 full cold start | `run_pipeline.py --p2` | reported whatever it shows |

Stage-1 decision rule (fixed in `scripts/run_stage1.py` before results): GO if at some m in
{5, 10, 15, 25} SPERC run with Tier N = target issues in >= 80% of draws, keeps mean escape <=
target + 3 SE, has a Tier H cap <= the plant's alpha_max, and sends fewer parts to review than CRC
at the same m by more than 2 paired standard errors. The comparison with CRC run *at* SPERC's cap
(same hard guarantee, no operating target) is reported alongside as C3.

Metrics (Section 12.2): realised escape (mean over draws, standard error, 95% interval, pooled
Wilson interval), issuance rate (Wilson), review load, parts labelled, Tier H cap, KS distance
real vs synthetic, and mAP@0.5 / mAP@0.5:0.95 over 3 seeds for context only. **m-star** = smallest
m at which a method keeps escape at or below the target, issues in at least 80% of draws, reaches a
review load within 1.1x of the oracle (or an absolute target), and (for SPERC) has a Tier H cap at
or below the plant's `alpha_max`; reported for alpha_max in {0.20, 0.25, 0.35, 0.50}. Seeds are
pooled by `scripts/summarize_sperc.py` (RQ1 m-star, RQ2 validity, RQ3 alignment vs m-star, RQ4
planned vs realised caps).

**Fallback paper** if Stage 1 fails: the scalar reduction, the quantisation analysis,
defect-seeking acquisition and its three refuted corrections, and the negative result
for synthetic transport.

## 7. Where to submit

*Computers in Industry* (Elsevier) or *Journal of Intelligent Manufacturing* (Springer)
for the full paper; COPA (PMLR) for a short paper on C3. Not a first-tier ML main track:
there is no new theorem. Re-run the novelty search in Scopus and Web of Science in the
submission month.

## 8. References (all opened and checked on 2026-09-23)

- Bashari, Lotan, Lee, Dobriban, Romano. Synthetic-Powered Predictive Inference. NeurIPS 2025. https://arxiv.org/abs/2505.13432 (code: https://github.com/Meshiba/spi)
- Angelopoulos, Bates, Fisch, Lei, Schuster. Conformal Risk Control. ICLR 2024.
- Angelopoulos, Bates, Candès, Jordan, Lei. Learn then Test. Annals of Applied Statistics 19(2), 2025. doi:10.1214/24-AOAS1998
- Conformal Risk Control under Non-Monotone Losses. arXiv:2604.01502.
- Andéol, Mossina, Mazoyer, Gerchinovitz. Confident Object Detection via Conformal Prediction and Conformal Risk Control: an Application to Railway Signaling. COPA 2023, PMLR 204.
- Andéol et al. Conformal Object Detection by Sequential Risk Control. arXiv:2505.24038.
- Shen, Liu. Conformal Segmentation in Industrial Surface Defect Detection with Statistical Guarantees. arXiv:2504.17721.
- Yuan, Li, Wang, Zhang. Conformal machine learning for reliable anomaly detection in industrial cyber-physical systems. Reliability Engineering & System Safety 274:112417, 2026.
- Zhao et al. DETRs Beat YOLOs on Real-time Object Detection. CVPR 2024.
- Hu et al. AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model. AAAI 2024.
- Xu et al. Training-Free Industrial Defect Generation with Diffusion Models. ICCV 2025.
- Fan et al. MaCoDiff. Journal of Intelligent Manufacturing, 2026. doi:10.1007/s10845-026-02893-5
- Mi et al. JGTDiff. Journal of Intelligent Manufacturing, 2026. doi:10.1007/s10845-026-02821-7
- Güğül, Levi, Acar. Few-shot diffusion-based defect synthesis for new product introduction. arXiv:2604.22850.
- Božič, Tabernik, Skočaj. Mixed supervision for surface-defect detection. Computers in Industry, 2021 (KolektorSDD2).
- Huang, Qiu, Yuan. Surface defect saliency of magnetic tile. The Visual Computer 36:85–96, 2020.
- He, Song, Meng, Yan. An end-to-end steel surface defect detection approach via fusing multiple hierarchical features. IEEE TIM 69(4):1493–1504, 2020.
- Lv, Duan, Jiang, Fu, Gan. Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network. Sensors 20(6):1562, 2020.
- Huang, Wei. A PCB Dataset for Defects Detection and Classification. arXiv:1901.08204.
- Bergmann et al. The MVTec Anomaly Detection Dataset. IJCV 129:1038–1059, 2021.
- DSAT, Scientific Reports 2025 (NEU-DET 83.14% mAP@0.5); SH-DETR, PLOS One 2025 (83.03%).
