# Project status

As of 2026-09-27. Replaces the former `OVERVIEW.md`. Every number here is either
measured by code in this repository (source named) or marked as not yet recomputed. The plan
itself is `docs/Project_Document.pdf`; section and stage numbers refer to it.

---

## 1. In one paragraph

An inspection system should report a **guarantee**, not a confidence: given a target
defect-escape rate, either certify an operating point or refuse. The guarantee is bought
with labelled **defects**, so a newly commissioned line, which has very few, cannot get
one. SPERC uses synthetic defects to operate near the target from a handful of real ones,
and states exactly what still holds when the synthetic defects are poor.

## 2. What is established

| Result | Numbers | Source |
|---|---|---|
| Escape reduces to one scalar per image | CRC on the reduced score: 0.032 / 0.082 / 0.179 escape at alpha 0.05 / 0.10 / 0.20 (600 trials) | `experiments/SPI_reduction_and_headroom.md` |
| Alpha floor is set by defective calibration images | KolektorSDD2 53 defective, floor 0.0185; Magnetic-Tile 59, floor 0.0167 | notebook Section 4 |
| CRC leaves budget unused at small m | m = 25: refuses at 0.05, uses 40% of budget at 0.10 | `experiments/SPI_reduction_and_headroom.md` |
| Defect-seeking labelling buys certificates | KolektorSDD2, budget 100: 0% issuance at random vs 82.7% guided; realised 0.1011 vs 0.10 | `experiments/AGENTIC_defect_seeking_calibration.md`, `ACQUISITION_frontier.md` |
| Three bias corrections fail for one reason | inverse-propensity 0%, stratified 6.0% best valid, deflated alpha 0% at 0.095 | `experiments/ACQUISITION_frontier.md` |
| Drift controller | 0.1007 after a regime shift where a static threshold reaches 0.9618 | first-study README (git history), notebook drift section |

**Note on the older numbers.** They were produced by a CRC routine that returned the
k-th order statistic where the (k+1)-th was still certified: valid, but conservative by
one step. `src/risk/escape.py` now computes the exact threshold (tested against the
generic CRC implementation), so re-runs will be slightly less conservative.

**Operating-point count (recomputed from `archive/v1_results/risk_results.json`).** Earlier
documents quoted 12/12, 15/15 and 20/20; all three were wrong. The file covers 5 datasets x 4
alpha levels = 20 operating points, 60 calibration resamples each. Mean realised escape is at or
below alpha on **18/20**. The two exceptions are Magnetic-Tile at alpha 0.10 (0.1042) and 0.20
(0.2006); both are within one resampling standard error (0.0057 and 0.0078), so they are
consistent with the guarantee but must not be reported as "20/20". These numbers come from
detectors trained with the old splits (calibration images used for checkpoint selection) and are
not certified results.

## 3. What was refuted (kept deliberately)

| Idea | Outcome | Evidence |
|---|---|---|
| Learned nonconformity score | Refuted | AUC +0.017 to +0.057, but auto-accept at certified 0.10 drops on 3 of 4 datasets (KolektorSDD2 88.1% to 0.7%) |
| CCE preprocessing | Neutral | mAP −0.76 pp mean over 3 valid pairs |
| Faithfulness-gated triage | Vacuous as built | optimiser picks phi = 0; faithfulness covered 5.3% of test images and never entered the fit. **Not a contribution.** |
| Inverse-propensity / stratified / deflated-alpha corrections | Refuted | Section 2 above |
| SPI reconstruction from the abstract (old `spi.py`) | Failed, now replaced | the exact transporter is in `src/risk/spi_exact.py` |

## 4. SPERC: current state

Implemented (`src/risk/spi_exact.py`, `src/risk/sperc.py`), tested (Stage 0:
`tests/test_spi_exact.py`, 10 tests, and `tests/test_sperc.py`, 9 tests - both pass) and checked on
simulation; every Tier H cap of the document's Section 8.4 table is reproduced to 3 decimals. Not yet
evaluated on the retrained detectors or with real generators.

Simulation, aligned generator, N = 1,000 synthetic defects, beta = 0.05, 400 draws
(mean true escape; `scripts/demo_sperc.py` reproduces the m = 15 case):

| m | target a | CRC at a | SPERC at a (Tier N a + 0.05) | Tier H cap | CRC at the Tier H cap |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.05 | refuses | 0.049 | 0.333 | 0.335 |
| 15 | 0.05 | refuses | 0.048 | 0.188 | 0.180 |
| 25 | 0.05 | 0.039 | 0.048 | 0.154 | 0.154 |
| 15 | 0.10 | 0.067 | 0.097 | 0.250 | 0.246 |
| 40 | 0.10 | 0.099 | 0.099 | 0.195 | 0.195 |

With a deliberately bad generator (synthetic defects far too easy) SPERC's escape was
0.185 against a Tier H cap of 0.1875; naive pooling reached 0.58.

**What this means for the claim.** SPERC operates at its target at every m, including
where CRC refuses. At the same *hard* guarantee, CRC is at least as efficient. Synthetic
data sharpens the operating point; it does not sharpen the worst case. The paper must
claim exactly this, not "fewer real defects for the same hard guarantee".

## 4b. Stage 1 preview on the first-study detectors (old splits, NOT certified)

`scripts/run_stage1.py` and the full sweep were run on the archived detections of the first study
(YOLOv8s, seed 0, old splits: checkpoints were selected on the calibration images, so these are a
signal, not results). 500 draws per point, each re-splitting the held-out images; ties broken at
random; part escape; target 0.10; unused real defects stand in for synthetic ones.

**Stage 1 verdict: GO** on GC10 (the document's Stage 1 dataset) and on both primary datasets. In
every case the passing setting is m = 5, beta = 0.05, alpha_max >= 0.35: CRC must refuse (review
load 1.0) while SPERC issues with mean escape 0.03-0.05 and a Tier H cap of 0.333. Review-load gain
over CRC at m = 5: GC10 +0.046, KolektorSDD2 +0.516, Magnetic-Tile +0.636 (paired SE <= 0.011).
Output: `runs/stage1_v1preview/`.

| Dataset | m | CRC at 0.10: escape / review | SPERC, Tier N = 0.10 (a = 0.05): escape / review / Tier H | SPERC at a = 0.10: escape / review | acquisition, uncorrected: escape / parts labelled |
|---|---:|---|---|---|---|
| KolektorSDD2 | 5 | refuses / 1.000 | 0.027 / 0.506 / 0.317 | 0.075 / 0.162 | refuses / 5 |
| KolektorSDD2 | 15 | 0.060 / 0.367 | 0.027 / 0.531 / 0.188 | 0.075 / 0.168 | **0.384** / 15 (random: 106) |
| KolektorSDD2 | 107 (all) | 0.093 / 0.142 | - | - | - |
| Magnetic-Tile | 5 | refuses / 1.000 | 0.033 / 0.367 / 0.329 | 0.077 / 0.342 | refuses / 5 |
| Magnetic-Tile | 15 | 0.065 / 0.356 | 0.031 / 0.370 / 0.188 | 0.083 / 0.339 | **0.200** / 15 (random: 42) |
| Magnetic-Tile | 117 (all) | 0.095 / 0.333 | - | - | - |

Stress sources at m = 15 (SPERC with Tier N = target, Tier H cap 0.188): KolektorSDD2 noise 0.075,
GC10 defects 0.155, NEU defects 0.148; Magnetic-Tile 0.010, 0.041, 0.037 - all under the cap.
Validity over all five datasets (`runs/coldstart_v1preview/summary/`): 1,246 rows carry a hard bound
(936 from stress sources); one (CRC at 0.1875, KolektorSDD2 localised, m = 15) sits 3.2 SE above its
cap, against 1.7 such rows expected by chance at this count, and none survives the Holm correction.
Re-run alone with 6,000 draws that row gives 0.1893 +/- 0.0013 against 0.1875. RQ2 verdict on the
preview: the claim holds; planned vs realised caps: 0 disagreements.

**What the preview says.**
1. SPERC's clearest gain is exactly where CRC must refuse (m below 1/alpha - 1): it issues a usable
   certificate from 5 real defects, provided the plant accepts a worst-case cap around 1/3.
2. Under alignment the realised escape of SPERC sits near alpha, not alpha + beta: run with Tier N =
   target it is conservative (escape ~0.03, more review than CRC at m >= 10). Run at alpha = target it
   operates near the target (0.075-0.083) with Tier N = target + beta. The paper must show both.
3. m-star (review within 1.1x of the oracle): Magnetic-Tile CRC 10, SPERC at a = 0.10 5 for
   alpha_max >= 0.35; KolektorSDD2 CRC 40, SPERC not reached (review 0.162-0.168 against a 0.156
   bar). The answer depends on the plant's tolerance, as the plan anticipated (C3).
4. Defect-seeking acquisition without correction is badly biased on these detectors (escape 0.38
   and 0.20 against 0.10), while needing 7x fewer labelled parts - baseline 4 behaves as the plan
   expects, and more strongly than the first study's budget-based numbers.

## 5. Assets

| | |
|---|---|
| Datasets | the plan's five (9,468 images, 11,543 boxes, 28 classes), all present in `data/raw` (checked by `scripts/check_ready.py`); MVTec AD (Stage 2) still to download; DAGM 2007 present but optional (not in the plan) |
| Detectors | RT-DETR-L (main), YOLOv8s (baseline); trained weights and caches live in `runs/`, which git does not track |
| Tests | 61: conformal 10, spi_exact 10 (Stage 0), sperc 9, adaptive 6, escape 5, prepare 5, synth 5, rigor 3, stage scripts 3, windows 3, cold-start sweep 2 |
| Notebook | `notebook/VisionQA_CertifiedInspection.ipynb`, rebuilt by `scripts/build_certified_notebook.py`, follows the document section by section; runs with 0 errors with or without results on disk |
| Figures | 14 PNG + PDF from the first study; fig15 = the SPERC system (document Section 7) |

## 6. Fixes made on 2026-09-23

1. **Calibration data no longer touches training.** `data.yaml` pointed Ultralytics'
   `val` split at the calibration block, so early stopping and `best.pt` selection used
   calibration images. That breaks the independence the guarantee needs. Splits are now
   train / val / cal / test with `val` carved from train. **All detectors must be retrained;
   results from earlier caches should not be reported as certified.**
2. `src/risk/spi.py` (a heuristic that could never beat CRC) replaced by the exact SPI
   transporter and the two-tier certificate (`src/risk/sperc.py`).
3. One module for escape (`src/risk/escape.py`): part-level and localised, both 0/1.
4. Exact CRC threshold (was conservative by one order statistic).
5. Detection caches: CCE models are no longer scored on non-CCE images; extra seeds get
   their own cache files; RT-DETR caches use one name everywhere (`rtdetr-l`); the confidence
   floor is 0.001 everywhere.
6. New: synthetic-defect compositing, synthetic scoring, cold-start sweep, commissioning
   table, one-minute demo, and their tests.
7. Pipeline: seeds, model choice, optional CCE, synthetic and SPERC stages, correct notebook.
8. `requirements.txt` now lists `ultralytics`; `configs/config.yaml` now configures SPERC
   (the old file described a removed Kaggle / Deformable-DETR pipeline).
9. Merged with the old working copy (`Desktop\cv project\vision_qa_xai`): pretrained weights
   copied; result files of the first study archived in `archive/v1_results/`; DAGM 2007
   converter added; `scripts/check_ready.py` added. The datasets themselves must be copied with
   File Explorer (the remote link is too slow; see README).
10. Sweep brought to the full plan: beta grid {0.05, 0.10, 0.20}; baselines naive pooling and the
    old heuristic; stress sources (uniform noise, other lines' defects); calibration/test
    re-splitting per draw; 95% intervals; 3-standard-error violation flag; m-star table.
11. Protocol P2 implemented (`src/data/build_coldstart_tree.py`, `run_pipeline.py --p2`); MVTec AD
    converter added (optional bridge dataset).
12. The manual copy from the old folder overwrote the edited code; it was restored, the old
    detectors moved to `archive/v1_runs_old_splits/`, and old-split data folders moved to
    `_old_splits_DELETE_ME/`.
13. Removed 16 obsolete files (old notebook builders and patchers, duplicate runners,
   `spi.py`, and the former OVERVIEW / RESEARCH_PLAN, rewritten into `docs/`).

## 6b. Built to the complete project document on 2026-09-27

1. `src/risk/spi_exact.py` (Section 11.3): SPI's exact rank law, windows, transporter T, prediction
   set, Theorem 3.3 / 3.5 bounds and Algorithm 4, checked against the paper's equations;
   `sperc.py` now builds on it; `src/risk/spi.py` is a facade under the name the document uses.
   Stage 0 tests `tests/test_spi_exact.py` (rank law vs a 40,000-draw simulation, Section 8.4 table,
   transporter brute force, Tier N under alignment, Tier H under four adversaries incl. Corollary
   3.6, Algorithm 4, random tie-breaking).
2. Random tie-breaking (Section 8.5, assumption 4) in every sweep, missed defects included.
3. Sweep (Sections 12.1-12.3): both detectors x 3 seeds; baseline 4 (defect-seeking acquisition,
   uncorrected) and parts labelled; Wilson intervals; paired review-load gains; alpha_max applied at
   analysis time so one sweep serves every plant tolerance; dataset-level KS alignment; multiplicity-
   corrected validity verdict.
4. `scripts/run_stage1.py` (Stage 1 go / no-go with a rule fixed in code), `scripts/run_mvtec_bridge.py`
   + `src/evaluation/anomaly_scores.py` (Stage 2, PatchCore-lite), `scripts/summarize_sperc.py`
   (seeds pooled; RQ1-RQ4 tables), `scripts/commissioning_table.py` prints the Section 8.4 layout.
5. `src/synth/`: train-split-only references (enforced), copy-paste and training-free
   SD-inpainting generators, AnomalyDiffusion and TF-IDG wrappers, a collector, audit sheets;
   `scripts/generate_synthetic.py`.
6. P2: a disjoint half of the synthetic set is held out from the P2 detector and used as SPERC's
   synthetic calibration data.
7. Splits: defect-aware allocation on every dataset (Section 9.1; configurable).
8. Datasets per the plan: five + MVTec AD bridge; DAGM optional.
9. Escape unified: `experiments.py` reports CRC on both 0/1 events; the triage certifies escape per
   defective part; the faithfulness gate is fixed off (dropped from the claims).
10. Pipeline: primary datasets first, every seed's detector scores the synthetic sets, Stage 1,
    summary, MVTec, P2 fixes, `--work` for a separate workspace; `scripts/smoke_test.py` runs every
    stage on a tiny data copy; `scripts/execute_notebook.py` pins the kernel to the venv.
11. `scripts/check_ready.py` checks the split allocation and the generator extras; requirements list
    diffusers / transformers / accelerate / pycocotools (Section 10.2).
12. Notebook rebuilt to follow the document; README, RESEARCH_PLAN aligned (stage numbering, 61 tests).

## 6c. Windows / RTX 5060 optimisations (2026-09-27)

Document Sections 10.3-10.4 made automatic (`configs/config.yaml` -> `training`,
`src/utils/winenv.py`): the 8 GB batch profile (YOLOv8s 8 / 4 / 2, RT-DETR-L half) chosen
automatically below 12 GB, with an out-of-memory retry at half the batch and the used batch recorded;
one process per training run; `workers=0` with a RAM image cache; fp16 detection caching; absolute
Ultralytics project path (no runs/detect/runs/detect nesting) and portable weight paths; keep-awake
during long runs; free-VRAM GPU gating; UTF-8 logs; write-then-rename caches and results; PyTorch
SDPA attention for the diffusion generator; Windows checks in `check_ready.py` (long paths, OneDrive,
battery, RAM, free VRAM, driver, WSL2); five double-click launchers in `scripts/windows/`.

## 7. Known limits and next steps

1. Retrain every detector with the fixed splits, 3 seeds, primary datasets first
   (`python scripts/run_pipeline.py --wait-for-gpu`), after `python scripts/smoke_test.py` passes.
2. Stage 1 on the new detectors (automatic in the pipeline); the preview above says GO.
3. Generators: `training_free` first (days), AnomalyDiffusion second (WSL2), N = 1,000 per primary
   dataset, audit sheets checked by eye; then the Stage 4 sweep and Stage 5 stress tests.
4. Download MVTec AD for Stage 2.
5. Drift in the experiments is synthetic; the guarantees are marginal, not per part.
6. NEU-DET accuracy 0.733 mAP@0.5 (first study) vs 83.14% (DSAT) and 83.03% (SH-DETR).
7. `src/risk/stratified.py` backs a reported negative result but is not run by any script;
   it is kept so that result stays reproducible.
8. Before submission: Scopus / Web of Science novelty check (Section 15.2).
