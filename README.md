# Risk-Controlled Explainable Defect Inspection

**Faithfulness-gated, drift-aware conformal risk control for industrial visual inspection.**

An inspection system should not report a confidence score. It should report a
**guarantee**. Given a target defect escape rate `α`, this system either

1. returns an operating point whose escape rate is **certified** at or below `α`
   — finite-sample, distribution-free — at the lowest human review load it can
   prove is safe; or
2. **refuses**, because no such operating point exists for this detector and this
   `α`, and hands the line back to full human review.

The refusal branch matters as much as the first. A method that always answers is
a method that sometimes lies.

---

## Contributions

| | Contribution | Gap it fills |
|---|---|---|
| **C1** | Instance-level escape risk with per-class severity budgets (Mondrian) | Prior conformal work on surface defects ([arXiv:2504.17721](https://arxiv.org/abs/2504.17721), 2025) controls *pixel* FDR/FNR under a single global budget |
| **C2** | Faithfulness-gated triage — explanation quality as a certified gating variable | No prior work uses explanation faithfulness as a control signal under a risk guarantee |
| **C3** | Drift-aware recalibration with a long-run bound valid for adversarial sequences | Named explicitly as open future work in arXiv:2504.17721 |

Validated on **5 industrial datasets** (9,468 images, 11,543 boxes, 28 classes)
across **2 detector families**.

## Headline results

* **12/12** dataset × α combinations satisfy the escape guarantee on held-out test data.
* After an abrupt regime shift a **static** conformal threshold degrades to a
  **0.9618 escape rate — 9.6× its 0.10 target**; the adaptive controller holds **0.1007**.
* Drift monitor: **0** false alarms over 1,500 exchangeable parts, 122-part detection latency.
* Audited faithfulness (25 NEU test images, occlusion saliency): insertion AUC
  **0.753**, energy pointing **0.561** against a **0.302** chance level (1.86×).
* Certificate-targeted active learning: **2.81×** defect lift in the top-50 acquisitions.

## Quick start

```bash
# environment (CUDA 12.8 build of torch, cloned from a known-good env)
./vqaenv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"

# property tests - if these fail nothing downstream means anything
./vqaenv/Scripts/python.exe tests/test_conformal.py     # 9/9
./vqaenv/Scripts/python.exe tests/test_adaptive.py      # 6/6
```

### Run the notebook

```bash
# one-time: register the project kernel so the notebook has one to select
./vqaenv/Scripts/python.exe -m ipykernel install --user     --name vision_qa_xai --display-name "Python (vision_qa_xai)"
```

Then open **`notebook/VisionQA_RiskControlled_Inspection.ipynb`**, select kernel
**Python (vision_qa_xai)**, and Run All. It reads cached artefacts from `runs/`
and `figures/`, so a full pass takes minutes and needs no GPU.

### Or regenerate everything from raw data

```bash
python scripts/run_pipeline.py                # datasets -> training -> analysis -> notebook
python scripts/run_pipeline.py --skip-train   # reuse existing weights
python scripts/run_pipeline.py --wait-for-gpu # block until the GPU is free first
```

Every stage is resumable, so an interrupted sweep restarts where it stopped
rather than from the beginning.

## Reproducing from scratch

`scripts/run_pipeline.py` chains these; run them individually to inspect a stage.

```bash
# 1. datasets -> COCO -> stratified train/cal/test -> YOLO layout
python -m src.data.prepare_datasets --raw_dir data/raw --out_dir data/processed
python -m src.data.splits_and_yolo

# 2. Complementary Channel Encoding copies of the trees
python -m src.data.build_cce_dataset

# 3. detectors (baseline and CCE arms)
python -m src.evaluation.train_baselines --model rtdetr-l.pt
python -m src.evaluation.train_baselines --model yolov8s.pt
python -m src.evaluation.train_baselines --model yolov8s.pt     --encoding cce --yolo_dir data/yolo_cce

# 4. cache predictions once, then every analysis reads that cache
python scripts/run_all_experiments.py
python scripts/compute_faithfulness.py 25 neu

# 5. rebuild the notebook from source
python -m src.paper.build_notebook
```

### Repository hygiene

```bash
python scripts/find_dead_code.py    # import-graph reachability over src/
```

The legacy Deformable-DETR pipeline this project started from was removed once
the reachability check showed nothing reached it; it remains in git history.

## Layout

```
src/
  risk/
    conformal.py       CRC, Learn-then-Test (Hoeffding-Bentkus), Mondrian, escape loss
    adaptive.py        online risk control under drift + KS drift monitor
    triage.py          AUTO_ACCEPT / HUMAN_REVIEW / AUTO_REJECT under LTT certification
  xai/
    faithfulness.py    insertion/deletion AUC, pointing game, energy pointing, sanity check
    detector_saliency.py  occlusion saliency + audited faithfulness (black-box)
    rollout.py         Swin attention rollout, gradient relevance
  agentic/
    inspector.py       closed-loop agent, certificate-targeted active learning
  data/                dataset converters, stratified splits, YOLO export
  evaluation/          training, prediction caching, risk experiments
  paper/               figures and the notebook generator
tests/                 property tests for every guarantee claimed
notebook/              the single end-to-end notebook
figures/               PDF + PNG at 300 dpi
```

## Design decisions worth knowing

**Why RT-DETR and not YOLO as the main method.** RT-DETR is NMS-free. NMS is a
score-dependent filter whose behaviour changes with box density, and conformal
calibration is a statement about precisely the score distribution NMS reshapes.
An end-to-end set-prediction head lets calibration target the model's own output.
YOLOv8s is included as a baseline for the comparison table.

**Why a dedicated `cal` split.** CRC requires calibration data exchangeable with
test *and* untouched by model fitting. Reusing a validation set that early
stopping looked at breaks exchangeability and silently voids the guarantee.

**Why occlusion saliency and not Grad-CAM.** The faithfulness score gates the
triage policy, so it must be computable for any detector the framework wraps —
including a vendor binary with no accessible internals. Occlusion needs only
`predict()`.

**Why the agent is a control loop and not an LLM.** Every action it takes must be
one a certificate can license or refuse. An LLM in that seat produces confident
prose with no guarantee attached — the exact failure mode this work argues against.

## Known limitations

* **Exchangeability.** The split-conformal guarantee needs the calibration block
  to be exchangeable with production. The adaptive layer addresses drift, but its
  bound is long-run, not per-part.
* **Calibration size.** On the negative-heavy datasets only ~55 calibration images
  carry a defect, so the `B/(n+1)` penalty alone consumes roughly a third of an
  `α = 0.05` budget. Tight budgets on rare classes need more labelled defects —
  and the method says so by refusing rather than by quietly degrading.
* **Faithfulness cost.** The audited score is reported on a sample; the full
  policy sweep uses a confidence-margin proxy because occlusion saliency costs
  `G²` forward passes per image.
* **Benchmark comparability.** Published numbers quoted in the comparison figure
  were obtained on other authors' splits.
