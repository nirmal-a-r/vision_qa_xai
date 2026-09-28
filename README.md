# Certified Cold-Start Defect Inspection (SPERC)

**Can a factory trust an AI inspector on day one, when it has seen only a handful of real defects?**
This project gives a mathematically backed answer: a *certificate* that says how many defective
parts the AI may let through, and an honest *refusal* when that cannot be guaranteed.

The full plan - problem, research questions, method, protocol, venues - is the project document
[`docs/Project_Document.pdf`](docs/Project_Document.pdf). This repository implements every part of
it; section numbers below refer to that document.

---

## 1. The problem in plain words

A camera photographs every part on a production line. An AI **detector** draws boxes around
defects and gives each box a **confidence score**. A part is **auto-accepted** when no box scores
above a **threshold**, otherwise it goes to a **human inspector**.

The dangerous mistake is a **defect escape**: a defective part that is auto-accepted. A plant wants
a promise like *"at most 5% of defective parts will escape"*. **Conformal risk control** turns a
detector's scores into exactly that promise, but with `m` real defect images it cannot promise
anything below `1/(m+1)`:

| real defect images `m` | best escape rate it can promise |
|---:|---:|
| 5 | 16.7% |
| 15 | 6.25% |
| 19 | 5.0% |
| 99 | 1.0% |

A **new** line has almost no defect images yet - exactly when a guarantee is needed most. This is
the **cold-start problem** (Section 2).

Two escape events are used, both 0/1 per defective image, reported separately and never mixed
(Section 2.2, `src/risk/escape.py`): **part escape** (no detection clears the threshold - what
reaches the customer; primary) and **localised escape** (some defect has no matching detection).

## 2. The idea: SPERC

**SPERC (Synthetic-Powered Escape-Risk Certification)** adds many **synthetic** defect images,
made by a few-shot generator from 3-5 real training examples, and calibrates with the exact
rank-coupled transporter of *Synthetic-Powered Predictive Inference* (Bashari et al., NeurIPS 2025),
which uses real data only through its *ranks*. It reports **two numbers**, because the mathematics
gives two (Sections 6 and 8):

| | What it says | When it holds |
|---|---|---|
| **Tier N** (near-nominal) | escape at most `alpha + beta` (+ a gap `eps`) | when synthetic defects look like real ones to the detector |
| **Tier H** (hard cap) | escape at most a fixed cap, e.g. 18.75% at `m = 15` | **always**, however bad the synthetic data is |
| **Refusal** | "send every part to a human" | when no setting meets the plant's maximum tolerable escape |

Tier H depends only on `m`, `N`, `alpha` and `beta`, so a plant can look it up **before collecting
any data** (`scripts/commissioning_table.py` reproduces the Section 8.4 table exactly). Synthetic
data can sharpen the operating point when it is good; it cannot sharpen the distribution-free worst
case below the real-data grid (C3). No new theorem is claimed.

```
 raw images ──► train / cal / test ──► detector (RT-DETR-L, YOLOv8s) ──► escape scores (real m)
                     │                                                          │
                     │ 3-5 train defects                                        ▼
                     └──► generator ──► synthetic defects ──► escape scores (synthetic N)
                                                                                │
                                          SPERC (transporter + beta) ◄──────────┘
                                                   │
                                  Tier H, Tier N  or  refuse ──► routing + drift monitor
```

## 3. Start working (Windows 11, RTX 5060, the `vqaenv` virtual environment)

Everything below runs from the repository folder with the venv's own Python. Nothing else needs
to be installed if `vqaenv` is already set up; `scripts\check_ready.py` says what, if anything, is
missing.

**The easy way: double-click, in order** (`scripts\windows\`; each one writes a log in `runs\`):

| | File | What it does | Time on an RTX 5060 |
|---|---|---|---|
| 1 | `1_check_setup.bat` | installs the extras, registers the notebook kernel, checks GPU / VRAM / RAM / long paths / power / datasets, runs the 61 tests | minutes |
| 2 | `2_smoke_test.bat` | every pipeline stage on a tiny copy of the data (in `runs_smoke\`) | ~15 min |
| 3 | `3_run_full_pipeline.bat` | Stage 3 onwards: both detectors x 5 datasets x 3 seeds, caches, Stage 1, sweeps, summary, notebook | 1-2 days |
| 4 | `4_generators_and_p2.bat` | Stages 4-6: synthetic defects, scoring, headline sweep + stress tests, P2 | 1-2 days |
| 5 | `5_run_notebook.bat` | rebuilds and runs the notebook, opens it in VS Code | minutes |

All five are resumable: if one stops (crash, reboot, power cut), double-click it again and it
continues where it stopped. No `activate` step is needed - they call `vqaenv\Scripts\python.exe`
directly, so PowerShell's script-execution policy does not matter.

The same steps as commands:
```bat
cd C:\Users\nirma\Desktop\vision_qa_xai
vqaenv\Scripts\activate

:: 1. is the machine ready? (packages, CUDA + sm_120, disk, all five datasets, split allocation)
python scripts\check_ready.py

:: 2. one-time: register the notebook kernel and add the generator extras
python -m ipykernel install --user --name vision_qa_xai --display-name "Python (vision_qa_xai)"
python -m pip install -r requirements.txt

:: 3. correctness tests (61), no GPU needed
python -m pytest -q tests

:: 4. ten-minute end-to-end check of EVERY stage on a tiny data copy (runs_smoke\, git-ignored)
python scripts\smoke_test.py

:: 5. the real run (see Section 4) - start it and leave it
python scripts\run_pipeline.py --wait-for-gpu
```

**The notebook.** Open `notebook\VisionQA_CertifiedInspection.ipynb` in VS Code or Jupyter, select
the `vqaenv` interpreter (or the "Python (vision_qa_xai)" kernel) and *Run All*. It follows the
project document section by section, runs Stage 0 itself, and reads every other result from `runs\`;
a cell whose input does not exist yet prints the one command that produces it.
`python scripts\execute_notebook.py` runs it headless and saves an executed copy.

**If the venv is ever rebuilt.** Python 3.10 or 3.11, then PyTorch for CUDA 12.8 or newer (the
RTX 5060 is Blackwell, `sm_120`):
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`, then
`pip install -r requirements.txt`. Check with
`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"` - the list
must contain `sm_120`.

**Windows / 8 GB optimisations (Sections 10.3-10.4, `configs\config.yaml` -> `training`).**
* Batch sizes follow the document's 8 GB profile automatically on a GPU under 12 GB (YOLOv8s 8 / 4 / 2
  at imgsz <= 384 / <= 640 / above; RT-DETR-L half). Gradients still accumulate to a nominal batch of
  64, so the optimisation is unchanged. An out-of-memory error retries the run at half the batch; the
  batch actually used is recorded in `runs\results_*.json`.
* Every training run is its own process, so a crash or leaked VRAM cannot take down the other runs;
  each starts from a clean CUDA context.
* DataLoader `workers=0` (Windows re-imports scripts in workers) with a RAM image cache (`cache: ram`),
  which removes most of the speed cost; detections are cached in fp16.
* The PC is kept awake while a pipeline, smoke test or generator runs (no power setting is changed);
  `check_ready.py` warns when the laptop is on battery.
* `--wait-for-gpu` waits for enough FREE VRAM (default 5.5 GB), because Chrome or VS Code can hold part
  of the card indefinitely.
* Logs are UTF-8 (Ultralytics prints emoji, which crash a cp1252 log); caches are written
  write-then-rename, so an interrupted run never leaves a half-written file.
* Keep `data\` and `runs\` out of OneDrive (checked), enable long paths if a path error appears
  (checked), and a Windows Defender exclusion for the project folder speeds up image reading.
* AnomalyDiffusion targets Ubuntu: run only that step under WSL2 with the same GPU (checked).

## 4. The run, stage by stage (Section 12)

| Stage | What | Command | GPU |
|---|---|---|---|
| 0 | SPI on synthetic Beta scores; adversarial synthetic data | `python -m pytest -q tests\test_spi_exact.py` | none |
| 1 | best case: held-out real GC10 defects as synthetic, m in {5,10,15,25}; **go / no-go** | part of `run_pipeline.py`, or `python scripts\run_stage1.py` | none (cached scores) |
| 2 | MVTec AD bridge: image-level anomaly scores | `python scripts\run_mvtec_bridge.py` (after downloading MVTec AD) | low |
| 3 | retrain with defect-aware splits, 3 seeds, primary datasets first | `python scripts\run_pipeline.py --wait-for-gpu` | high (overnight) |
| 4 | P1 sweep on KolektorSDD2 and Magnetic-Tile with both generators | `python scripts\run_pipeline.py --skip-train --generate training_free` | medium (generation) |
| 5 | misalignment stress: another line's defects, uniform noise, under-trained generator | automatic in every sweep | none |
| 6 | P2 full cold start on the two primary datasets | `python scripts\run_pipeline.py --skip-train --p2` | high |

`run_pipeline.py` runs, in order: prepare (raw -> COCO -> train / val / cal / test -> YOLO trees),
train (RT-DETR-L and YOLOv8s, seeds 0-2, primary datasets first), cache detections, CRC / drift /
triage analysis, synthetic generation and scoring by **every** detector and seed, the commissioning
table, Stage 1, the cold-start sweep for every model x seed, the pooled summary (RQ1-RQ4), P2 and
the MVTec bridge when requested / available, the seed summary, and the notebook. Every stage skips
finished work, so an interrupted run is simply restarted with the same command.

Useful switches: `--datasets kolektor,magnetic_tile` (primary only), `--seeds 0`, `--models rtdetr-l`,
`--skip-train`, `--generate copy_paste,training_free`, `--p2`, `--with-cce` (ablation), `--device cpu`.

**Where results land.** `runs\commissioning_table.csv`; `runs\stage1\stage1_verdict.json`;
`runs\coldstart\coldstart_<detector>.json|csv` (one row per dataset x escape event x source x m x
method x beta), `_mstar.csv`, `_alignment.csv`; `runs\coldstart\summary\` (seeds pooled: m-star,
validity, planned-vs-realised caps, alignment-vs-m-star); `runs\coldstart_p2\`; `runs\mvtec\`;
`runs\risk_results.json`, `runs\seed_summary.json`.

**What the sweep compares** on the same random draws (Section 12.1): CRC on real defects only; naive
pooling (invalid, kept to show why ranks matter); the old `spi.py` heuristic; defect-seeking
acquisition without correction; the oracle (CRC on the full pool); SPERC at beta in
{0.05, 0.10, 0.20}, run so that Tier N equals the target and at the nominal level; and CRC at
SPERC's Tier H cap (the same hard guarantee, C3). Metrics (Section 12.2): realised escape with
standard error, 95% interval and Wilson interval; issuance rate; review load; parts labelled;
m-star; KS alignment; detector mAP@0.5 and mAP@0.5:0.95 over 3 seeds for context. Refusals count
as refusals, never dropped; ties are broken at random.

## 5. Datasets (Section 9)

Download each dataset from its official page and unpack it under `data\raw\` with these names:

| Dataset | Folder | Role | Source paper |
|---|---|---|---|
| KolektorSDD2 | `data\raw\KOLEKTORSDD2\` (`train\`, `test\`) | **primary** | Božič et al., *Computers in Industry* 2021 |
| Magnetic-Tile | `data\raw\MAGNETIC-TILE\` (`MT_*`) | **primary replication** | Huang et al., *The Visual Computer* 2020 |
| NEU-DET | `data\raw\NEU-DET\` | validity check | He et al., *IEEE TIM* 2020 |
| GC10-DET | `data\raw\GC10-DET\` | validity check (and Stage 1) | Lv et al., *Sensors* 2020 |
| PKU PCB defects | `data\raw\PCB-DEFECTS\PCB_DATASET\` | validity check (defects are artificial) | Huang & Wei, arXiv:1901.08204 |
| MVTec AD | `data\raw\MVTEC-AD\<category>\` | Stage-2 bridge (optional) | Bergmann et al., *IJCV* 2021 |

Only the two primary datasets contain clean parts, so only they can measure review load. DAGM 2007
(`data\raw\DAGM2007\`) has a converter too but is not part of the plan's five; add `dagm` to
`--datasets` to use it as an extra check.

Splits (Section 9.1, `configs\config.yaml`): train / val / cal / test. The detector trains on
`train` and selects its checkpoint on `val` (carved from train); `cal` is never shown to training in
any way; `test` only checks the guarantee. Defect-aware allocation: defective images 45/30/25, clean
60/15/25, which roughly halves every alpha floor.

**Old results.** Detectors and caches of the first study (trained before the split fix) are in
`archive\` for reference only; never copy them into `runs\`. `_old_splits_DELETE_ME\` holds data
folders built with the old splits and can be deleted in File Explorer.

## 6. Synthetic defects (Sections 7, 11.3)

References (3-5 real defects) and clean backgrounds come from the **training split only**
(`src\synth\references.py` enforces it and refuses any file that sits in val / cal / test).

```bat
:: in-repo generators, N = 1000 per primary dataset (config generators.n_synthetic)
python scripts\generate_synthetic.py --dataset kolektor --generator training_free
python scripts\generate_synthetic.py --dataset kolektor --generator copy_paste

:: AnomalyDiffusion (AAAI 2024; fine-tuned few-shot arm; Ubuntu / WSL2)
python scripts\generate_synthetic.py --dataset kolektor --export-anomalydiffusion   :: prints the WSL commands
python scripts\generate_synthetic.py --dataset kolektor --collect <its output> --source anomalydiffusion

:: official TF-IDG (ICCV 2025) instead of the in-repo training-free generator
python scripts\generate_synthetic.py --dataset kolektor --export-tfidg
python scripts\generate_synthetic.py --dataset kolektor --collect <its output> --source training_free
```

* `training_free` (default) is a training-free, one-shot diffusion generator in the spirit of TF-IDG
  (reference placed with an adaptive mask, the region cropped and enlarged, re-synthesised by Stable
  Diffusion 2 inpainting at partial strength, blended back so the background is preserved). It is
  not the official TF-IDG code; the paper must name whichever was used. First use downloads the
  model (~5 GB) from Hugging Face; fp16 fits the 8 GB card.
* `copy_paste` needs no model: a real reference defect, randomly transformed, Poisson-blended onto a
  clean part. A low-fidelity reference point for RQ3 and the smoke-test generator.
* Each set gets `generator.json` (provenance) and `_audit_sheet.png` - look at it before using it.
* For Stage 5, an under-trained AnomalyDiffusion checkpoint is collected as source
  `anomalydiffusion_undertrained` (listed in `stress_generators` in the config).

## 7. What is in the folder

```
configs/config.yaml        every experiment setting (datasets, splits, seeds, sweep, generators, MVTec)
docs/Project_Document.pdf  the complete project document (the plan this code implements)
docs/PROJECT_STATUS.md     what is measured so far, with numbers and known limits
docs/RESEARCH_PLAN.md      the plan in short, next to the code
experiments/               write-ups of individual experiments, including failed ideas
figures/                   figures (fig15 = the SPERC system of Section 7)
notebook/                  the notebook that follows the document section by section
scripts/
  windows/1_..5_*.bat      double-click launchers (setup, smoke test, full run, generators + P2, notebook)
  check_ready.py           pre-flight check: packages, GPU, VRAM, RAM, power, long paths, datasets, splits
  smoke_test.py            every pipeline stage on a tiny data copy, before the long run
  run_pipeline.py          everything, end to end, resumable
  commissioning_table.py   Section 8.4: hard escape cap vs number of real defects (no data)
  run_stage1.py            Stage 1 best case and the go / no-go decision
  run_coldstart_sweep.py   Stages 4-5: the headline sweep, every model x seed
  summarize_sperc.py       seeds pooled; RQ1-RQ4 tables
  run_mvtec_bridge.py      Stage 2: MVTec AD bridge
  generate_synthetic.py    synthetic defects (in-repo generators, AnomalyDiffusion, TF-IDG)
  build_certified_notebook.py, execute_notebook.py   build / run the notebook
  demo_sperc.py            one-minute simulation, no data
  compute_faithfulness.py, run_rigor_audit.py, find_dead_code.py, nb_repair.py   diagnostics
src/
  risk/spi_exact.py        exact SPI: rank law, windows, transporter, Theorems 3.3 / 3.5, Algorithm 4
  risk/sperc.py            SPERC certificate: Tier H / Tier N / refusal, commissioning table
  risk/spi.py              the name the document uses; re-exports escape scoring + exact SPI
  risk/escape.py           the two escape events, exact CRC threshold, random tie-breaking
  risk/baselines.py        naive pooling and the old heuristic (baselines only)
  risk/conformal.py        CRC, Learn-then-Test, per-class (Mondrian) control
  risk/adaptive.py         drift monitor and online recalibration
  risk/triage.py           accept / review / reject policy (escape certified per defective part)
  risk/acquisition.py, stratified.py   defect-seeking labelling (and refuted corrections)
  synth/                   references (train split only), generators, external wrappers, compositing
  data/                    dataset converters, splits, YOLO export, P2 cold-start trees
  evaluation/              training, detection caching, experiments, anomaly scores, seed statistics
  utils/winenv.py          Windows helpers: UTF-8 logs, keep-awake, power / long-path / VRAM checks
  xai/, agentic/           saliency audit and the closed-loop agent (outside the paper's core claim)
tests/                     61 tests; each checks a property the theory or the protocol requires
```

## 8. Reading a certificate

```python
from src.risk.escape import escape_scores
from src.risk.sperc import certify_sperc

cert = certify_sperc(c_real, c_syn, alpha=0.05, beta=0.05, alpha_max=0.25)
cert.issued        # False means refuse: send every part to review
cert.threshold     # auto-accept a part only if its highest detection score is below this
cert.tier_n_bound  # 0.10 : escape bound when synthetic defects resemble real ones (+ eps)
cert.tier_h_cap    # e.g. 0.1875 : escape bound that holds no matter what
```

Both bounds are averages over the choice of calibration images, like every conformal guarantee;
neither is a promise about one individual part. They also assume the line has not changed since
calibration; the drift monitor in `src/risk/adaptive.py` watches for that.

## 9. Status and honest limits

See `docs/PROJECT_STATUS.md`. In short: the method is implemented and verified (Stage 0 passes; the
Section 8.4 table is reproduced exactly); the first-study detectors give an encouraging Stage-1
preview; the certified results need the retrained detectors (Stage 3) and the generators. Detection
accuracy on NEU-DET (0.733 mAP@0.5 in the first study) is below the published best (83.14% DSAT,
83.03% SH-DETR); the contribution is measured in real defects needed, not mAP.

## 10. Key references

- Bashari, Lotan, Lee, Dobriban & Romano. *Synthetic-Powered Predictive Inference.* NeurIPS 2025.
- Angelopoulos, Bates, Fisch, Lei & Schuster. *Conformal Risk Control.* ICLR 2024.
- Zhao et al. *DETRs Beat YOLOs on Real-time Object Detection.* CVPR 2024.
- Hu et al. *AnomalyDiffusion.* AAAI 2024. Xu et al. *Training-Free Industrial Defect Generation.* ICCV 2025.

The full, verified reference list is in `docs/Project_Document.pdf` (Section 16) and `docs/RESEARCH_PLAN.md`.
