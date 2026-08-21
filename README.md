# VisionQA — Phase 1 (Vision Core) + Phase 2 (XAI Engine): Kaggle build

Adapts `PROJECT.md`'s Weeks 1–7 scope to run end-to-end in a Kaggle notebook,
against:

- **PCB defects**: [`akhatova/pcb-defects`](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- **NEU surface defects**: [`kaustubhdikshit/neu-surface-defect-database`](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

## Fastest way to run this

Open `notebook/VisionQA_Phase1_Phase2.ipynb` in Kaggle (upload it, or copy its
cells into a fresh notebook), attach both datasets under **Add Data**, turn on
a GPU, and run top to bottom. The notebook writes the `src/` package shown
below into `/kaggle/working/src` itself — you don't need to separately upload
this repo, the notebook *is* the whole deliverable if you just want to click
run. If you'd rather work from the package directly (e.g. push it to your own
repo and `%cd`/`import` from Kaggle), the same files are laid out below in
`src/`.

## Directory layout

```
visionqa-kaggle/
├── configs/config.yaml          # single source of truth (paths, model, training, XAI)
├── src/
│   ├── data/
│   │   ├── neu_xml_to_coco.py   # NEU-DET PASCAL-VOC XML -> COCO  (run first)
│   │   ├── prepare_pcb_coco.py  # PCB defects -> COCO (Roboflow export OR raw XML)
│   │   ├── merge_coco.py        # unify label spaces, copy images, train/val/test split
│   │   ├── dataset.py           # torch Dataset -> HF DeformableDetr input format
│   │   └── transforms.py        # Albumentations train/val/synthetic-augmentation pipelines
│   ├── models/
│   │   └── detector.py          # Swin backbone + Deformable DETR (HF transformers)
│   ├── segmentation/
│   │   └── pseudo_mask.py       # GrabCut box->pixel-mask refinement + damage% calc
│   ├── xai/
│   │   ├── rollout.py           # Swin-aware attention rollout + gradient relevance (LRP-lite)
│   │   └── overlay.py           # heatmap + box + mask PNG rendering
│   ├── train.py                 # training loop (M2 milestone)
│   ├── evaluate.py              # COCO mAP eval (M2 milestone target: >95% mAP)
│   └── inference.py             # full pipeline -> PROJECT.md §1.5 handoff JSON (M3 milestone)
└── notebook/
    └── VisionQA_Phase1_Phase2.ipynb   # drives all of the above on Kaggle
```

## Three deliberate deviations from `PROJECT.md` — and why

**1. NEU-DET is PASCAL-VOC XML, not COCO.** `neu_xml_to_coco.py` walks the
dataset tree for `*.xml` files, matches each to its image by filename stem
(robust to the couple of different folder layouts this dataset has been
re-uploaded under on Kaggle), and writes one COCO json. `prepare_pcb_coco.py`
reuses the exact same converter for PCB defects if you're using the raw
Kaggle XML download rather than a Roboflow COCO export — the two datasets are
shaped identically on disk (VOC XML + images).

**2. Segmentation is GrabCut, not a trained Mask2Former.** Neither
`akhatova/pcb-defects` nor `kaustubhdikshit/neu-surface-defect-database` ships
polygon/pixel-level annotations — both are bounding-box only, on Kaggle and
upstream. `PROJECT.md`'s Mask2Former head needs mask ground truth to train
against; without it, "training" one would just teach it to reproduce
rectangles, at real implementation cost for no pixel-accuracy gain over the
box itself. `src/segmentation/pseudo_mask.py` instead runs OpenCV **GrabCut**,
initialized with each detected box as the foreground prior, to get a genuine
pixel-accurate mask with no training required — `damage_pct` is computed from
that real mask, matching `PROJECT.md`'s formula
(`defective_pixel_area / total_component_area × 100`). If you later get real
polygon labels (e.g. SAM-assisted labeling on a sample), swap this one module
for a trained Mask2Former head; nothing downstream (the overlay renderer, the
handoff JSON schema, the decision rule) needs to change.

**3. LRP is approximated with gradient×input, not hand-rolled attention-LRP.**
`PROJECT.md` itself notes standard LRP conservation rules aren't solved
off-the-shelf for softmax/attention layers (citing Chefer et al. 2021).
Rather than implement a bespoke, likely-fragile relevance-propagation rule set
for Deformable DETR's sparse deformable attention, `src/xai/rollout.py` uses
Captum's `InputXGradient` — a robust, well-understood per-pixel relevance
method — as the practical stand-in. Swin's own windowed self-attention *does*
get a real, window-aware attention-rollout implementation (with the window-
merging handling `PROJECT.md` calls out as necessary), and the two are fused
for the final heatmap overlay.

## Setup

```
pip install -r requirements.txt
```
(Kaggle notebooks already have `torch`/`torchvision`/`opencv`/`matplotlib`
preinstalled — this file only lists what needs an explicit install there.)

## Config

Everything path/model/training/XAI-related lives in `configs/config.yaml`.
The two things you'll actually need to change per Kaggle session are
`paths.pcb_raw_dir` / `paths.neu_raw_dir` (match whatever "Add Data" mounted
your datasets as) and `data.pcb_format` (`"voc"` for the raw Kaggle download,
`"coco"` if you've re-exported through Roboflow).

## Not included here (deliberately out of scope for Phase 1/2)

- TensorRT/INT8 export and the >30 FPS RTX 4070/4090 benchmark (Objective 5)
  — needs the actual target workstation to validate against.
- Phase 3 (agentic reasoning) / Phase 4 (active learning) — Shruhath's half.
