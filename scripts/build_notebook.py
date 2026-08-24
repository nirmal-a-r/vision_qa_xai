"""
build_notebook.py
=================
Generates the single deliverable notebook.

Generated rather than hand-edited so that it can be regenerated from source
after any module changes, which is what stops the notebook and `src/` drifting
apart - the failure mode that makes most research notebooks unreproducible a
month after submission.
"""

import json, io, os

NB = "notebook/VisionQA_RiskControlled_Inspection.ipynb"
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})


# ===========================================================================
md(r"""
# Risk-Controlled Explainable Defect Inspection

**Certified defect-escape guarantees with faithfulness-gated human triage, across five industrial datasets.**

---

## The problem

A deployed inspection model outputs a confidence score. Nobody can say what that
number means. "0.7" is not a probability of anything, it is not comparable
between two models, and it is not comparable between Tuesday and Thursday on the
same model. So the score threshold on a real production line is set by a
technician turning a dial until the reject pile looks about right.

That is the actual state of practice, and it is why a plant cannot answer the
only question that matters to its customer: **what fraction of defective parts
do we ship?**

## The claim

Given a held-out calibration set that is exchangeable with production, we
produce an operating point with a finite-sample, distribution-free guarantee

$$\mathbb{E}\big[\text{escape rate}\big] \;\le\; \alpha$$

that holds for **any** detector, **any** dataset size, and requires **no**
distributional assumption. When the target $\alpha$ is not achievable by the
detector, the system **refuses to issue a certificate** instead of quietly
returning a threshold that cannot deliver it.

## Contributions

| | contribution | where |
|---|---|---|
| **C1** | Instance-level escape risk control with per-class severity budgets (Mondrian) | §5, §6 |
| **C2** | Faithfulness-gated triage: explanation quality as a *certified gating variable*, minimising human review load subject to the escape guarantee | §8, §9 |
| **C3** | Drift-aware recalibration under production distribution shift | §7 |
| **C4** | Complementary Channel Encoding (CCE) — a zero-cost input encoding, reported with an honest ablation | §4 |

## How to run

Select the **`vqaenv`** kernel and run top to bottom. Every cell prints its
inputs and outputs; every processing stage is rendered before and after.
Figures are written to `figures/` as 300-dpi PDF and PNG.
""")

# --------------------------------------------------------------- 0 setup
md("## 0 · Environment, configuration, reproducibility")
code(r"""
import os, sys, json, glob, warnings, platform, random
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".." if os.path.basename(os.getcwd()) == "notebook" else "."))
ROOT = os.path.abspath(".." if os.path.basename(os.getcwd()) == "notebook" else ".")
os.chdir(ROOT)

import numpy as np, cv2, torch, matplotlib.pyplot as plt
from src.paper import figures as F
F.set_style()

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"root        : {ROOT}")
print(f"python      : {platform.python_version()}")
print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu         : {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
print(f"opencv      : {cv2.__version__}")
print(f"seed        : {SEED}")

DATASETS = ["neu", "gc10", "pcb", "magnetic_tile", "kolektor"]
ALPHAS   = [0.01, 0.05, 0.10, 0.20]
os.makedirs("figures", exist_ok=True)
""")

md(r"""
### System architecture

Each block is a section below. The calibration block is disjoint from training —
that disjointness is what makes the guarantee valid rather than decorative.
""")
code(r"""
fig = F.architecture_diagram(); F.save(fig, "fig01_architecture"); plt.show()
""")

# --------------------------------------------------------------- 1 data
md(r"""
## 1 · Datasets

Five public industrial datasets spanning five different manufacturing domains.

Two of them (Magnetic-Tile, KolektorSDD2) are **load-bearing for a specific
reason**: they are the only ones containing genuine defect-free parts. NEU, GC10
and PCB have a defect in *every* image, so on those the auto-accept branch of a
triage policy can never be correct and the escape/review trade-off is not
measurable at all.
""")
code(r"""
import json
rows = []
for name in DATASETS:
    d = json.load(open(f"data/processed/{name}_coco.json"))
    neg = sum(1 for i in d["images"] if i.get("n_boxes", 1) == 0)
    ws = [a["bbox"][2] / next(im["width"] for im in d["images"] if im["id"] == a["image_id"])
          for a in d["annotations"][:400]]
    rows.append(dict(dataset=name, images=len(d["images"]), boxes=len(d["annotations"]),
                     classes=len(d["categories"]), defect_free=neg,
                     mean_box_w_pct=round(100 * float(np.mean(ws)), 1)))
import pandas as pd
df_data = pd.DataFrame(rows)
print(df_data.to_string(index=False))
print(f"\nTOTAL: {df_data.images.sum()} images, {df_data.boxes.sum()} boxes, "
      f"{df_data.classes.sum()} classes, {df_data.defect_free.sum()} defect-free")
""")

md("### Sample images with ground truth, one row per dataset")
code(r"""
fig, axes = plt.subplots(len(DATASETS), 4, figsize=(11, 2.5 * len(DATASETS)))
for r, name in enumerate(DATASETS):
    d = json.load(open(f"data/processed/{name}_coco.json"))
    imgs = {i["id"]: i for i in d["images"]}
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a["bbox"])
    ids = list(by)[:4]
    for c in range(4):
        ax = axes[r, c]; ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if c < len(ids):
            im = cv2.cvtColor(cv2.imread(imgs[ids[c]]["file_name"]), cv2.COLOR_BGR2RGB)
            ax.imshow(im)
            for x, y, w, h in by[ids[c]]:
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=F.OKABE[1], lw=1.5))
        if c == 0:
            ax.set_ylabel(name, fontsize=10, rotation=0, ha="right", va="center", labelpad=42)
fig.suptitle("Ground-truth annotations across the five domains", y=1.0)
fig.tight_layout(); F.save(fig, "fig02_dataset_samples"); plt.show()
""")

md("### Class distribution and defect scale")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
counts = {n: json.load(open(f"data/processed/{n}_coco.json")) for n in DATASETS}
axes[0].bar(DATASETS, [len(counts[n]["images"]) for n in DATASETS], color=F.OKABE[0], label="images")
axes[0].bar(DATASETS, [sum(1 for i in counts[n]["images"] if i.get("n_boxes",1)==0) for n in DATASETS],
            color=F.OKABE[2], label="defect-free")
axes[0].set_ylabel("count"); axes[0].set_title("Dataset size and negative availability"); axes[0].legend()

for i, n in enumerate(DATASETS):
    d = counts[n]; imgs = {im["id"]: im for im in d["images"]}
    fr = [100*a["bbox"][2]*a["bbox"][3]/(imgs[a["image_id"]]["width"]*imgs[a["image_id"]]["height"])
          for a in d["annotations"]]
    axes[1].hist(fr, bins=40, range=(0, 40), histtype="step", label=n, color=F.OKABE[i])
axes[1].set_xlabel("defect area (% of frame)"); axes[1].set_ylabel("boxes")
axes[1].set_title("Defect scale — PCB is a small-object regime"); axes[1].legend(fontsize=8)
fig.tight_layout(); F.save(fig, "fig03_dataset_stats"); plt.show()
""")

md(r"""
## 2 · Splits — why calibration is held out

Three disjoint blocks: **train** (60%), **cal** (15%), **test** (25%), stratified
by each image's rarest class so that rare defects and defect-free parts appear in
all three.

The calibration block is the load-bearing detail. Conformal risk control requires
calibration data that is exchangeable with test data **and untouched by model
fitting**. Reusing a validation set that early stopping or checkpoint selection
looked at breaks exchangeability and silently voids the guarantee — the numbers
would still print, they would simply be wrong.
""")
code(r"""
rows = []
for n in DATASETS:
    for s in ["train", "cal", "test"]:
        p = f"data/processed/splits/{n}_{s}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            rows.append(dict(dataset=n, split=s, images=len(d["images"]),
                             boxes=len(d["annotations"]),
                             clean=sum(1 for i in d["images"] if i.get("n_boxes",1)==0)))
df_split = pd.DataFrame(rows)
print(df_split.pivot(index="dataset", columns="split", values="images").to_string())
""")

json.dump({"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python (vqaenv)", "language": "python", "name": "vqaenv"},
    "language_info": {"name": "python", "version": "3.10.11"}},
    "nbformat": 4, "nbformat_minor": 5},
    io.open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"part 1 written: {len(cells)} cells")
