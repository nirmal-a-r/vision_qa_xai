"""Appends sections 3-11 to the notebook started by build_notebook.py."""
import json, io

NB = "notebook/VisionQA_RiskControlled_Inspection.ipynb"
nb = json.load(io.open(NB, encoding="utf-8"))
cells = nb["cells"]


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})


# ------------------------------------------------------- 3 photometric
md(r"""
## 3 · Photometric analysis — what the sensor actually gives us

Before any modelling, a fact that determines the input design.
""")
code(r"""
from src.data.photometric import to_gray
print("Are these images colour, or grey replicated into three channels?\n")
print(f"{'dataset':16s} {'shape':18s} {'ch0==ch1':>9s} {'ch1==ch2':>9s}  verdict")
print("-" * 74)
CHANNEL_REDUNDANT = {}
for name in DATASETS:
    d = json.load(open(f"data/processed/{name}_coco.json"))
    s01 = s12 = n = 0; shp = ""
    for info in d["images"][:60]:
        im = cv2.imread(info["file_name"], cv2.IMREAD_UNCHANGED)
        if im is None: continue
        shp = str(im.shape); n += 1
        if im.ndim == 2: s01 += 1; s12 += 1
        else:
            s01 += int(np.array_equal(im[...,0], im[...,1]))
            s12 += int(np.array_equal(im[...,1], im[...,2]))
    red = (s01 == n and s12 == n); CHANNEL_REDUNDANT[name] = red
    print(f"{name:16s} {shp:18s} {s01}/{n:<7d} {s12}/{n:<7d}  "
          f"{'GREY x3 — 2 channels are dead weight' if red else 'genuine colour'}")
print(f"\n{sum(CHANNEL_REDUNDANT.values())} of {len(DATASETS)} datasets waste two thirds of every input tensor.")
""")

md(r"""
### Per-stage effect of each photometric operation

Every stage rendered before and after, with the intensity histogram that
explains *why* it moved.
""")
code(r"""
from src.data.photometric import (apply_clahe, high_frequency_residual,
                                  complementary_channels, local_contrast, entropy)
d = json.load(open("data/processed/neu_coco.json"))
imgs = {i["id"]: i for i in d["images"]}
by = {}
for a in d["annotations"]: by.setdefault(a["image_id"], []).append(a["bbox"])
iid = list(by)[7]
img = cv2.cvtColor(cv2.imread(imgs[iid]["file_name"]), cv2.COLOR_BGR2RGB)
boxes = [[x, y, x+w, y+h] for x, y, w, h in by[iid]]

enc, stages = complementary_channels(img, return_stages=True)
fig = F.image_strip(stages, title=f"CCE stages — {os.path.basename(imgs[iid]['file_name'])}", boxes=boxes)
F.save(fig, "fig04_cce_stages"); plt.show()

print(f"{'stage':26s} {'local contrast':>15s} {'entropy (bits)':>15s}")
for k, v in stages.items():
    print(f"{k:26s} {local_contrast(v):15.3f} {entropy(v):15.3f}")
""")

code(r"""
fig = F.histogram_panel({k: v for k, v in stages.items() if k != "04_cce_composite"},
                        "Intensity distribution at each CCE stage")
F.save(fig, "fig05_cce_histograms"); plt.show()
fig = F.cce_diagram(); F.save(fig, "fig06_cce_concept"); plt.show()
""")

md(r"""
## 4 · Complementary Channel Encoding (C4)

CLAHE **as a substitution** is a bad trade — measured below. CLAHE **as a
complement** is free. The dead channels get spent on representations the grey
channel does not already contain:

| channel | content | preserves |
|---|---|---|
| ch0 | raw intensity | intensity-domain separability, which CLAHE destroys |
| ch1 | CLAHE | local contrast / edge energy |
| ch2 | $I - G_\sigma * I$ | surface texture and roughness |

Measured with the **multivariate Fisher discriminant**

$$J = (\mu_d - \mu_b)^\top S_w^{-1} (\mu_d - \mu_b)$$

which accounts for inter-channel covariance — so a channel that merely
duplicates another contributes **nothing** to $J$. That property is exactly what
makes it the right test here.
""")
code(r"""
from src.data.photometric import multivariate_separability

def _rep3(im):   g = to_gray(im); return np.stack([g, g, g], -1)
def _clahe3(im): g = apply_clahe(to_gray(im), 2.0, 8); return np.stack([g, g, g], -1)

def _sample(name, k=40):
    d = json.load(open(f"data/processed/{name}_coco.json"))
    im_map = {i["id"]: i for i in d["images"]}; b = {}
    for a in d["annotations"]: b.setdefault(a["image_id"], []).append(a["bbox"])
    out = []
    for iid in sorted(b)[:k]:
        im = cv2.imread(im_map[iid]["file_name"])
        if im is None: continue
        out.append((cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                    [[x, y, x+w, y+h] for x, y, w, h in b[iid]]))
    return out

ENC = {"grey x3 (standard)": _rep3, "CLAHE x3 (naive)": _clahe3,
       "CCE [raw|clahe|hf]": lambda im: complementary_channels(im)}
sep = {e: [] for e in ENC}
for n in DATASETS:
    s = _sample(n)
    for e, fn in ENC.items():
        sep[e].append(float(np.mean([multivariate_separability(fn(im), b) for im, b in s])))

print(f"{'encoding':22s} " + " ".join(f"{n[:11]:>12s}" for n in DATASETS))
print("-" * 88)
base = sep["grey x3 (standard)"]
for e, vals in sep.items():
    if e.startswith("grey"): print(f"{e:22s} " + " ".join(f"{v:12.4f}" for v in vals))
    else: print(f"{e:22s} " + " ".join(f"{100*(v-b)/max(b,1e-9):+11.1f}%" for v, b in zip(vals, base)))

fig = F.grouped_bars(DATASETS, {e: v for e, v in sep.items()},
                     "Fisher separability $J$", "Input encoding — multivariate separability")
fig.axes[0].set_yscale("log")
F.save(fig, "fig07_cce_separability"); plt.show()
""")

md(r"""
### Ablation — does the separability gain survive into detection accuracy?

Separability is a **proxy**. Only a training ablation settles it, and the result
is reported here whatever it says.
""")
code(r"""
res = []
for f in ["runs/results_yolov8s.json", "runs/results_rtdetr.json"]:
    if os.path.exists(f): res += json.load(open(f))
key = lambda r: (r["model"], r["dataset"])
base_r = {key(r): r for r in res if r.get("encoding", "baseline") == "baseline"}
cce_r  = {key(r): r for r in res if r.get("encoding") == "cce"}

rows = []
for k in sorted(set(base_r) & set(cce_r)):
    b, c = base_r[k], cce_r[k]
    rows.append(dict(model=k[0].replace(".pt",""), dataset=k[1],
                     baseline=round(b["test_mAP50"], 4), cce=round(c["test_mAP50"], 4),
                     delta_pp=round(100*(c["test_mAP50"]-b["test_mAP50"]), 2)))
df_cce = pd.DataFrame(rows)
if len(df_cce):
    print(df_cce.to_string(index=False))
    print(f"\nCCE better on {int((df_cce.delta_pp>0).sum())}/{len(df_cce)} paired comparisons; "
          f"mean delta {df_cce.delta_pp.mean():+.2f} pp")
    fig = F.grouped_bars(list(df_cce.dataset), {"baseline": list(df_cce.baseline), "CCE": list(df_cce.cce)},
                         "test mAP@0.5", "CCE ablation (paired, same model/dataset/seed)")
    F.save(fig, "fig08_cce_ablation"); plt.show()
else:
    print("no completed CCE pairs yet")
""")

# ------------------------------------------------------- 5 detectors
md(r"""
## 5 · Detectors

**RT-DETR is the main method; YOLOv8s is a comparison baseline only.**

The choice is not about leaderboard position. RT-DETR is **NMS-free**. NMS is a
non-differentiable, score-dependent filter whose behaviour changes with box
density — and conformal calibration is a statement about precisely that score
distribution. An end-to-end set-prediction head emits one score per object with
no post-hoc suppression, so the calibration target is the model's own output
rather than an artefact of the filter applied after it.
""")
code(r"""
rows = [dict(model=r["model"].replace(".pt",""), dataset=r["dataset"],
             encoding=r.get("encoding","baseline"), imgsz=r["imgsz"], epochs=r["epochs"],
             mAP50=round(r["test_mAP50"],4), mAP50_95=round(r["test_mAP50_95"],4),
             minutes=round(r["train_seconds"]/60,1)) for r in res]
df_det = pd.DataFrame(rows).sort_values(["dataset","model","encoding"])
print(df_det.to_string(index=False))
""")

code(r"""
piv = df_det[df_det.encoding=="baseline"].pivot_table(index="dataset", columns="model", values="mAP50")
series = {m: [piv[m].get(d, np.nan) for d in DATASETS] for m in piv.columns}
fig = F.grouped_bars(DATASETS, series, "test mAP@0.5", "Detector comparison (baseline encoding)")
F.save(fig, "fig09_detector_comparison"); plt.show()
""")

md("### Training dynamics")
code(r"""
curves = {}
for p in sorted(glob.glob("runs/detect/runs/detect/*/results.csv")):
    tag = os.path.basename(os.path.dirname(p))
    try:
        c = pd.read_csv(p); c.columns = [x.strip() for x in c.columns]
        col = [x for x in c.columns if "mAP50(B)" in x and "95" not in x]
        if col: curves[tag] = (c.index.values + 1, c[col[0]].values)
    except Exception: pass
if curves:
    sel = dict(list(curves.items())[:6])
    fig = F.curve_panel(sel, "epoch", "val mAP@0.5", "Training dynamics", figsize=(7.5, 4))
    F.save(fig, "fig10_training_curves"); plt.show()
""")

json.dump(nb, io.open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"part 2 written: {len(cells)} cells total")
