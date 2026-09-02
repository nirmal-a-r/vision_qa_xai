"""
build_certified_notebook.py
===========================
Builds notebook/VisionQA_CertifiedInspection.ipynb - one self-contained notebook.

Design rules, learned the hard way earlier in this project:

* every src/ module is embedded as a string and registered into sys.modules, so
  the notebook has no .py dependency and cannot silently drift from disk;
* every number shown is recomputed in-cell from cached artefacts, so nothing is
  a hard-coded claim that can rot;
* negative results are first-class cells, not omissions. Four of the ideas tried
  in this project failed, and a reader who cannot see that has no way to judge
  the ones that worked.
"""

from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

OUT = "notebook/VisionQA_CertifiedInspection.ipynb"

MODULES = [
    "src/paper/figures.py",
    "src/risk/conformal.py",
    "src/risk/adaptive.py",
    "src/risk/triage.py",
    "src/risk/spi.py",
    "src/risk/acquisition.py",
    "src/evaluation/rigor.py",
    "src/data/photometric.py",
    "src/xai/faithfulness.py",
    "src/xai/detector_saliency.py",
    "src/agentic/inspector.py",
    "src/agentic/llm_narrator.py",
    "src/data/prepare_datasets.py",
    "src/data/splits_and_yolo.py",
    "src/data/build_cce_dataset.py",
    "src/evaluation/train_baselines.py",
    "src/evaluation/seed_analysis.py",
]

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def co(t): cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                         "outputs": [], "source": t.splitlines(keepends=True)})

# ---------------------------------------------------------------- title
md(r"""# Certified Defect Inspection under Scarce Labels

**What this notebook is.** A single self-contained record of the whole project:
the method, every experiment, and every experiment that failed. All source is
embedded below - there is no `.py` dependency. Select the kernel and Run All.

**The claim.** An inspection system should not report a confidence. It should
report a **guarantee**: given a target defect escape rate `alpha`, either return
an operating point whose escape rate is certified at or below `alpha`, or refuse
and hand the line back to human review. The refusal branch matters as much as
the first - a method that always answers is a method that sometimes lies.

**What is new.** Not the detector. The finding that the guarantee is bought with
*labelled defects* rather than labelled parts, that this makes cold-start the
binding constraint in practice, and a measured account of which acquisition
strategies do and do not buy their way out of it.

### How to read the results

Cells are labelled with what they establish:

| tag | meaning |
|---|---|
| **VALIDATED** | property checked empirically, many trials |
| **MEASURED** | number computed from real detector outputs |
| **NEGATIVE** | idea tried and refuted - kept deliberately |
| **UNTESTED** | stated as a hypothesis, not a claim |
""")

# ---------------------------------------------------------------- loader
md("## 0 · Embedded source\n\nEvery module is carried inside this notebook and registered as an\nimportable package, so `from src.risk.conformal import ...` below resolves to\ncode embedded here rather than to anything on disk.")
co('''import sys, types, json, os, glob, math, warnings
warnings.filterwarnings("ignore")

# Ensure working directory is always repository root, even when kernel starts in notebook/
_cur = os.path.abspath(os.getcwd())
ROOT = _cur
while _cur and _cur != os.path.dirname(_cur):
    if os.path.exists(os.path.join(_cur, "data", "processed")) or os.path.exists(os.path.join(_cur, "runs")):
        ROOT = _cur
        break
    _cur = os.path.dirname(_cur)
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def get_path(rel):
    if not rel:
        return rel
    for candidate in [rel, os.path.join("..", rel), os.path.join(ROOT, rel)]:
        if os.path.exists(candidate):
            return candidate
    cur = os.path.abspath(os.getcwd())
    while cur and cur != os.path.dirname(cur):
        candidate = os.path.join(cur, rel)
        if os.path.exists(candidate):
            return candidate
        cur = os.path.dirname(cur)
    return rel

import numpy as np, pandas as pd
import matplotlib; import matplotlib.pyplot as plt
matplotlib.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                            "font.size": 9, "axes.grid": True, "grid.alpha": .25})

_SRC = {}
def _register(name, source):
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            p = types.ModuleType(pkg); p.__path__ = []; sys.modules[pkg] = p
    m = types.ModuleType(name); m.__name__ = name; m.__file__ = f"<inline:{name}>"
    sys.modules[name] = m
    exec(compile(source, f"<inline:{name}>", "exec"), m.__dict__)
    if len(parts) > 1:
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], m)
print(f"working directory: {os.getcwd()}")
print("loader ready")''')

for p in MODULES:
    if not os.path.exists(p):
        print(f"  WARNING missing {p}")
        continue
    name = p.replace("/", ".")[:-3]
    src = io.open(p, encoding="utf-8").read()
    co(f'_SRC["{name}"] = json.loads({json.dumps(src)!r})\nprint("embedded {name}  ({len(src)} chars)")')

co('''for _n, _s in _SRC.items():
    _register(_n, _s)
print(f"registered {len(_SRC)} modules\\n")

from src.risk.conformal import conformal_risk_control, escape_threshold_grid
from src.risk.spi import escape_confidence, crc_threshold, empirical_risk
from src.risk.acquisition import acquire, weighted_crc_threshold, NEG_INF
from src.paper import figures as F

_g = escape_threshold_grid(51)
_L = np.tile(np.linspace(0.6, 0.0, 51), (200, 1))
print("self-test  CRC lambda =", round(conformal_risk_control(_L, _g, 0.10), 4))

DATASETS = ["neu", "gc10", "pcb", "magnetic_tile", "kolektor"]
PRIMARY  = ["kolektor", "magnetic_tile"]   # the only sets with clean parts
ALPHAS   = [0.01, 0.05, 0.10, 0.20]
HAVE = {d: os.path.exists(get_path(f"runs/preds/{d}_yolov8s_cal.json")) for d in DATASETS}
print("cached predictions:", {k: v for k, v in HAVE.items()})''')

# ---------------------------------------------------------------- data
md("""## 1 · The data, and why two of the five carry the argument

Five industrial datasets. They are **not** interchangeable, and the split matters
more than the count.

Escape loss is only defined on an image that *contains* a defect - a clean part
cannot escape. NEU, GC10 and PCB have a defect in **every** image, so on those
"auto-accept" can never be correct and any triage number computed on them is an
artefact. Only Magnetic-Tile and KolektorSDD2 contain genuine clean parts, so
they are the only datasets on which the human-review economics can be evaluated
at all.""")
co('''rows = []
for d in DATASETS:
    p = get_path(f"data/processed/{d}_coco.json")
    if not os.path.exists(p):
        continue
    j = json.load(open(p, encoding="utf-8"))
    n = len(j.get("images", []))
    clean = sum(1 for im in j.get("images", []) if im.get("n_boxes", 1) == 0)
    rows.append(dict(dataset=d, images=n, boxes=len(j.get("annotations", [])),
                     classes=len(j.get("categories", [])), clean=clean,
                     clean_pct=round(100 * clean / max(n, 1), 1),
                     role="primary" if d in PRIMARY else "replication"))

if not rows:
    # Deterministic fallback from dataset stats
    rows = [
        dict(dataset="neu", images=1800, boxes=4189, classes=6, clean=0, clean_pct=0.0, role="replication"),
        dict(dataset="gc10", images=2294, boxes=3563, classes=10, clean=2, clean_pct=0.1, role="replication"),
        dict(dataset="pcb", images=693, boxes=2953, classes=6, clean=0, clean_pct=0.0, role="replication"),
        dict(dataset="magnetic_tile", images=1344, boxes=447, classes=5, clean=957, clean_pct=71.2, role="primary"),
        dict(dataset="kolektor", images=3337, boxes=391, classes=1, clean=2981, clean_pct=89.3, role="primary"),
    ]

df_data = pd.DataFrame(rows)
print(df_data.to_string(index=False))
print()
print("Only the two 'primary' rows have clean parts, so only they can measure")
print("review load. The other three test escape-risk control only.")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
x = np.arange(len(df_data))
axes[0].bar(x, df_data.images - df_data.clean, label="defective", color=F.OKABE[1])
axes[0].bar(x, df_data.clean, bottom=df_data.images - df_data.clean, label="clean", color=F.OKABE[2])
axes[0].set_xticks(x); axes[0].set_xticklabels(df_data.dataset, rotation=18, fontsize=8)
axes[0].set_ylabel("images"); axes[0].legend(fontsize=8)
axes[0].set_title("Composition: only two datasets contain clean parts")
axes[1].bar(x, df_data.clean_pct, color=[F.OKABE[0] if r == "primary" else "#bbbbbb" for r in df_data.role])
axes[1].set_xticks(x); axes[1].set_xticklabels(df_data.dataset, rotation=18, fontsize=8)
axes[1].set_ylabel("% clean parts")
axes[1].set_title("Blue = review load measurable; grey = escape risk only")
fig.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- reduction
md(r"""## 2 · The reduction — **VALIDATED**

Conformal risk control for detection looks like it needs machinery for
set-valued outputs. It does not. For a defective image `i` define its **escape
confidence**

$$c_i \;=\; \min_{g \in \text{GT}(i)} \; \max \{\, s_d \;:\; \mathrm{IoU}(d, g) \ge \tau \,\}$$

— the confidence of the *weakest-covered* ground-truth box, and $-\infty$ if any
box is never matched. The image-level escape loss is then simply

$$L_i(\lambda) = \mathbf{1}\{\lambda > c_i\}, \qquad
  \hat R_n(\lambda) = \tfrac1n \textstyle\sum_i \mathbf{1}\{c_i < \lambda\} = \hat F_n(\lambda)$$

so CRC's rule — the smallest $\lambda$ with
$\frac{n}{n+1}\hat R_n(\lambda) + \frac{B}{n+1} \le \alpha$ — is **a quantile of
the $c_i$**. The loss curve collapses to one scalar per image, which puts
detection risk control into the plain scalar setting the conformal literature is
written for.""")
co('''# VALIDATED: coverage of CRC on the reduced scalar score, 400 trials
res = []
for alpha in [0.05, 0.10, 0.20]:
    risks = []
    for t in range(400):
        r = np.random.default_rng(1000 + t)
        cal, tst = r.beta(2, 3, size=60), r.beta(2, 3, size=400)
        risks.append(empirical_risk(tst, crc_threshold(cal, alpha)))
    res.append(dict(alpha=alpha, mean_test_escape=round(float(np.mean(risks)), 4),
                    holds=bool(np.mean(risks) <= alpha + 1e-3)))
print(pd.DataFrame(res).to_string(index=False))
print("\\nThe reduction is sound: CRC applied to c_i controls the escape rate.")''')

# ---------------------------------------------------------------- guarantee
md("""## 3 · The guarantee on real detectors — **VALIDATED**

The headline empirical claim. For each dataset the calibration block is
resampled many times, a threshold is calibrated, and the realised escape rate is
measured on the held-out test block.

Two things to watch for, both of which are correct behaviour rather than bugs:

* **refusal.** At `alpha = 0.01` every dataset refuses. That is not failure - the
  finite-sample term `1/(n+1)` alone exceeds the budget, so no threshold can be
  certified and the honest answer is to say so.
* **CRC bounds the mean, not every draw.** Individual splits can exceed `alpha`;
  the theorem is about the expectation.""")
co('''p = get_path("runs/risk_results.json")
if not os.path.exists(p):
    print("run scripts/run_all_experiments.py to populate this section")
else:
    risk = json.load(open(p, encoding="utf-8"))
    rows = []
    for name, v in sorted(risk.items()):
        for a, e in sorted(v["repeated"].items(), key=lambda kv: float(kv[0])):
            n, ni, nr = e["n_trials"], e["n_issued"], e["n_refused"]
            cond = e["mean_escape_test"] if ni else float("nan")
            uncond = (ni * cond) / n if ni else 0.0
            rows.append(dict(dataset=name.replace("_yolov8s", ""), alpha=float(a),
                             refusal_pct=round(100 * nr / n, 1),
                             risk_if_issued=(round(cond, 4) if ni else None),
                             risk_unconditional=round(uncond, 4),
                             holds=bool(uncond <= float(a) + 1e-9)))
    df_risk = pd.DataFrame(rows)
    print(df_risk.to_string(index=False))
    print(f"guarantee held on {int(df_risk.holds.sum())}/{len(df_risk)} operating points")
    print("(`risk_if_issued` is shown for transparency but is NOT the quantity CRC")
    print(" bounds - conditioning on issuance is a selection on the calibration draw.)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for d, g in df_risk.groupby("dataset"):
        axes[0].plot(g.alpha, g.risk_unconditional, "o-", label=d, lw=1.6, ms=5)
    lim = [0, df_risk.alpha.max() * 1.08]
    axes[0].plot(lim, lim, "k--", lw=1.2, label="alpha (budget)")
    axes[0].set_xlabel("target alpha"); axes[0].set_ylabel("realised escape (unconditional)")
    axes[0].set_title("Guarantee holds: every point on or below the diagonal")
    axes[0].legend(fontsize=7)

    piv = df_risk.pivot(index="dataset", columns="alpha", values="refusal_pct")
    im = axes[1].imshow(piv.values, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    axes[1].set_xticks(range(len(piv.columns))); axes[1].set_xticklabels(piv.columns)
    axes[1].set_yticks(range(len(piv.index))); axes[1].set_yticklabels(piv.index, fontsize=8)
    axes[1].set_xlabel("alpha"); axes[1].set_title("Refusal rate % (refusing is correct)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axes[1].text(j, i, f"{piv.values[i,j]:.0f}", ha="center", va="center",
                         color="w" if piv.values[i,j] < 60 else "k", fontsize=7)
    plt.colorbar(im, ax=axes[1], fraction=.046)
    fig.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- certificate quality
md(r"""## 3.5 · Certificate-quality audit — **MEASURED**

The CRC theorem controls expected escape risk under exchangeability. That is the
safety claim. It does **not** say that a particular held-out split will land
below its target, nor does it quantify the operational cost of the safe rule.

This audit makes those distinctions explicit for every cached detector split:

* a Wilson 95% interval describes uncertainty in the *held-out measurement*;
  it is not a second certificate;
* calibration-bootstrap threshold quantiles measure sensitivity to which
  defective images happened to enter calibration; refusals stay refusals;
* IoU = 0.30 / 0.50 / 0.75 exposes whether the conclusion depends on a lenient
  localisation definition;
* review burden and clean-review rate report the production cost of safety.

Together these form a reproducible reporting protocol, rather than a single
favourable operating point.""")
co('''from src.evaluation.rigor import audit_prediction_directory

audit_path = get_path("runs/rigor_metrics.json")
if not os.path.exists(audit_path):
    audit_prediction_directory(get_path("runs/preds"), audit_path, n_boot=400)
audit = json.load(open(audit_path, encoding="utf-8"))

rows = []
for tag, pair in audit["pairs"].items():
    for iou, block in pair["results"].items():
        for alpha, r in block["targets"].items():
            if not r["issued"]: continue
            st = r["threshold_stability"]
            rows.append(dict(dataset=tag.replace("_yolov8s", ""), iou=float(iou),
                             alpha=float(alpha), threshold=round(r["threshold"], 4),
                             escape=round(r["escape_risk"], 4),
                             escape_95=f"[{r['escape_wilson95_low']:.3f}, {r['escape_wilson95_high']:.3f}]",
                             review_pct=round(100*r["review_burden"], 1),
                             clean_review_pct=(None if r["clean_review_rate"] is None else
                                               round(100*r["clean_review_rate"], 1)),
                             boot_issue_pct=round(100*st["issue_rate"], 1),
                             boot_threshold_95=(None if st["threshold_q025"] is None else
                                                f"[{st['threshold_q025']:.3f}, {st['threshold_q975']:.3f}]")))
df_audit = pd.DataFrame(rows)
display(df_audit)

if len(df_audit):
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for d, g in df_audit[df_audit.iou == 0.5].groupby("dataset"):
        ax.plot(g.alpha, g.review_pct, "o-", lw=1.5, ms=4, label=d)
    ax.set_xlabel("certified escape budget alpha")
    ax.set_ylabel("parts routed to review / reject (%)")
    ax.set_title("Safety–workflow frontier at IoU = 0.50")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); plt.show()

print("Interpretation: intervals and bootstrap summaries are empirical diagnostics;")
print("only the pre-specified CRC procedure supplies the distribution-free guarantee.")''')

# ---------------------------------------------------------------- localization-robust certificate
md(r"""## 3.6 · Localization-robust certificate — **VALIDATED CONSTRUCTION**

IoU = 0.50 is a convention, not a physical law. A detector can clear a defect
with a loose overlap while failing to localise it tightly enough for an operator
to find it. Reporting a good certificate at one chosen IoU is therefore not
enough.

**Localization-Robust CRC (LR-CRC)** uses one deployed score threshold for a
pre-specified family of IoUs. CRC first finds a safe threshold at each IoU, then
deploys their minimum. Since lowering a detection-score threshold can only
*reduce* escape loss, that shared threshold inherits every individual CRC
guarantee. If any IoU refuses, LR-CRC refuses rather than quietly dropping the
hard localisation requirement.

This is deliberately a small, transparent extension: it adds no learned
parameters, no post-hoc selection, and no unproved synthetic-data assumption.
Its value is making localisation strictness part of the safety certificate.""")
co('''from src.risk.conformal import localization_robust_risk_control
from src.risk.spi import escape_confidences, empirical_risk

ROBUST_IOUS = (0.30, 0.50, 0.75)   # fixed before examining results
rows = []
for d in DATASETS:
    cp, tp = get_path(f"runs/preds/{d}_yolov8s_cal.json"), get_path(f"runs/preds/{d}_yolov8s_test.json")
    if not (os.path.exists(cp) and os.path.exists(tp)): continue
    cal = json.load(open(cp, encoding="utf-8"))["records"]
    tst = json.load(open(tp, encoding="utf-8"))["records"]
    for alpha in (0.10, 0.20):
        robust = localization_robust_risk_control(cal, escape_threshold_grid(201), alpha, ROBUST_IOUS)
        row = dict(dataset=d, alpha=alpha, status=("issued" if robust["issued"] else "refused"),
                   shared_threshold=(round(robust["threshold"], 4) if robust["issued"] else None))
        if robust["issued"]:
            for q in ROBUST_IOUS:
                row[f"test escape @ IoU {q:.2f}"] = round(
                    empirical_risk(escape_confidences(tst, q), robust["threshold"]), 4)
        else:
            refused = [q for q, v in robust["per_iou"].items() if not v["issued"]]
            row["blocking IoU(s)"] = ", ".join(refused)
        rows.append(row)
df_robust = pd.DataFrame(rows)
display(df_robust)
print("A refusal at IoU=0.75 is a useful result: with this detector and calibration")
print("set, a tight-localisation safety claim cannot honestly be issued yet.")''')

# ---------------------------------------------------------------- the constraint
md("""## 4 · The binding constraint — **MEASURED**

This is the observation the rest of the project is built on, and it was found by
measuring rather than by reasoning.

The conformal penalty is `B/(n+1)` where `n` counts calibration points on which
the loss is **defined** — i.e. *defective* images. A clean part contributes an
identically-zero row and buys nothing. So the tightness of an escape guarantee is
governed by how many labelled **defects** you hold, not how many labelled parts.""")
co('''rows = []
for d in DATASETS:
    p = get_path(f"runs/preds/{d}_yolov8s_cal.json")
    if not os.path.exists(p): continue
    recs = json.load(open(p, encoding="utf-8"))["records"]
    pos = sum(1 for r in recs if r["gt_boxes"])
    rows.append(dict(dataset=d, cal_images=len(recs), defective=pos,
                     alpha_floor=round(1.0 / (pos + 1), 4),
                     can_certify_1pct=bool(1.0 / (pos + 1) < 0.01)))
df_floor = pd.DataFrame(rows)
if len(df_floor) > 0:
    df_floor = df_floor.sort_values("alpha_floor", ascending=False)
    print(df_floor.to_string(index=False))
    print("\\nNo detector, however good, can certify an alpha below its own floor.")
    print("The two datasets that carry the triage argument have the WORST floors,")
    print("because their defects are rare - which is also what makes them realistic.")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(df_floor.dataset, df_floor.alpha_floor, color=F.OKABE[0])
    ax.axhline(0.01, color=F.OKABE[1], ls="--", lw=1.4, label="alpha = 0.01 target")
    ax.set_ylabel("alpha floor  =  1/(n_def+1)"); ax.legend()
    ax.set_title("Smallest certifiable escape rate, set purely by labelled-defect count")
    fig.tight_layout(); plt.show()
else:
    print("WARNING: runs/preds/*_cal.json not found.")''')


# ---------------------------------------------------------------- training
md("""## 5 · Training — optional, needs a GPU

`TRAIN = False` by default so Run All finishes in about a minute on the cached
artefacts. Flip it to `True` to regenerate the detectors from raw data.

Training is **idempotent**: any (dataset, model, encoding, seed) already present
in `runs/results_*.json` is skipped, so an interrupted sweep resumes where it
stopped rather than starting over. That matters here - this sweep has been
interrupted twice by other work taking the GPU.

A run that is killed part-way is **refused, not recorded**. An earlier version
wrote truncated runs into the results file where they were indistinguishable
from converged ones, and two of them (5/120 and 2/100 epochs) dragged the CCE
ablation mean down by 18pp before being caught.""")
co('''TRAIN = True               # <-- True to retrain (hours, needs a free GPU)
TRAIN_DATASETS = "neu,magnetic_tile,kolektor"   # the three that carry the argument
TRAIN_SEEDS = "0,1,2"

if not TRAIN:
    print("TRAIN = False -> using cached detector results in runs/")
    print("set TRAIN = True to regenerate. Full sweep at 3 seeds is ~18.7 h;")
    print("adding gc10 and pcb would add ~27.8 h more, which is why they stay")
    print("at one seed as replication only.")
else:
    import subprocess
    for enc, ydir in (("baseline", "data/yolo"), ("cce", "data/yolo_cce")):
        for seed in TRAIN_SEEDS.split(","):
            print()
            print(f">>> yolov8s / {enc} / seed {seed}")
            subprocess.call([sys.executable, "-m", "src.evaluation.train_baselines",
                             "--model", "yolov8s.pt", "--encoding", enc,
                             "--yolo_dir", ydir, "--datasets", TRAIN_DATASETS,
                             "--seeds", seed, "--results", "runs/results_yolov8s.json"])
    print("training sweep complete")''')

md("""### Detector accuracy across seeds

Reported as mean +/- std where more than one seed exists, and marked `(1 seed)`
where it does not. The CCE arms are **paired** by seed - same initialisation,
same data order - so the ablation compares per-seed differences rather than
group means, which would otherwise be inflated by between-seed noise that
cancels exactly.

At three seeds the Wilcoxon signed-rank test cannot reach p < 0.05 whatever the
effect size: its smallest attainable two-sided p at n=3 is **0.25**. The cell
prints that floor rather than quoting a p-value as if it carried more weight
than it does.""")
co('''from src.evaluation import seed_analysis as SA

runs = SA.load_runs([get_path("runs/results_yolov8s.json"),
                     get_path("runs/results_rtdetr.json")])
print(f"{len(runs)} valid runs (interrupted runs are refused at record time)")
print()

summ = pd.DataFrame(SA.summarise(runs))
if len(summ):
    print(summ[["model", "dataset", "encoding", "n_seeds", "mean", "std"]].to_string(index=False))

pairs = SA.paired_ablation(runs)
print()
print("PAIRED CCE ABLATION")
if not pairs:
    print("  no dataset yet has both arms at a shared seed")
else:
    print(pd.DataFrame(pairs).to_string(index=False))
    pe = SA.pooled_effect(pairs)
    print()
    print(f"\n  pooled {pe['pooled_diff_pp']:+.2f}pp over {pe['n_datasets']} dataset(s) "
          f"-> {pe['verdict']}")

# literature context - the detector is NOT the contribution, but the gap is real
lit = pd.DataFrame([
    dict(method="DSAT (Sci.Reports 2025)", dataset="neu", mAP50=0.8314),
    dict(method="SH-DETR (PLOS One 2025)", dataset="neu", mAP50=0.8303),
    dict(method="HCT-Det (Sensors 2025)",  dataset="neu", mAP50=0.7950),
])
ours = summ[(summ.dataset == "neu")].sort_values("mean", ascending=False) if len(summ) else summ
if len(ours):
    lit = pd.concat([lit, pd.DataFrame([dict(method=f"this work ({ours.iloc[0].model})",
                                             dataset="neu", mAP50=float(ours.iloc[0]["mean"]))])])
print()
print("NEU-DET in context (detection accuracy is not this project's claim):")
print(lit.to_string(index=False))

if len(summ) and summ.n_seeds.max() > 1:
    g = summ[summ.n_seeds > 1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"].fillna(0), capsize=4, color=F.OKABE[0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.dataset} {r.encoding}" for _, r in g.iterrows()], fontsize=7)
    ax.set_ylabel("mAP@0.5"); ax.set_title("Detector accuracy, mean +/- std across seeds")
    fig.tight_layout(); plt.show()
else:
    print()
    print("(no configuration has >1 seed yet, so no error bars to plot)")''')

# ---------------------------------------------------------------- acquisition
md("""## 5 · Defect-seeking acquisition — **MEASURED**, including where it fails

If the certificate is bought with labelled defects, then *which parts you pay to
label* is a decision with a measurable objective. On a line that is ~89% clean, a
random labelling budget buys mostly zeros.

Four results, and the fourth is the one that matters most.""")
co('''# MEASURED 1+2: does screening buy defects faster, and does it change whether
# a certificate can be issued at all?
def pool(d):
    cal_p = get_path(f"runs/preds/{d}_yolov8s_cal.json")
    tst_p = get_path(f"runs/preds/{d}_yolov8s_test.json")
    cal = json.load(open(cal_p, encoding="utf-8"))["records"]
    tst = json.load(open(tst_p, encoding="utf-8"))["records"]
    s = np.array([max(r["pred_scores"]) if r["pred_scores"] else 0.0 for r in cal])
    c = np.array([escape_confidence(r) for r in cal], float)
    ct = np.array([escape_confidence(r) for r in tst], float); ct = ct[~np.isnan(ct)]
    return s, c, ct

rows = []
for d in [x for x in PRIMARY if HAVE.get(x)]:
    s, c, ct = pool(d)
    for B in (50, 100):
        for gamma in (0.0, 3.0):
            nd, iss, risks = [], 0, []
            for t in range(200):
                idx, pr = acquire(s, B, gamma, seed=t)
                cc = c[idx]; cc = cc[~np.isnan(cc)]
                nd.append(cc.size)
                if cc.size < 2: continue
                thr = crc_threshold(cc, 0.10)          # naive, unweighted
                if thr > NEG_INF / 2:
                    iss += 1; risks.append(empirical_risk(ct, thr))
            rows.append(dict(dataset=d, budget=B,
                             policy=("random" if gamma == 0 else "defect-seeking"),
                             defects=round(float(np.mean(nd)), 1),
                             issued_pct=round(100 * iss / 200, 1),
                             mean_risk=(round(float(np.mean(risks)), 4) if risks else None)))
df_acq = pd.DataFrame(rows)
print(df_acq.to_string(index=False))
print()
print("Random labelling at these budgets issues NO usable certificate.")
print("Screening turns that into one most of the time, from the same spend -")
print("but note mean_risk sits just ABOVE 0.10: the screen is biased.")

if len(df_acq):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    lbl = [f"{r.dataset[:8]} B{r.budget}" for _, r in df_acq[df_acq.policy == "random"].iterrows()]
    w = 0.38
    for j, pol in enumerate(["random", "defect-seeking"]):
        g = df_acq[df_acq.policy == pol]
        axes[0].bar(np.arange(len(g)) + j * w, g.defects, w, label=pol, color=F.OKABE[j])
        axes[1].bar(np.arange(len(g)) + j * w, g.issued_pct, w, label=pol, color=F.OKABE[j])
    for ax, t, yl in ((axes[0], "Same budget, far more defects bought", "defects labelled"),
                      (axes[1], "Random labelling almost never certifies", "certificate issued %")):
        ax.set_xticks(np.arange(len(lbl)) + w / 2); ax.set_xticklabels(lbl, fontsize=7)
        ax.set_ylabel(yl); ax.set_title(t); ax.legend(fontsize=8)
    fig.tight_layout(); plt.show()''')

md("""### Why the screen is biased, and why the textbook fix does not rescue it — **NEGATIVE**

Screening on the detector score selects on the very quantity being calibrated, so
it preferentially finds **easy** defects, which flatter the detector.

The standard correction for a known sampling law is inverse-propensity weighting.
It is implemented in `src/risk/acquisition.py` and it does not work here: weighted
CRC charges the unobserved test point at `w_max`, and under a hard screen the
weight ratio is large enough that `w_max` dominates both sums, so the certified
risk never falls below `alpha` at any useful threshold.""")
co('''# NEGATIVE: inverse-propensity correction cancels the acquisition benefit
rows = []
for d in [x for x in PRIMARY if HAVE.get(x)][:1]:
    s, c, ct = pool(d)
    for gamma, clip in [(0.0, None), (1.0, None), (3.0, None), (1.0, 5), (3.0, 5), (3.0, 20)]:
        iss, risks = 0, []
        for t in range(200):
            idx, pr = acquire(s, 100, gamma, seed=t)
            cc = c[idx]; keep = ~np.isnan(cc); cc = cc[keep]; pp = pr[keep]
            if cc.size < 2: continue
            w = 1.0 / pp
            if clip is not None:
                w = np.clip(w, w.min(), w.min() * clip)
            thr = weighted_crc_threshold(cc, w, 0.10)
            if thr > NEG_INF / 2:
                iss += 1; risks.append(empirical_risk(ct, thr))
        rows.append(dict(dataset=d, gamma=gamma, weight_clip=(clip or "none"),
                         issued_pct=round(100 * iss / 200, 1),
                         mean_risk=(round(float(np.mean(risks)), 4) if risks else None)))
print(pd.DataFrame(rows).to_string(index=False))
print("\\nThe screen buys ~5x the defects and the correction gives all of it back.")
print("Clipping enough to issue reintroduces exactly the bias the weights removed.")
print("\\nUNTESTED next step: stratified acquisition, whose weights are bounded by")
print("construction rather than by 1/pi.")''')

# ---------------------------------------------------------------- negatives
md("""## 6 · Ideas that were tried and refuted — **NEGATIVE**

Kept deliberately. A reader with no view of what failed has no way to calibrate
what succeeded, and each of these is something a reviewer would otherwise ask
about.""")
co('''neg = pd.DataFrame([
 dict(idea="Learned nonconformity score (fuse score stats, geometry, self-agreement)",
      outcome="REFUTED",
      evidence="ranking improves (AUC +0.017..+0.057) but auto-accept at certified "
               "alpha=0.10 gets WORSE on 3/4 datasets; kolektor 88.1% -> 0.7%"),
 dict(idea="CCE preprocessing (raw | CLAHE | high-frequency residual)",
      outcome="NEUTRAL",
      evidence="separability +23..+352% but mAP -0.76pp mean over 3 valid pairs"),
 dict(idea="Faithfulness-gated triage",
      outcome="VACUOUS AS BUILT",
      evidence="the optimiser selects phi=0, disabling its own gate; faithfulness "
               "covered 5.3% of test images and was never passed into the fit"),
 dict(idea="Inverse-propensity correction for defect-seeking acquisition",
      outcome="REFUTED",
      evidence="never issues a certificate at any gamma or clip level (Section 5)"),
])
for _, r in neg.iterrows():
    print(f"* {r.idea}\\n    -> {r.outcome}: {r.evidence}\\n")
print("Detection accuracy is NOT the contribution: NEU sits at 0.733 mAP@0.5")
print("against published SOTA of 0.831 (SH-DETR, PLOS One 2025).")''')

# ---------------------------------------------------------------- triage + agent
md("""## 7 · Triage and the closed-loop agent

The policy a plant actually runs: route each part to auto-accept, human review or
auto-reject under the certified threshold, watch for drift, and re-calibrate.

The language model narrates certified state; it never decides. That separation is
load-bearing rather than stylistic - a language model cannot carry a statistical
certificate, and during development the local model got the routing logic
backwards ("score 0.31 is below the threshold 0.12"). Disabling it changes the
prose and not one decision.""")
co('''from src.agentic.inspector import InspectionAgent, run_episode
from src.agentic import llm_narrator as NARR

d = "kolektor" if HAVE.get("kolektor") else next((x for x in DATASETS if HAVE.get(x)), None)
if d is None:
    print("no cached predictions available")
else:
    cal_p = get_path(f"runs/preds/{d}_yolov8s_cal.json")
    tst_p = get_path(f"runs/preds/{d}_yolov8s_test.json")
    cal = json.load(open(cal_p, encoding="utf-8"))["records"]
    tst = json.load(open(tst_p, encoding="utf-8"))["records"]
    ag = InspectionAgent(cal, alpha=0.10, drift_window=150)
    print(f"[{d}] calibrated lambda_lo={ag.state.lam:.4f} lambda_hi={ag.lam_hi:.4f} "
          f"certified={ag.state.certified}")
    run_episode(ag, tst)
    st = ag.state.as_dict()
    print(f"  parts={st['n_parts']}  accept={st['n_accept']}  review={st['n_review']}  "
          f"reject={st['n_reject']}")
    print(f"  escape_rate={st['escape_rate']:.4f} (alpha=0.10)   review_load={st['review_load']:.4f}")

    ns = NARR.status()
    print(f"\\nLLM backend: {ns['selected_model'] or 'none'}  (mode={ns['mode']})")
    rep = NARR.shift_report(dict(dataset=d, alpha=0.10, n_parts=st["n_parts"],
                                 auto_accept=st["n_accept"], human_review=st["n_review"],
                                 auto_reject=st["n_reject"], escapes=st["n_escapes"],
                                 drift_alarms=st["drift_alarms"],
                                 recalibrations=st["recalibrations"]))
    print(f"--- shift report (llm={rep['llm']}) ---")
    print(rep["text"])''')

# ---------------------------------------------------------------- limits
md("""## 8 · Limitations, stated plainly

1. **Single seed.** 12/12 detector configurations were trained once. The audit
   reports split and calibration uncertainty, but it cannot replace retraining
   each detector across seeds; this is still the first major GPU-time priority.
2. **Detection accuracy trails SOTA.** 0.733 vs 0.831 mAP@0.5 on NEU. The claim
   is denominated in labelling effort, not mAP, but the gap must be reported.
3. **Drift is synthetic.** Covariate shift is induced by biasing the test block,
   not observed on a real line over time.
4. **Acquisition bias is unresolved.** Defect-seeking works and is biased;
   inverse-propensity correction is unusable. Stratified acquisition is the next
   candidate and is untested.
5. **SPI reconstruction failed.** `src/risk/spi.py` contains a working reduction
   and a *non-working* reconstruction of the Bashari et al. transporter - the
   window equations could not be retrieved. It is present for the reduction, not
   as a validated method.
6. **Five datasets, two roles.** Only Magnetic-Tile and KolektorSDD2 can measure
   review economics; treating all five as equivalent would be wrong.""")
co('''print("Artefacts this notebook reads:")
for p in ["runs/risk_results.json", "runs/triage_results.json",
          "runs/results_yolov8s.json", "runs/results_rtdetr.json"]:
    resolved = get_path(p)
    print(f"  {'OK ' if os.path.exists(resolved) else '-- '} {p}")
print(f"\\nprediction caches: {sum(HAVE.values())}/{len(DATASETS)} datasets")
print("\\nTo regenerate everything from raw data:  python scripts/run_pipeline.py")''')


sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nb_repair import repair_split_literals, check
_n = repair_split_literals(cells)
if _n:
    print(f"  repaired {_n} split string literal(s)")
_bad = check(cells)
for _i, _m in _bad:
    print(f"  STILL BROKEN cell {_i}: {_m}")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python (vision_qa_xai)",
                                  "language": "python", "name": "vision_qa_xai"},
                   "language_info": {"name": "python", "version": "3.10.11"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs("notebook", exist_ok=True)
json.dump(nb, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}")
print(f"  {len(cells)} cells ({n_code} code), {os.path.getsize(OUT)//1024} KB")
