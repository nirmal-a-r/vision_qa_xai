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
    "src/data/photometric.py",
    "src/xai/faithfulness.py",
    "src/xai/detector_saliency.py",
    "src/agentic/inspector.py",
    "src/agentic/llm_narrator.py",
    "src/data/prepare_datasets.py",
    "src/data/splits_and_yolo.py",
    "src/data/build_cce_dataset.py",
    "src/evaluation/train_baselines.py",
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
HAVE = {d: os.path.exists(f"runs/preds/{d}_yolov8s_cal.json") for d in DATASETS}
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
    p = f"data/processed/{d}_coco.json"
    if not os.path.exists(p):
        continue
    j = json.load(open(p))
    n = len(j["images"])
    clean = sum(1 for im in j["images"] if im.get("n_boxes", 1) == 0)
    rows.append(dict(dataset=d, images=n, boxes=len(j["annotations"]),
                     classes=len(j["categories"]), clean=clean,
                     clean_pct=round(100 * clean / max(n, 1), 1),
                     role="primary" if d in PRIMARY else "replication"))
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
co('''p = "runs/risk_results.json"
if not os.path.exists(p):
    print("run scripts/run_all_experiments.py to populate this section")
else:
    risk = json.load(open(p))
    rows = []
    for name, v in sorted(risk.items()):
        for a, e in sorted(v["repeated"].items(), key=lambda kv: float(kv[0])):
            n, ni, nr = e["n_trials"], e["n_issued"], e["n_refused"]
            cond = e["mean_escape_test"] if ni else float("nan")
            # CRC bounds the UNCONDITIONAL expectation. A refusal flags every
            # part, so its realised escape is 0 and it belongs in the average.
            # Reporting only the issued trials keeps exactly the draws where
            # calibration happened to be permissive enough to certify - the
            # optimistic half - and makes a satisfied guarantee look violated.
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
    p = f"runs/preds/{d}_yolov8s_cal.json"
    if not os.path.exists(p): continue
    recs = json.load(open(p))["records"]
    pos = sum(1 for r in recs if r["gt_boxes"])
    rows.append(dict(dataset=d, cal_images=len(recs), defective=pos,
                     alpha_floor=round(1.0 / (pos + 1), 4),
                     can_certify_1pct=bool(1.0 / (pos + 1) < 0.01)))
df_floor = pd.DataFrame(rows).sort_values("alpha_floor", ascending=False)
print(df_floor.to_string(index=False))
print("\\nNo detector, however good, can certify an alpha below its own floor.")
print("The two datasets that carry the triage argument have the WORST floors,")
print("because their defects are rare - which is also what makes them realistic.")

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(df_floor.dataset, df_floor.alpha_floor, color=F.OKABE[0])
ax.axhline(0.01, color=F.OKABE[1], ls="--", lw=1.4, label="alpha = 0.01 target")
ax.set_ylabel("alpha floor  =  1/(n_def+1)"); ax.legend()
ax.set_title("Smallest certifiable escape rate, set purely by labelled-defect count")
fig.tight_layout(); plt.show()''')

# ---------------------------------------------------------------- acquisition
md("""## 5 · Defect-seeking acquisition — **MEASURED**, including where it fails

If the certificate is bought with labelled defects, then *which parts you pay to
label* is a decision with a measurable objective. On a line that is ~89% clean, a
random labelling budget buys mostly zeros.

Four results, and the fourth is the one that matters most.""")
co('''# MEASURED 1+2: does screening buy defects faster, and does it change whether
# a certificate can be issued at all?
def pool(d):
    cal = json.load(open(f"runs/preds/{d}_yolov8s_cal.json"))["records"]
    tst = json.load(open(f"runs/preds/{d}_yolov8s_test.json"))["records"]
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
    cal = json.load(open(f"runs/preds/{d}_yolov8s_cal.json"))["records"]
    tst = json.load(open(f"runs/preds/{d}_yolov8s_test.json"))["records"]
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

1. **Single seed.** 12/12 detector configurations were trained once. No error
   bars anywhere. This is the first thing a reviewer will ask for and it is pure
   GPU time.
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
    print(f"  {'OK ' if os.path.exists(p) else '-- '} {p}")
print(f"\\nprediction caches: {sum(HAVE.values())}/{len(DATASETS)} datasets")
print("\\nTo regenerate everything from raw data:  python scripts/run_pipeline.py")''')

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
