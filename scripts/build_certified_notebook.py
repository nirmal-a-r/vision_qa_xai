"""
build_certified_notebook.py
===========================
Builds notebook/VisionQA_CertifiedInspection.ipynb: one notebook that follows the
"Certified Cold-Start Defect Inspection - Complete Project Document" section by
section (docs/Project_Document.pdf), from the problem statement to every
experimental stage (0-6), the research questions (RQ1-RQ4) and the status list.

Design rules
* The notebook imports the code from this repository (src/), so it can never
  drift from what the scripts run; the first cell records the git commit and
  library versions for provenance.
* Nothing heavy runs by default: every GPU stage is a pipeline command, and the
  notebook READS its outputs. Every result cell says exactly which command
  produces its input when that input is missing, so Run All always finishes.
* Every number is recomputed from artefacts on disk; nothing is typed in by hand
  except where a cell explicitly quotes the first study (old splits) as context.

    python scripts/build_certified_notebook.py
    python scripts/execute_notebook.py
"""

from __future__ import annotations

import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebook", "VisionQA_CertifiedInspection.ipynb")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):03d}", "metadata": {},
                  "source": t.strip("\n").splitlines(keepends=True)})


def co(t):
    cells.append({"cell_type": "code", "id": f"c{len(cells):03d}", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})


# =============================================================== title
md(r'''
# Certified Cold-Start Defect Inspection (SPERC)

A new inspection line needs a **certified defect-escape rate on day one**, but it has only 5-40
real defect images to certify with. This notebook is the executable companion of the *Complete
Project Document* (`docs/Project_Document.pdf`): it follows the document section by section and
recomputes every number from the artefacts the pipeline writes.

**How to use it.** Select the `vqaenv` interpreter as the kernel and *Run All*. Nothing heavy runs
here; the GPU work is done by one command, and each result cell below says which command fills it:

```
vqaenv\Scripts\python.exe scripts\smoke_test.py          # 10-minute end-to-end check
vqaenv\Scripts\python.exe scripts\run_pipeline.py        # the full run (Stages 1-6)
```

| Tag | Meaning |
|---|---|
| **THEORY** | follows from SPI's theorems; checked numerically here |
| **VALIDATED** | property checked empirically over many draws |
| **MEASURED** | computed from real detector outputs |
| **PENDING** | needs pipeline output that is not on disk yet (the cell says how to make it) |
''')

co(r'''
import os, sys, json, glob, subprocess, platform, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def _find_root(start):
    cur = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(cur, "src", "risk", "sperc.py")) and \
           os.path.exists(os.path.join(cur, "configs", "config.yaml")):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            return None
        cur = nxt

ROOT = os.environ.get("VQA_ROOT") or _find_root(os.getcwd())
assert ROOT, "open this notebook from inside the vision_qa_xai repository"
WORK = os.environ.get("VQA_WORK") or ROOT          # where data/ and runs/ live
sys.path.insert(0, ROOT)
os.chdir(WORK)
R = lambda *p: os.path.join(ROOT, *p)              # code / config
W = lambda *p: os.path.join(WORK, *p)              # data / results

import numpy as np, pandas as pd, yaml
import matplotlib
try:                                   # figures inline even if MPLBACKEND=Agg is set in the shell
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
from src.paper import figures as F
F.set_style(9)
matplotlib.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25})
CFG = yaml.safe_load(open(R("configs", "config.yaml")))
OK = F.OKABE
METHOD_COLOR = {"crc": OK[0], "sperc_tierN": OK[1], "sperc_nominal": OK[2], "crc_at_cap": OK[3],
                "acq_uncorrected": OK[4], "legacy": OK[5], "naive_pooled": "#999999"}
METHOD_LABEL = {"crc": "CRC (real only)", "sperc_tierN": "SPERC, Tier N = target",
                "sperc_nominal": "SPERC at alpha = target", "crc_at_cap": "CRC at SPERC's Tier H cap",
                "acq_uncorrected": "defect-seeking acquisition (uncorrected)",
                "legacy": "old spi.py heuristic", "naive_pooled": "naive pooling (invalid)"}

def pending(what, command):
    display(Markdown(f"**PENDING** - {what} is not on disk yet. Produce it with:\n\n"
                     f"```\n{command}\n```"))

def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

# ---- provenance
def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"
import scipy
prov = {"repository": ROOT, "results folder": WORK, "git commit": _git("rev-parse", "--short", "HEAD"),
        "uncommitted changes": ("yes" if _git("status", "--porcelain") not in ("", "n/a") else "no"),
        "python": platform.python_version(), "executable": sys.executable,
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}
try:
    import torch
    prov["torch"] = torch.__version__
    prov["CUDA available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        prov["GPU"] = torch.cuda.get_device_name(0)
        prov["compute capability"] = "%d.%d" % torch.cuda.get_device_capability(0)
        prov["sm_120 in build (RTX 50xx)"] = any(a.startswith("sm_12") for a in torch.cuda.get_arch_list())
except Exception as e:
    prov["torch"] = f"not importable ({type(e).__name__}) - analysis cells still run"
try:
    import ultralytics; prov["ultralytics"] = ultralytics.__version__
except Exception:
    prov["ultralytics"] = "not installed"
display(pd.DataFrame(prov.items(), columns=["item", "value"]).set_index("item"))
''')

# =============================================================== 1-2 problem
md(r'''
## 1 · Problem (document Sections 1-2)

Certify, before a line has produced enough scrap to learn from, that a frozen object detector lets
through at most a chosen fraction `alpha` of defective parts - and refuse when that cannot be
certified.

**Routing rule.** A part is auto-accepted when no detection clears the threshold `lambda`, and sent
to review otherwise. The detector never saw the calibration images.

**Two escape events, defined once (Section 2.2)**, both 0/1 per defective image, reported
separately and never mixed (`src/risk/escape.py`):

$$c_i^{\text{part}} = \max_{d \in f(X_i)} \operatorname{score}(d), \qquad
  c_i^{\text{loc}} = \min_{g \in G(X_i)} \max\{\operatorname{score}(d) : \operatorname{IoU}(d,g) \ge \tau\}$$

with an empty set giving $-\infty$ (the part always escapes). Escape at threshold $\lambda$ is
$L_i(\lambda) = \mathbf 1\{c_i < \lambda\}$, so with nonconformity $s = -c$ the escape guarantee
$\mathbb P(c_{m+1} < \hat\lambda) \le \alpha$ is exactly split-conformal coverage (Section 2.3).
It is a rate **per defective part**; per shipped part it is this times the line's prevalence.
''')

co(r'''
# THEORY -> VALIDATED: the scalar reduction. CRC on c controls escape (400 draws per alpha).
from src.risk.escape import (part_escape_score, localized_escape_score, crc_escape_threshold,
                             empirical_escape)
rec = {"gt_boxes": [[0, 0, 10, 10], [50, 50, 60, 60]],
       "pred_boxes": [[0, 0, 10, 10], [51, 50, 61, 60], [200, 200, 210, 210]],
       "pred_scores": [0.80, 0.40, 0.95]}
print(f"example part: c_part = {part_escape_score(rec):.2f} (any detection counts), "
      f"c_loc = {localized_escape_score(rec):.2f} (weakest-covered defect)")
rows = []
for alpha in (0.05, 0.10, 0.20):
    esc = []
    for t in range(400):
        r = np.random.default_rng(1000 + t)
        cal, tst = r.beta(2, 3, 60), r.beta(2, 3, 400)
        esc.append(empirical_escape(tst, crc_escape_threshold(cal, alpha)))
    j = int(np.floor(alpha * 61 + 1e-9))
    rows.append(dict(alpha=alpha, mean_escape=round(float(np.mean(esc)), 4), theory_j_over_m1=round(j / 61, 4),
                     se=round(float(np.std(esc) / 20), 4), holds=bool(np.mean(esc) <= alpha + 3 * np.std(esc) / 20)))
display(pd.DataFrame(rows))
''')

md(r'''
### 1.1 · Why cold start breaks conformal risk control (Section 2.4) - **THEORY**

With `m` real defective calibration images CRC certifies the `k`-th smallest real escape score only
when `k/(m+1) <= alpha`. Two consequences: a **refusal floor** (nothing certifiable when
`1/(m+1) > alpha`: below m = 19 at 0.05, below m = 9 at 0.10) and **quantisation** (certifiable
escape levels are multiples of `1/(m+1)`; at m = 15, alpha = 0.10 only the lowest real score is
certifiable, so almost every part goes to review).
''')

co(r'''
ms = np.arange(1, 61)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for a, col in ((0.05, OK[0]), (0.10, OK[1])):
    j = np.floor(a * (ms + 1) + 1e-9)
    axes[0].step(ms, j / (ms + 1), where="post", color=col, lw=2, label=f"alpha = {a}")
    axes[0].axhline(a, color=col, lw=1, ls=":")
axes[0].set_xlabel("real defective calibration images m"); axes[0].set_ylabel("largest certifiable escape level")
axes[0].set_title("CRC can only use multiples of 1/(m+1) of its budget"); axes[0].legend()
axes[1].plot(ms, 1 / (ms + 1), color=OK[0], lw=2)
for a in (0.05, 0.10):
    axes[1].axhline(a, color="0.4", lw=1, ls="--"); axes[1].text(58, a + .004, f"alpha = {a}", ha="right", fontsize=8)
axes[1].axvspan(5, 40, color=OK[4], alpha=.12); axes[1].text(22, .4, "cold start: m = 5-40", ha="center", fontsize=8)
axes[1].set_xlabel("m"); axes[1].set_ylabel("refusal floor 1/(m+1)"); axes[1].set_title("Below the floor CRC must refuse")
fig.tight_layout(); plt.show()
print("m = 19 is the first m that certifies alpha = 0.05; m = 9 the first for 0.10.",
      "KolektorSDD2 had 53 and Magnetic-Tile 59 defective calibration images in the first study.")
''')

# =============================================================== 3 RQs
md(r'''
## 2 · Research questions (Section 3)

| # | Question | Measured by | Fails if | Answered in |
|---|---|---|---|---|
| RQ1 | Can synthetic defects tighten a certified escape threshold at m = 5-40 without breaking the guarantee? | review load at the issued threshold vs CRC at equal m | no tightening in the best case | Stage 1 (§6), Stage 4 (§9) |
| RQ2 | What exactly is guaranteed when the generator is bad? | realised escape under misaligned synthetic data vs Tier H | escape above Tier H beyond sampling error | Stage 0 (§4), Stage 5 (§10) |
| RQ3 | Does a better generator buy certification with fewer real defects? | m-star per generator at a fixed Tier H cap; KS alignment | no difference, or alignment does not predict m-star | §11 |
| RQ4 | How many real defects must a new line collect before it can be certified? | Tier H cap as a function of (m, N, alpha, beta) vs held-out outcomes | planned and realised caps disagree | §3.3, §12 |

Out of scope: beating published mAP on NEU-DET (the claim is denominated in real labelled defects),
and faithfulness-gated triage (dropped from the contributions; Section 13).
''')

# =============================================================== 3 method
md(r'''
## 3 · The method: SPERC (Sections 6 and 8)

**SPERC (Synthetic-Powered Escape-Risk Certification)** calibrates the scalar escape score with the
exact rank-coupled transporter of Synthetic-Powered Predictive Inference (SPI, Bashari et al.,
NeurIPS 2025; `src/risk/spi_exact.py`), using `m` real and `N` synthetic defects, and issues two
statements, not one (`src/risk/sperc.py`):

* **Tier H (hard, SPI Theorem 3.5)** - for *any* synthetic data:
  $\mathbb P(\text{escape}) \le 1 - |\{j \in [m+1] : R_j^+ \le \lceil (1-\alpha)(N+1)\rceil\}|/(m+1)$.
  Depends only on (m, N, alpha, beta), so a plant can compute it before collecting data.
* **Tier N (near-nominal, SPI Theorem 3.3)**:
  $\alpha - \beta - \varepsilon - \tfrac{1}{N+1} \le \mathbb P(\text{escape}) \le \alpha + \beta + \varepsilon$,
  with $\varepsilon$ the (unobservable) distance between real and synthetic order-statistic laws.
* **Refusal** if no beta gives a Tier H cap at or below the plant's `alpha_max`.

Rank window of real rank r among N synthetic scores:
$p_{m,N,r}(k) = \binom{k+r-2}{r-1}\binom{N+m-k-r+2}{m-r+1}/\binom{N+m+1}{m+1}$,
$R_r^- = \max\{t : F(t-1) \le \beta/2\}$, $R_r^+ = \min\{t : F(t) \ge 1-\beta/2\}$.
''')
co(r'''
display(Image(filename=R("figures", "fig15_sperc_architecture.png")) if os.path.exists(R("figures", "fig15_sperc_architecture.png"))
        else F.sperc_architecture_diagram())
''')

co(r'''
# THEORY: the rank law and the windows (depend on m, N, beta only - never on scores)
from src.risk.spi_exact import rank_pmf, rank_windows, tier_h_cap, tier_h_floor
m, N, beta = 15, 1000, 0.05
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for r, col in ((1, OK[0]), (8, OK[2]), (16, OK[1])):
    p = rank_pmf(m, N, r); axes[0].plot(np.arange(1, N + 2), p, color=col, lw=1.6, label=f"r = {r}")
axes[0].set_xlabel("position k among N = 1000 synthetic scores"); axes[0].set_ylabel("p_{m,N,r}(k)")
axes[0].set_title("Where the r-th real order statistic lands (m = 15)"); axes[0].legend()
lo, hi = rank_windows(m, N, beta)
axes[1].vlines(np.arange(1, m + 2), lo, hi, color=OK[0], lw=3)
axes[1].axhline(int(np.ceil(0.95 * (N + 1))), color=OK[1], ls="--", lw=1.2, label="ceil((1-alpha)(N+1)), alpha = 0.05")
axes[1].set_xlabel("real rank r"); axes[1].set_ylabel("synthetic index window [R-, R+]")
axes[1].set_title("Rank windows at beta = 0.05"); axes[1].legend(fontsize=8)
fig.tight_layout(); plt.show()
print(f"sum of every pmf = 1: {all(abs(rank_pmf(m, N, r).sum() - 1) < 1e-10 for r in range(1, m + 2))}; "
      f"Tier H cap here = {tier_h_cap(m, N, 0.05, beta):.4f} (coverage floor {1 - tier_h_cap(m, N, 0.05, beta):.4f})")
''')

md(r'''
### 3.3 · Commissioning table (Section 8.4) - **THEORY**, computed, not measured

The hard escape cap a plant can plan with before collecting data (N = 1,000 synthetic defects).
The cell asserts that every value equals the document's table.
''')
co(r'''
from src.risk.sperc import commissioning_table
DOC = {5: (0.333, 0.167, 0.333), 10: (0.182, 0.182, 0.273), 15: (0.188, 0.125, 0.250),
       25: (0.154, 0.115, 0.231), 40: (0.122, 0.098, 0.195)}
rows = []
for m_, want in DOC.items():
    j = int(np.floor(0.10 * (m_ + 1) + 1e-9))
    got = (tier_h_cap(m_, 1000, .05, .05), tier_h_cap(m_, 1000, .05, .20), tier_h_cap(m_, 1000, .10, .05))
    assert tuple(round(g, 3) for g in got) == want, (m_, got, want)
    rows.append({"real defects m": m_, "CRC at alpha = 0.10": "refuses" if j == 0 else f"up to order statistic {j}",
                 "Tier H, a=.05 b=.05": round(got[0], 3), "Tier H, a=.05 b=.20": round(got[1], 3),
                 "Tier H, a=.10 b=.05": round(got[2], 3)})
display(pd.DataFrame(rows).set_index("real defects m"))
print("All 15 caps equal the document's Section 8.4 table. Cross-check: coverage floor at m = 15,",
      f"a = b = 0.05 is {1 - tier_h_cap(15, 1000, .05, .05):.4f} (SPI's example: at least 0.80).")

fig, ax = plt.subplots(figsize=(7.2, 3.6))
mm = np.arange(3, 81)
for (a, b), col in (((.05, .05), OK[0]), ((.05, .20), OK[2]), ((.10, .05), OK[1])):
    ax.step(mm, [tier_h_cap(int(x), 1000, a, b) for x in mm], where="post", color=col, lw=1.8,
            label=f"alpha = {a}, beta = {b}")
ax.set_xlabel("real defective calibration images m"); ax.set_ylabel("Tier H cap (hard escape bound)")
ax.set_title("RQ4: the planning curve, N = 1000"); ax.legend(fontsize=8); fig.tight_layout(); plt.show()
''')

# =============================================================== Stage 0
md(r'''
## 4 · Stage 0 - SPI on synthetic Beta scores (Section 12) - **VALIDATED**

Stop rule: realised coverage outside SPI's Theorem 3.3 / 3.5 bounds = implementation bug. The unit
tests `tests/test_spi_exact.py` (Stage 0) and `tests/test_sperc.py` are run below, then a
simulation of a new line with m = 15 real defects shows both tiers at work: with a **good**
generator SPERC operates near its target, with a **bad** one it degrades to - but not beyond - its
Tier H cap, and naive pooling has no guarantee at all.
''')
co(r'''
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider",
                    R("tests", "test_spi_exact.py"), R("tests", "test_sperc.py")],
                   cwd=ROOT, capture_output=True, text=True)
print(r.stdout[-1500:] or r.stderr[-1500:])
STAGE0_PASSED = r.returncode == 0
print("STAGE 0:", "PASSED - implementation reproduces SPI's bounds" if STAGE0_PASSED else "FAILED - fix before anything else")
''')
co(r'''
from src.risk.sperc import certify_crc, certify_sperc
from src.risk.baselines import naive_pooled_crc_threshold
from scipy.stats import beta as Beta
REAL = (5, 2)                      # real escape scores ~ Beta(5, 2): true escape = BetaCDF(lambda)
pesc = lambda lam: 0.0 if lam == -np.inf else (1.0 if lam == np.inf else float(Beta.cdf(lam, *REAL)))
rng = np.random.default_rng(0)
rows = []
for gname, syn in (("good generator", (5, 2)), ("bad generator (defects far too easy)", (12, 1.5))):
    e = {k: [] for k in ("crc", "sperc", "naive")}; iss = []
    for _ in range(600):
        cr, cs = rng.beta(*REAL, 15), rng.beta(*syn, 1000)
        c = certify_crc(cr, 0.10); e["crc"].append(pesc(c.threshold)); iss.append(c.issued)
        s = certify_sperc(cr, cs, 0.05, beta=0.05); e["sperc"].append(pesc(s.threshold))
        e["naive"].append(pesc(naive_pooled_crc_threshold(cr, cs, 0.10)))
    rows.append({"generator": gname, "CRC at 0.10": round(np.mean(e["crc"]), 3),
                 "SPERC a=0.05 b=0.05": round(np.mean(e["sperc"]), 3),
                 "Tier N bound": 0.10, "Tier H cap": round(s.tier_h_cap, 4),
                 "naive pooling": round(np.mean(e["naive"]), 3)})
display(pd.DataFrame(rows).set_index("generator"))
print("Mean true escape over 600 calibration draws, m = 15, N = 1000.")
''')

# =============================================================== datasets
md(r'''
## 5 · Datasets, splits and the cold-start protocol (Section 9)

Five datasets with different jobs; the headline rests on the two with clean parts (KolektorSDD2,
Magnetic-Tile), where review load is measurable. NEU-DET, GC10-DET and PCB are validity checks
(escape control only). MVTec AD is the Stage-2 bridge. Splits are train / cal / test with cal never
seen by training or model selection (the detector's own val slice is carved from train), and the
defect-aware allocation (defective 45/30/25, clean 60/15/25) roughly halves every alpha floor.
''')
co(r'''
ROLE = {d: "primary" for d in CFG["primary_datasets"]}; ROLE.update({d: "validity check" for d in CFG["validity_datasets"]})
rows = []
for d in CFG["datasets"]:
    p = W("data", "processed", f"{d}_coco.json")
    if not os.path.exists(p):
        continue
    j = read_json(p)
    n = len(j["images"]); clean = sum(1 for im in j["images"] if im.get("n_boxes", 1) == 0)
    row = dict(dataset=d, role=ROLE.get(d, "-"), images=n, clean=clean, boxes=len(j["annotations"]),
               classes=len(j["categories"]))
    for s in ("train", "val", "cal", "test"):
        sp = W("data", "processed", "splits", f"{d}_{s}.json")
        if os.path.exists(sp):
            row[f"{s} defective"] = sum(1 for im in read_json(sp)["images"] if im.get("n_boxes", 1) > 0)
    if "cal defective" in row:
        row["alpha floor 1/(m+1)"] = round(1 / (row["cal defective"] + 1), 4)
    rows.append(row)
if rows:
    DS = pd.DataFrame(rows).set_index("dataset"); display(DS)
    print("totals:", int(DS.images.sum()), "images,", int(DS.boxes.sum()), "boxes,", int(DS.classes.sum()), "classes",
          "(the document: 9,468 images, 11,543 boxes, 28 classes across the five)")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(DS))
    ax.bar(x, DS.images - DS.clean, color=OK[1], label="defective", width=.6)
    ax.bar(x, DS.clean, bottom=DS.images - DS.clean, color=OK[2], label="clean", width=.6)
    ax.set_xticks(x); ax.set_xticklabels(DS.index); ax.set_ylabel("images"); ax.legend()
    ax.set_title("Only the primary datasets contain clean parts"); fig.tight_layout(); plt.show()
else:
    pending("the prepared datasets (data/processed)", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --datasets kolektor,magnetic_tile,neu,gc10,pcb")
print(f"split settings: {CFG['splits']}")
''')

# =============================================================== detectors
md(r'''
## 6 · Detectors and the Windows / RTX 5060 setup (Sections 10, 12 Stage 3)

RT-DETR-L is the main detector (NMS-free, so calibration targets the model's own output); YOLOv8s
the comparison. Three seeds, primary datasets first, one GPU job at a time. Detector accuracy is
**context only** - the claim is denominated in real defects, not mAP.

Training is not run from the notebook (it takes GPU-days). Run it in a terminal:

```
vqaenv\Scripts\python.exe scripts\check_ready.py
vqaenv\Scripts\python.exe scripts\run_pipeline.py --wait-for-gpu
```
''')
co(r'''
from src.evaluation import seed_analysis as SA
runs = SA.load_runs([W("runs", "results_yolov8s.json"), W("runs", "results_rtdetr.json")])
expected = [(m + ".pt", d, s) for m in CFG["models"] for d in CFG["datasets"] for s in CFG["seeds"]]
done = {(r["model"], r["dataset"], r["seed"]) for r in runs if (r.get("encoding") or "baseline") == "baseline"}
print(f"detector runs complete: {sum(e in done for e in expected)} of {len(expected)} "
      f"({len(CFG['models'])} models x {len(CFG['datasets'])} datasets x {len(CFG['seeds'])} seeds)")
missing = [e for e in expected if e not in done]
if missing:
    print("missing:", ", ".join(f"{m.replace('.pt','')}/{d}/s{s}" for m, d, s in missing[:15]),
          "..." if len(missing) > 15 else "")
if runs:
    t = []
    for metric in ("test_mAP50", "test_mAP50_95"):
        for r in SA.summarise(runs, metric):
            t.append(dict(metric=metric, model=r["model"], dataset=r["dataset"], encoding=r["encoding"],
                          n_seeds=r["n_seeds"], mean=r["mean"], sd=r["std"]))
    acc = pd.DataFrame(t)
    display(acc.pivot_table(index=["model", "dataset", "encoding"], columns="metric", values=["mean", "sd", "n_seeds"]).round(4))
    neu = acc[(acc.dataset == "neu") & (acc.metric == "test_mAP50")]
    lit = pd.DataFrame([dict(method="DSAT (Sci. Reports 2025)", mAP50=0.8314), dict(method="SH-DETR (PLOS One 2025)", mAP50=0.8303)]
                       + [dict(method=f"this work ({r.model}, {r.n_seeds} seed(s))", mAP50=r["mean"]) for _, r in neu.iterrows()])
    print("\nNEU-DET in context (not the claim):"); display(lit)
else:
    pending("trained detectors (runs/results_*.json)", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --wait-for-gpu")
''')

md(r'''
### 6.1 · CRC on the full calibration pool, both escape events - **MEASURED**

The oracle row of every sweep: CRC with all real defective calibration images, 200 re-splits of the
held-out defective images. Refusals count as escape 0 (the batch goes to review) and are never dropped.
''')
co(r'''
p = W("runs", "risk_results.json")
if os.path.exists(p):
    risk = read_json(p); rows = []
    for tag, v in sorted(risk.items()):
        for kind, kv in v.get("kinds", {}).items():
            for a, e in kv["repeated"].items():
                rows.append(dict(detector=tag, kind=kind, alpha=float(a), m=kv["n_cal_defective"],
                                 floor=round(kv["alpha_floor"], 4), mean_escape=round(e["mean_escape_test"], 4),
                                 se=round(e["se"], 4), refusal=round(e["refusal_rate"], 3), holds=e["holds"]))
    if rows:
        RK = pd.DataFrame(rows); display(RK)
        print(f"guarantee holds (mean <= alpha + 3 SE) on {int(RK.holds.sum())}/{len(RK)} operating points")
    else:
        print("risk_results.json predates the two-event format - rerun: scripts\\run_pipeline.py --skip-train")
else:
    pending("runs/risk_results.json", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
''')

# =============================================================== Stage 1
md(r'''
## 7 · Stage 1 - best case and the go / no-go gate (Sections 12, 15) - RQ1

Held-out real GC10 defects play the synthetic role, so the synthetic data is as good as it can ever
be. **Decision rule** (`scripts/run_stage1.py`, fixed before looking at results): GO if, at some
m in {5, 10, 15, 25}, SPERC run with Tier N = target (a) issues in >= 80% of draws, (b) keeps mean
escape <= target + 3 SE, (c) has a Tier H cap within a plant tolerance `alpha_max` and (d) sends
fewer parts to review than CRC at the same m by more than 2 paired SE. NO-GO = write the fallback
paper (Section 14).
''')
co(r'''
def show_stage1(folder, title):
    p = os.path.join(folder, "stage1_verdict.json")
    if not os.path.exists(p):
        return False
    v = read_json(p)
    display(Markdown(f"**{title}: {v['verdict']}** - detector `{v['detector']}`, {v['n_draws']} draws, "
                     f"target {v['target']}, alpha_max in {v['alpha_max_values']} (predictions: `{v['preds_dir']}`)"))
    ev = pd.DataFrame([dict(key=k, **e) for k, x in v["per_dataset"].items() for e in x["evidence"]])
    if len(ev):
        best = ev.sort_values(["passes", "review_gain_vs_crc"], ascending=False).groupby("key").head(4)
        display(best[["key", "m", "beta", "alpha_max", "tier_h_cap", "issued", "escape", "review_sperc",
                      "review_gain_vs_crc", "gain_se", "passes"]].round(4))
    c3 = pd.DataFrame([dict(key=k, **e) for k, x in v["per_dataset"].items() for e in x["c3_same_hard_bound"]])
    if len(c3):
        print("C3 - against CRC run AT SPERC's Tier H cap (same hard guarantee, no operating target):")
        display(c3.round(4).head(8))
    return True

if not show_stage1(W("runs", "stage1"), "Stage 1 verdict"):
    pending("the Stage 1 verdict on the new detectors", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
if show_stage1(W("runs", "stage1_v1preview"), "PREVIEW on the first-study detectors (old splits, not certified)"):
    print("The preview uses detectors whose checkpoints were selected on the calibration images;",
          "it is a signal, not a result.")
''')

# =============================================================== Stage 2
md(r'''
## 8 · Stage 2 - MVTec AD bridge (Sections 9, 12)

SPI's native setting is one score per image. Image-level PatchCore anomaly scores (memory bank from
`train/good` only) play the role of escape scores; stop rule: the transporter must behave as SPI
reports on its own kind of score.
''')
co(r'''
p = W("runs", "mvtec", "bridge_summary.json")
if os.path.exists(p):
    v = read_json(p)
    display(pd.DataFrame(v["categories"]).T)
    print(f"hard-bound violations: {v['hard_bound_violations']}; aligned rows inside Tier N: {v['aligned_inside_tier_n']};",
          "STAGE 2", "PASSED" if v["passes"] else "NEEDS A LOOK")
else:
    pending("the MVTec AD bridge (download MVTec AD to data/raw/MVTEC-AD first)",
            r"vqaenv\Scripts\python.exe scripts\run_mvtec_bridge.py")
''')

# =============================================================== Stage 4
md(r'''
## 9 · Stage 4 - the headline: protocol P1 on the primary datasets (Sections 9.2, 12) - RQ1

Real defective calibration images subsampled to m in {5, 10, 15, 25, 40, full}, 500 draws each,
every draw re-splitting the held-out images; all methods on the same draws. Shown: the headline
detector, part escape, the best available synthetic source, beta = 0.05, mean over seeds (error
bars = sd across seeds).
''')
co(r'''
SUM = W("runs", "coldstart", "summary")
PREVIEW = False
if not os.path.exists(os.path.join(SUM, "seeds_summary.csv")) and \
        os.path.exists(W("runs", "coldstart_v1preview", "summary", "seeds_summary.csv")):
    SUM, PREVIEW = W("runs", "coldstart_v1preview", "summary"), True
    display(Markdown("**PREVIEW** - no sweep on the retrained detectors yet, so Sections 9-12 show the sweep "
                     "on the *first-study* detectors (YOLOv8s, seed 0, old splits: checkpoints were selected "
                     "on the calibration images). A signal, not a result; the real run replaces it."))
SS = pd.read_csv(os.path.join(SUM, "seeds_summary.csv")) if os.path.exists(os.path.join(SUM, "seeds_summary.csv")) else None
if SS is None:
    pending("the cold-start sweep summary", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
else:
    head = CFG.get("headline_model", "rtdetr-l")
    model = head if head in set(SS.model) else sorted(SS.model.unique())[0]
    pref = ["training_free", "anomalydiffusion", "copy_paste", "real_as_synthetic"]
    fig, axes = plt.subplots(len(CFG["primary_datasets"]), 2, figsize=(11, 3.4 * len(CFG["primary_datasets"])), squeeze=False)
    for i, d in enumerate(CFG["primary_datasets"]):
        sub = SS[(SS.model == model) & (SS.dataset == d) & (SS.kind == "part")]
        src = next((s for s in pref if s in set(sub.source)), None)
        if src is None:
            axes[i, 0].set_title(f"{d}: no sweep yet"); continue
        sub = sub[sub.source == src]
        full = sub[sub.method == "crc"].m.max()
        for meth in ("crc", "sperc_tierN", "sperc_nominal", "acq_uncorrected"):
            g = sub[(sub.method == meth) & ((sub.beta.isna()) | (sub.beta == 0.05)) & (sub.m < full)].sort_values("m")
            if not len(g):
                continue
            for j, col in enumerate(("escape_mean", "review_mean")):
                axes[i, j].errorbar(g.m, g[col], yerr=g[col.replace("_mean", "_sd_seeds")].fillna(0),
                                    color=METHOD_COLOR[meth], marker="o", ms=4, lw=1.6, capsize=2, label=METHOD_LABEL[meth])
        orc = sub[(sub.method == "crc") & (sub.m == full)]
        for j, col in enumerate(("escape_mean", "review_mean")):
            if len(orc):
                axes[i, j].axhline(float(orc[col].iloc[0]), color="0.3", ls=":", lw=1.2, label=f"oracle: CRC, all {int(full)} defects")
            axes[i, j].set_xticks([5, 10, 15, 25, 40])
            axes[i, j].set_xlabel("real defects m")
        axes[i, 0].axhline(CFG["certificate"]["target_escape"], color=OK[1], ls="--", lw=1)
        axes[i, 0].set_ylabel("realised escape"); axes[i, 1].set_ylabel("review load")
        axes[i, 0].set_title(f"{d} ({model}, source: {src})" + (" - PREVIEW" if PREVIEW else ""), loc="left")
        if i == 0:
            axes[i, 1].legend(fontsize=7)
    fig.tight_layout(); plt.show()
    print("Refusals count as escape 0 / review 1. Seeds per point:", int(SS.n_seeds.max()),
          "- draws per seed:", int(SS.draws_per_seed.median()))
''')
co(r'''
p = os.path.join(SUM, "mstar_summary.csv")
if os.path.exists(p):
    MS = pd.read_csv(p)
    show = MS[(MS.kind == "part") & MS.method.isin(["crc", "sperc_tierN", "sperc_nominal", "acq_uncorrected"])
              & (MS.beta.isna() | (MS.beta == 0.05))]
    piv = show.pivot_table(index=["model", "dataset", "source", "alpha_max"], columns="method",
                           values="m_star_mean", aggfunc="first")
    print("m-star: real defects needed to reach the full-data operating point (mean over seeds; empty = not reached).")
    print("SPERC must also have a Tier H cap <= alpha_max; acquisition and naive pooling carry no valid guarantee.")
    display(piv)
else:
    pending("the m-star table", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
''')

# =============================================================== Stage 5
md(r'''
## 10 · Stage 5 - misalignment stress and validity (Sections 8.3, 12) - RQ2

Uniform-noise scores, another line's real defects, and an under-trained generator checkpoint play
the synthetic role. Stop rule: realised escape above Tier H beyond sampling error = the claim fails.
Every row that carries a hard bound (SPERC's Tier H; CRC's alpha) is plotted against it; points must
lie on or below the diagonal. With ~1000 rows a per-row 3-SE flag fires about once by chance, so the
verdict uses a one-sided test per row with a Holm correction over all of them.
''')
co(r'''
p = os.path.join(SUM, "validity.csv")
if os.path.exists(p):
    V = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for role, col, mk in (("best_case", OK[2], "o"), ("generator", OK[0], "s"), ("stress", OK[1], "^")):
        g = V[V.source_role == role]
        if len(g):
            ax.errorbar(g.tier_h_cap, g.escape, yerr=3 * g.escape_se, fmt=mk, color=col, ms=4, lw=.6, alpha=.75,
                        label=f"{role} ({len(g)})")
    lim = [0, max(0.05, float(V.tier_h_cap.max()) * 1.05)]
    ax.plot(lim, lim, color="0.2", ls="--", lw=1, label="escape = Tier H cap")
    ax.set_xlabel("Tier H cap"); ax.set_ylabel("realised escape (+/- 3 SE)"); ax.legend(fontsize=8)
    ax.set_title("RQ2: realised escape against its hard cap" + (" (PREVIEW)" if PREVIEW else ""), loc="left")
    fig.tight_layout(); plt.show()
    n3 = int(V.exceeds_hard_bound.fillna(False).astype(bool).sum())
    nh = int(V.get("holm_violation", pd.Series(False, index=V.index)).fillna(False).astype(bool).sum())
    print(f"{len(V)} rows carry a hard bound ({int((V.source_role == 'stress').sum())} from stress sources).")
    print(f"rows above their cap by > 3 SE: {n3} (about {len(V) * 0.00135:.1f} expected by chance at this many rows)")
    print(f"significant violations after Holm correction (family-wise 0.05): {nh} ->",
          "claim HOLDS" if nh == 0 else "CLAIM FAILS - investigate")
else:
    pending("the validity table", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
''')

# =============================================================== RQ3
md(r'''
## 11 · RQ3 - does a better generator buy certification with fewer real defects? (Section 6, C4)

A training-free one-shot generator vs a fine-tuned few-shot one (AnomalyDiffusion), compared by the
real defects each saves at a fixed Tier H cap, plus an alignment diagnostic: the KS distance between
real and synthetic escape scores. Generators: `scripts/generate_synthetic.py`; audit sheets below.
''')
co(r'''
sheets = sorted(glob.glob(W("data", "generated", "*", "*", "_audit_sheet.png")))
for s in sheets[:6]:
    display(Markdown(f"`{os.path.relpath(s, WORK)}`")); display(Image(filename=s, width=720))
if not sheets:
    pending("synthetic defects", r"vqaenv\Scripts\python.exe scripts\generate_synthetic.py --dataset kolektor --generator training_free")
p = os.path.join(SUM, "alignment_vs_mstar.csv")
if os.path.exists(p):
    A = pd.read_csv(p)
    display(A[A.kind == "part"].sort_values(["model", "dataset", "ks_full_pool"]).round(4))
    s = read_json(os.path.join(SUM, "summary.json")).get("alignment_correlation", {})
    print("Spearman(KS, m-star) per detector:", s if s else "needs >= 4 generator/stress sources per detector")
else:
    pending("the alignment table", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
''')

# =============================================================== RQ4
md(r'''
## 12 · RQ4 - planned vs realised caps (Section 8.4)

The commissioning table promises a cap before data exist; the sweep measures escape afterwards. They
must agree: the worst realised escape (any source) never above the cap beyond sampling error.
''')
co(r'''
p = os.path.join(SUM, "planned_vs_realised.csv")
if os.path.exists(p):
    PV = pd.read_csv(p)
    display(PV[PV.kind == "part"].round(4))
    print(f"disagreements: {int((~PV.agree.astype(bool)).sum())} of {len(PV)} "
          "(planned = the Tier H formula at each source's own N, computed before any score is seen;"
          " realised = the worst mean escape across sources)")
else:
    pending("planned-vs-realised caps", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
''')

# =============================================================== Stage 6
md(r'''
## 13 · Stage 6 - protocol P2, full cold start (Sections 9.2, 12)

The detector itself is trained on only k = 5 real defects plus synthetic defects plus all clean
images; calibration on m disjoint real defects; a disjoint half of the synthetic set, never seen by
the P2 detector, is SPERC's synthetic calibration data. Reported whatever it shows.
''')
co(r'''
p = W("runs", "coldstart_p2", "summary", "seeds_summary.csv")
if os.path.exists(p):
    P2 = pd.read_csv(p)
    sel = P2[(P2.kind == "part") & P2.method.isin(["crc", "sperc_tierN", "sperc_nominal"]) & (P2.beta.isna() | (P2.beta == 0.05))]
    display(sel.pivot_table(index=["model", "dataset", "source", "m"], columns="method",
                            values=["escape_mean", "review_mean"], aggfunc="first").round(3))
else:
    pending("protocol P2", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train --p2")
''')

# =============================================================== baselines
md(r'''
## 14 · Baselines (Section 12.1)

1. CRC, real only; 2. naive stacking of real + synthetic (invalid); 3. the old `spi.py` heuristic
(never beats CRC); 4. defect-seeking acquisition without correction (biased); 5. oracle CRC on the
full pool; 6. SPERC with beta in {0.05, 0.10, 0.20} and both generators. The table shows, at m = 15,
every method's escape, review load and the parts labelled to obtain the m defects.
''')
co(r'''
if SS is not None:
    b = SS[(SS.kind == "part") & (SS.m == 15)]
    if len(b):
        cols = ["model", "dataset", "source", "method", "beta", "n_seeds", "issued_mean", "escape_mean", "review_mean", "tier_h_cap"]
        display(b[cols].sort_values(["model", "dataset", "source", "method", "beta"]).round(3).head(60))
    raw = sorted(glob.glob(W("runs", "coldstart", "coldstart_*.json")))
    if raw:
        rr = pd.DataFrame(read_json(raw[0]))
        lab = rr[(rr.method.isin(["crc", "acq_uncorrected"])) & (rr.kind == "part")].pivot_table(
            index=["dataset", "m"], columns="method", values="labelled_parts", aggfunc="mean")
        print(f"parts labelled to obtain m defects ({os.path.basename(raw[0])}): random vs defect-seeking")
        display(lab.round(1))
else:
    pending("the baseline comparison", r"vqaenv\Scripts\python.exe scripts\run_pipeline.py --skip-train")
''')

# =============================================================== established
md(r'''
## 15 · Already established (Section 13.1) and the drift controller

Results measured in the first study (old splits) are quoted as context; the simulation-based ones
are recomputed here.
''')
co(r'''
display(pd.DataFrame([
    ("Escape scalar reduction holds", "CRC on the reduced score: 0.032 / 0.082 / 0.179 at alpha 0.05 / 0.10 / 0.20, 600 trials", "recomputed in Section 1"),
    ("CRC leaves budget unused at small m", "m = 25: refuses at 0.05; 40% of budget at 0.10, 77% at 0.20", "Section 1.1 (theory)"),
    ("Alpha floors are set by defective counts", "KolektorSDD2 0.0185 (53), Magnetic-Tile 0.0167 (59)", "Section 5 (new splits)"),
    ("Defect-seeking acquisition buys certificates", "KolektorSDD2, budget 100: 0% random vs ~83% guided; realised 0.1011 vs 0.10", "baseline 4 in every sweep"),
    ("Three bias corrections fail for one reason", "inverse-propensity 0%, stratified 6.0% best valid, deflated alpha cliff to 0%", "experiments/ACQUISITION_frontier.md"),
    ("Learned nonconformity score does not help", "KolektorSDD2 auto-accept 88.1% -> 0.7%", "experiments/NEGATIVE_learned_nonconformity.md"),
    ("Drift controller", "holds 0.1007 after a regime shift where a static threshold reaches 0.9618", "recomputed below"),
], columns=["result", "evidence (first study)", "where"]).set_index("result"))
''')
co(r'''
# VALIDATED: drift controller vs a static threshold on a simulated regime shift
from src.risk.adaptive import run_adaptive_experiment, simulated_line_loss
sched = lambda t: 1.0 if t < 1500 else 0.45                      # detector quality drops at t = 1500
res = run_adaptive_experiment(simulated_line_loss(sched), 4000, alpha=0.10, gamma=0.02, seed=0)
static_lam = res["lambdas"][1499]
rng = np.random.default_rng(1); f = simulated_line_loss(sched)
static = np.array([f(t, static_lam, rng) for t in range(4000)])
after = slice(2000, 4000)
print(f"after the shift: adaptive escape {res['losses'][after].mean():.4f} vs static threshold {static[after].mean():.4f} (target 0.10)")
fig, ax = plt.subplots(figsize=(8, 3))
k = 200; sm = lambda x: np.convolve(x, np.ones(k) / k, mode="valid")     # trailing window
xs = np.arange(k - 1, 4000)                                                # plotted at the window's end
ax.plot(xs, sm(res["losses"]), color=OK[0], lw=1.6, label="adaptive controller")
ax.plot(xs, sm(static), color=OK[1], lw=1.6, label="static threshold")
ax.axhline(0.10, color="0.3", ls="--", lw=1); ax.axvline(1500, color="0.6", lw=1)
ax.set_xlabel("part"); ax.set_ylabel(f"escape ({k}-part moving mean)"); ax.legend(fontsize=8)
ax.set_title("Exchangeability breaks on a real line: the drift monitor, not SPERC, handles it", loc="left")
fig.tight_layout(); plt.show()
''')

# =============================================================== use it
md(r'''
## 16 · Reading a certificate (for a plant)
''')
co(r'''
rng = np.random.default_rng(3)
c_real, c_syn = rng.beta(5, 2, 15), rng.beta(5, 2, 1000)          # replace with escape_scores(...)
cert = certify_sperc(c_real, c_syn, alpha=0.05, beta=0.05, alpha_max=0.25)
print(f"issued       : {cert.issued}   (False = refuse: every part goes to review)")
print(f"threshold    : {cert.threshold:.4f}   (auto-accept a part only if its highest detection score is below this)")
print(f"Tier N bound : {cert.tier_n_bound:.2f}   (escape bound when synthetic defects resemble real ones, + eps)")
print(f"Tier H cap   : {cert.tier_h_cap:.4f}   (escape bound that holds no matter what)")
print("Both bounds average over the calibration draw; neither is a per-part promise; drift is the monitor's job.")
''')

# =============================================================== status
md(r'''
## 17 · Status against the document (Section 13.2) and limitations (Section 14)

The checklist below is computed from the repository, not typed.
''')
co(r'''
def has(path, text=None, absent=None):
    p = R(path)
    if not os.path.exists(p):
        return False
    s = open(p, encoding="utf-8", errors="ignore").read()
    return (text is None or text in s) and (absent is None or absent not in s)
n_tests = sum(open(f, encoding="utf-8").read().count("\ndef test_") for f in glob.glob(R("tests", "test_*.py")))
readme = open(R("README.md"), encoding="utf-8").read()
checks = [
    ("spi.py replaced by the exact SPI implementation (spi_exact.py)", has("src/risk/spi_exact.py", "def rank_windows") and has("src/risk/spi.py", absent="normal approximation")),
    ("escape unified to the two 0/1 events (escape.py; triage per defective part)", has("src/risk/escape.py", "localized_escape_score") and has("src/risk/triage.py", "per_defective")),
    ("faithfulness-gated triage not claimed as a contribution", "faithfulness-gated" not in readme.lower() or "not a contribution" in readme.lower()),
    ("one operating-point count (recomputed, docs/PROJECT_STATUS.md)", has("docs/PROJECT_STATUS.md", "18/20")),
    ("detector-run count computed, not quoted (Section 6 above)", True),
    ("README test count matches the tests on disk", f"{n_tests} tests" in readme),
    ("requirements.txt lists ultralytics, no Deformable-DETR", has("requirements.txt", "ultralytics", absent="deformable")),
    ("configs/config.yaml describes SPERC, not the removed Kaggle pipeline", has("configs/config.yaml", "SPERC", absent="Swin")),
    ("README does not list src/xai/rollout.py", "rollout.py" not in readme),
    ("SH-DETR quoted as 83.03%", has("docs/RESEARCH_PLAN.md", "83.03")),
    (".gitattributes with * text=auto", has(".gitattributes", "* text=auto")),
]
display(pd.DataFrame(checks, columns=["Section 13.2 item", "done"]).set_index("Section 13.2 item"))
print(f"{n_tests} tests in tests/")
''')
md(r'''
**Limitations to state in the paper (Section 14).** Guarantees are marginal, over calibration draws,
not per part. Drift in the experiments is synthetic. PCB defects are artificially inserted. Tier N
depends on an unobservable distance epsilon; the KS diagnostic estimates alignment but does not
certify it. Tier H caps are loose at small m (e.g. 0.188 at m = 15) - C3 turns that looseness into a
result: *synthetic data can sharpen the operating point when it is good, but it cannot sharpen the
distribution-free worst case below the real-data grid.* Detector accuracy trails the state of the
art on NEU-DET; the claim is in real defects, not mAP.

**Fallback paper, if Stage 1 fails:** "Where conformal escape control breaks at cold start, and why
corrections fail" - the scalar reduction, the quantisation analysis, the defect-seeking acquisition
results, the three refuted corrections and the Stage-1 negative result.
''')


# =============================================================== write
bad = []
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        try:
            compile("".join(c["source"]), f"<cell {i}>", "exec")
        except SyntaxError as e:
            bad.append((i, str(e)))
if bad:
    for i, e in bad:
        print(f"cell {i} does not compile: {e}")
    sys.exit(1)

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3 (vqaenv)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with io.open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}: {len(cells)} cells ({n_code} code), {os.path.getsize(OUT) // 1024} KB")
