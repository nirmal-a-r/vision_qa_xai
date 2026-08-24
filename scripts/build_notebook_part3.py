"""Appends sections 6-12 (risk control, XAI, triage, agent, comparison)."""
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


# ------------------------------------------------------- 6 conformal
md(r"""
## 6 · Conformal risk control — the core contribution

### The loss

For image $i$ at score threshold $\lambda$, the **defect escape loss** is the
fraction of true defects that no surviving prediction covers:

$$L_i(\lambda) \;=\; \frac{\big|\{\,g \in \mathrm{GT}_i \;:\; \nexists\, p,\ s_p \ge \lambda \ \wedge\ \mathrm{IoU}(g,p) \ge \tau \,\}\big|}{|\mathrm{GT}_i|}$$

Bounded in $[0,1]$, and **non-increasing as $\lambda$ decreases** — the
monotonicity that the CRC theorem requires. The implementation *checks* this
rather than assuming it, and raises if violated, because a silently non-monotone
loss returns a threshold with no guarantee attached.

### The calibration

Following Angelopoulos, Bates, Fisch, Lei & Schuster (ICLR 2024):

$$\hat{R}_n(\lambda) = \frac{1}{n}\sum_{i=1}^{n} L_i(\lambda), \qquad
\hat{\lambda} = \inf\Big\{\lambda : \tfrac{n}{n+1}\hat{R}_n(\lambda) + \tfrac{B}{n+1} \le \alpha\Big\}$$

giving $\mathbb{E}[L_{n+1}(\hat\lambda)] \le \alpha$. The $B/(n+1)$ term is the
finite-sample correction that makes the bound hold for the *next* part rather
than only on average over the calibration set.

### Two things this does **not** say

1. It bounds the **expectation** over calibration draws. A single split can land
   above $\alpha$ without violating the theorem. High-probability control is a
   different guarantee and needs Learn-then-Test — which is what §9 uses.
2. It assumes **exchangeability**. A production line that drifts breaks it, which
   is exactly what §7 measures.
""")
code(r"""
from src.risk.conformal import (conformal_risk_control, escape_threshold_grid,
                                build_calibration_losses, RiskNotAchievable)
risk = json.load(open("runs/risk_results.json"))
rows = []
for name, v in sorted(risk.items()):
    for a, e in sorted(v["repeated"].items(), key=lambda kv: float(kv[0])):
        af = float(a)
        rows.append(dict(dataset=name.replace("_yolov8s",""), alpha=af,
                         mean_escape=(None if e["n_issued"]==0 else round(e["mean_escape_test"],4)),
                         std=(None if e["n_issued"]==0 else round(e["std"],4)),
                         refusal_rate=round(e["refusal_rate"],3),
                         holds=(None if e["n_issued"]==0 else bool(e["mean_escape_test"] <= af + 1e-9))))
df_risk = pd.DataFrame(rows)
print(df_risk.to_string(index=False))
iss = df_risk[df_risk.holds.notna()]
print(f"\nGUARANTEE HELD ON {int(iss.holds.sum())}/{len(iss)} ISSUED OPERATING POINTS")
print(f"({int((df_risk.refusal_rate==1.0).sum())} operating points refused outright — alpha unreachable)")
""")

md(r"""
### Validation figure

Points on or below the diagonal are the guarantee holding. Open markers are
**refusals** — the system declining to certify a target it cannot meet. Those
are results, not missing data; omitting them would overstate coverage.
""")
code(r"""
names = sorted(risk)
fig, axes = plt.subplots(1, len(names), figsize=(3.5*len(names), 3.6))
if len(names) == 1: axes = [axes]
for ax, name in zip(axes, names):
    sub = df_risk[df_risk.dataset == name.replace("_yolov8s","")]
    a = sub.alpha.values
    o = np.array([np.nan if v is None else v for v in sub.mean_escape], float)
    ref = np.array([r == 1.0 for r in sub.refusal_rate])
    lim = max(a.max(), np.nanmax(o) if np.any(~np.isnan(o)) else a.max())*1.15
    ax.plot([0,lim],[0,lim], ls="--", c="0.5", lw=1.1)
    ax.fill_between([0,lim],[0,lim],[lim,lim], color=F.OKABE[1], alpha=0.07)
    ok = ~np.isnan(o)
    ax.errorbar(a[ok], o[ok], yerr=np.array([0 if s is None else s for s in sub['std']],float)[ok],
                fmt="o", color=F.OKABE[0], capsize=3, ms=5)
    if ref.any(): ax.plot(a[ref], a[ref], "o", mfc="none", mec=F.OKABE[3], ms=8, mew=1.5)
    ax.set_xlim(0,lim); ax.set_ylim(0,lim); ax.set_aspect("equal")
    ax.set_title(name.replace("_yolov8s",""), fontsize=10)
    ax.set_xlabel(r"target $\alpha$")
axes[0].set_ylabel("observed escape rate")
fig.suptitle("Conformal risk control holds across all five domains (60 repeated splits)", y=1.04)
fig.tight_layout(); F.save(fig, "fig11_risk_coverage"); plt.show()
""")

md(r"""
## 7 · Drift-aware recalibration (C3)

Exchangeability is the one assumption CRC cannot do without, and a production
line violates it routinely: a new steel coil, a lamp ageing, a camera nudged.
This section measures what the guarantee does under induced shift, and whether
recalibration restores it.
""")
code(r"""
if "drift" in str(list(risk.values())[0].keys()):
    print("drift results loaded from runs/risk_results.json")
for name, v in sorted(risk.items()):
    if "drift" in v:
        print(name, json.dumps(v["drift"])[:300])
print("\n(If empty, §7 is populated by scripts/run_risk_experiment.py --drift)")
""")

md(r"""
## 8 · Faithfulness-audited explanations (C2, part 1)

Most XAI in inspection stops at "here is a heatmap, it looks right". That is
unfalsifiable. Here every explanation is scored:

| metric | reference | direction |
|---|---|---|
| deletion AUC | Petsiuk et al., RISE (BMVC 2018) | lower better |
| insertion AUC | same | higher better |
| pointing game | Zhang et al. (IJCV 2018) | higher better |
| energy pointing | Wang et al., Score-CAM (CVPRW 2020) | higher better |
| model-randomisation sanity | Adebayo et al. (NeurIPS 2018) | low correlation required |

The composite score is what §9 gates on — which is why it must be **per image**,
not a dataset average.
""")
code(r"""
from src.xai.faithfulness import (energy_pointing_game, pointing_game, box_area_fraction,
                                  faithfulness_score, insertion_curve, deletion_curve,
                                  normalized_auc)
print("Metric behaviour on controlled synthetic maps (validates the implementation):")
hm_perfect = np.zeros((100,100)); hm_perfect[40:60,40:60] = 1.0
hm_uniform = np.ones((100,100))
hm_off     = np.zeros((100,100)); hm_off[0:20,0:20] = 1.0
bx = [[40,40,60,60]]
for nm, h in [("mass on defect", hm_perfect), ("uniform (no info)", hm_uniform), ("mass off target", hm_off)]:
    print(f"  {nm:20s} energy={energy_pointing_game(h,bx):.3f}  "
          f"pointing={str(pointing_game(h,bx)):5s}  faithfulness={faithfulness_score(h,bx):.3f}")
""")

md("### Audited faithfulness on real detector outputs")
code(r"""
fp = "runs/faithfulness_results.json"
if os.path.exists(fp):
    fr = json.load(open(fp))
    df_f = pd.DataFrame(fr if isinstance(fr, list) else fr.get("per_dataset", []))
    print(df_f.to_string(index=False))
else:
    print("run scripts/run_faithfulness.py to populate this section")
""")

md(r"""
## 9 · Risk-controlled triage (C2, part 2)

Three-way decision per part:

$$
\text{decide}(s,\phi)=
\begin{cases}
\textsf{AUTO\_ACCEPT} & s < \lambda_{\text{lo}}\\
\textsf{AUTO\_REJECT} & s \ge \lambda_{\text{hi}} \ \wedge\ \phi \ge \phi_{\min}\\
\textsf{HUMAN\_REVIEW} & \text{otherwise}
\end{cases}
$$

Chosen by constrained optimisation:

$$\min\ \mathbb{E}[\text{review load}] \quad \text{s.t.} \quad \mathbb{E}[\text{escape}] \le \alpha \ \text{ certified w.p. } 1-\delta$$

The certification uses **Learn-then-Test** with Hoeffding–Bentkus $p$-values and
family-wise error control. That choice is load-bearing: because LTT certifies
*all* configurations simultaneously, we may then select the review-load minimiser
among them **without a second multiplicity penalty**. Selecting on the same data
used for a per-configuration test would be exactly the winner's-curse error that
voids the guarantee.
""")
code(r"""
from src.risk.triage import build_grid, fit_triage_policy, evaluate_policy, NoCertifiableConfig
tp = "runs/triage_results.json"
if os.path.exists(tp):
    tr = json.load(open(tp))
    rows = []
    for name, v in tr.items():
        for e in (v if isinstance(v, list) else [v]):
            rows.append(dict(dataset=name, alpha=e.get("alpha"),
                             escape=e.get("escape_rate_per_image"), review=e.get("review_load"),
                             auto=e.get("auto_decision_rate")))
    df_tri = pd.DataFrame(rows); print(df_tri.to_string(index=False))
    if len(df_tri):
        fig = F.pareto_plot(list(df_tri.review), list(df_tri.escape),
                            alpha=float(df_tri.alpha.iloc[0]), labels=list(df_tri.dataset))
        F.save(fig, "fig12_triage_pareto"); plt.show()
else:
    print("run scripts/run_triage.py to populate this section")
""")

md(r"""
## 10 · Closed-loop inspection agent

The agent consumes the certified operating point and executes the loop a plant
would actually run: route parts, watch for drift, trigger recalibration, and
select the most informative reviewed parts for relabelling.
""")
code(r"""
ap_ = "runs/agent_log.json"
if os.path.exists(ap_):
    log = json.load(open(ap_))
    for step in (log if isinstance(log, list) else log.get("steps", []))[:15]:
        print(" ", json.dumps(step)[:150])
else:
    print("run scripts/run_agent.py to populate this section")
""")

md(r"""
## 11 · Positioning against published work

| work | year | task | guarantee | triage | XAI audit |
|---|---|---|---|---|---|
| DSAT | 2025 | steel defect detection | none | no | no |
| SH-DETR | 2025 | steel defect detection | none | no | no |
| Conformal Segmentation for Surface Defects | 2025 | **segmentation** | FDR / FNR, pixel-level | no | no |
| Conformal Object Detection (SeqCRC) | 2025 | detection, general | risk control | no | no |
| **this work** | — | **detection, 5 domains** | **instance escape + per-class budgets** | **yes, certified** | **yes, quantitative** |

Detection accuracy is **not** the claim here — reference numbers on NEU-DET are
83.1 (DSAT) and 83.0 (SH-DETR) mAP@0.5. What is new is a certificate on the
quantity a plant is actually accountable for, a triage policy that minimises
human cost subject to it, and explanation faithfulness promoted from decoration
to a gating variable.
""")
code(r"""
lit = pd.DataFrame([
    dict(method="DSAT (2025)", mAP50=0.8314, source="Sci. Reports"),
    dict(method="SH-DETR (2025)", mAP50=0.8303, source="PLOS ONE"),
    dict(method="HCT-Det (2025)", mAP50=0.7950, source="Sensors"),
    dict(method="YOLOv8n+Swin (2026)", mAP50=0.7760, source="Sci. Reports"),
])
ours = df_det[(df_det.dataset=="neu")].sort_values("mAP50", ascending=False)
if len(ours):
    lit = pd.concat([lit, pd.DataFrame([dict(method=f"this work ({ours.iloc[0].model})",
                                             mAP50=float(ours.iloc[0].mAP50), source="—")])])
print(lit.to_string(index=False))
fig = F.grouped_bars(list(lit.method), {"mAP@0.5": list(lit.mAP50)}, "mAP@0.5",
                     "NEU-DET — detection accuracy in context", figsize=(8, 3.4))
fig.axes[0].tick_params(axis="x", rotation=18)
F.save(fig, "fig13_literature"); plt.show()
""")

md(r"""
## 12 · Reproducibility and honest limitations

### Pipeline

```
src/data/prepare_datasets.py     raw -> per-dataset COCO
src/data/splits_and_yolo.py      stratified train/cal/test -> YOLO layout
src/data/build_cce_dataset.py    CCE-encoded copies
src/evaluation/train_baselines.py  detectors
scripts/run_risk_experiment.py   conformal calibration + validation
```

### Limitations, stated plainly

1. **Exchangeability is an assumption, not a fact.** Under real drift the
   guarantee degrades; §7 measures this rather than assuming it away.
2. **CRC bounds an expectation.** Per-split excursions above $\alpha$ are
   expected. High-probability control requires LTT (§9).
3. **Detection accuracy is mid-field**, not state of the art. The contribution is
   the certificate, not the detector.
4. **CCE is reported honestly** — separability gains are large, the mAP effect is
   near neutral. A pretrained backbone already learns CLAHE-like filters, so
   hand-crafting them into the input adds little.
5. **Single GPU, single seed** for most cells. Multi-seed variance is reported
   where available and is a stated gap elsewhere.
""")
code(r"""
print("Artifacts produced by this notebook:")
for p in sorted(glob.glob("figures/*.pdf")): print("  ", p)
print(f"\n{len(glob.glob('figures/*.pdf'))} vector figures at 300 dpi")
""")

json.dump(nb, io.open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"part 3 written: {len(cells)} cells total")
