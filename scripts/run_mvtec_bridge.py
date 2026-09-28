"""
run_mvtec_bridge.py - Stage 2 of the experimental protocol (project document,
Section 12): the MVTec AD bridge.

SPI was evaluated on one score per image; MVTec AD (Bergmann et al., IJCV 2021) is
the benchmark reviewers know. Each category's image-level anomaly scores
(PatchCore-lite, src/evaluation/anomaly_scores.py; memory bank from train/good
only) play the role of escape scores: a defective test image "escapes" when its
score is below the threshold, and a good test image is "reviewed" when its score
reaches it. The held-out test images are split into calibration and test halves
(re-split on every draw) and swept exactly like the detector sweep.

Stop rule: the transporter must behave as SPI reports on its own kind of score -
with real_as_synthetic (aligned) data realised escape sits inside the Tier N band,
and for every source (including misaligned cross-category and noise data) it stays
under Tier H beyond sampling error.

    python scripts/run_mvtec_bridge.py                       # all categories found
    python scripts/run_mvtec_bridge.py --categories bottle,cable --n_draws 200

Data: download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad
(CC BY-NC-SA 4.0) and unpack it to data/raw/MVTEC-AD/<category>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_coldstart_sweep import sweep_one, write_csv   # noqa: E402

NOISE_N = 1000


def load_or_score(root, category, out_dir, mcfg, device=None):
    cache = os.path.join(out_dir, f"{category}_patchcore.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    from src.evaluation.anomaly_scores import PatchCoreLite, mvtec_category_files
    good, test = mvtec_category_files(root, category)
    pc = PatchCoreLite(mcfg.get("backbone", "wide_resnet50_2"), mcfg.get("image_size", 224),
                       mcfg.get("coreset_fraction", 0.1), device=device).fit(good)
    s = pc.score([p for p, _ in test])
    recs = [{"file": os.path.relpath(p, root), "defect_type": t, "score": float(v)}
            for (p, t), v in zip(test, s)]
    os.makedirs(out_dir, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(recs, f)
    return recs


def image_auroc(recs):
    y = np.array([r["defect_type"] != "good" for r in recs])
    s = np.array([r["score"] for r in recs])
    if y.all() or (~y).all():
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--root", default=None, help="MVTec AD root (default data/raw/MVTEC-AD)")
    ap.add_argument("--categories", default=None)
    ap.add_argument("--n_draws", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "runs", "mvtec"))
    a = ap.parse_args(argv)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg.get("mvtec", {})
    root = a.root or os.path.join(ROOT, cfg["paths"].get("raw_dir", "data/raw"), "MVTEC-AD")
    if not os.path.isdir(root):
        print(f"MVTec AD not found at {root}. Download it (see the docstring) to run Stage 2.")
        return 1
    cats = (a.categories.split(",") if a.categories else
            (sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "train")))
             if mcfg.get("categories", "all") == "all" else list(mcfg["categories"])))
    target = cfg["certificate"]["target_escape"]
    betas = cfg["certificate"]["betas"]
    n_draws = a.n_draws or mcfg.get("n_draws", 500)
    m_values = list(mcfg.get("m_values", [5, 10, 15, 25])) + ["full"]
    rng = np.random.default_rng(cfg["coldstart"]["seed"])

    from src.utils.winenv import keep_awake, setup_console
    setup_console()
    with keep_awake("MVTec AD anomaly scoring"):
        scores = {c: load_or_score(root, c, a.out_dir, mcfg, a.device) for c in cats}
    rows, summary = [], {}
    for i, c in enumerate(cats):
        recs = scores[c]
        perm = np.random.default_rng(0).permutation(len(recs))     # fixed cal/test block sizes
        recs = [recs[j] for j in perm]
        s = np.array([r["score"] for r in recs])
        defective = np.array([r["defect_type"] != "good" for r in recs])
        c_all = np.where(defective, s, np.nan)
        n_cal = len(recs) // 2
        other = cats[(i + 1) % len(cats)] if len(cats) > 1 else None
        srcs = {"real_as_synthetic": None,
                "uniform_noise": np.random.default_rng(7).uniform(s.min(), s.max(), NOISE_N)}
        if other and other != c:
            srcs[f"cross:{other}"] = np.array([r["score"] for r in scores[other] if r["defect_type"] != "good"])
        crow = []
        for src, syn in srcs.items():
            rr = sweep_one(c_all, s, n_cal, syn, src, m_values, n_draws, target, betas,
                           cfg["certificate"]["alpha_max"], rng,
                           tie_break=False, acq_gamma=None)        # continuous scores: no ties
            for r in rr:
                r.update(dataset=f"mvtec/{c}", kind="image", model="patchcore_lite", seed=0,
                         primary=False)
            crow += rr
        aligned = [r for r in crow if r["source"] == "real_as_synthetic" and r["method"] == "sperc_nominal"]
        summary[c] = {"auroc": image_auroc(recs), "n_test": len(recs), "n_defective": int(defective.sum()),
                      "hard_bound_violations": int(sum(1 for r in crow if r.get("exceeds_hard_bound"))),
                      "aligned_rows": len(aligned),
                      "aligned_inside_tier_n": int(sum(1 for r in aligned if r.get("inside_tier_n")))}
        print(f"[mvtec/{c}] AUROC {summary[c]['auroc']}  violations {summary[c]['hard_bound_violations']}"
              f"  aligned inside Tier N {summary[c]['aligned_inside_tier_n']}/{summary[c]['aligned_rows']}",
              flush=True)
        rows += crow
    if not rows:
        return 1
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "bridge_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    write_csv(rows, os.path.join(a.out_dir, "bridge_rows.csv"))
    from summarize_sperc import validity as _validity
    val = _validity(rows)
    holm_v = sum(1 for v in val if v.get("holm_violation"))
    total_v = sum(v["hard_bound_violations"] for v in summary.values())
    al = sum(v["aligned_rows"] for v in summary.values())
    ins = sum(v["aligned_inside_tier_n"] for v in summary.values())
    verdict = {"categories": summary, "hard_bound_violations": total_v,
               "expected_by_chance_3se": round(len(val) * 0.00135, 2),
               "holm_significant_violations": holm_v,
               "aligned_inside_tier_n": f"{ins}/{al}",
               "passes": bool(holm_v == 0 and (al == 0 or ins / al >= 0.9))}
    with open(os.path.join(a.out_dir, "bridge_summary.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    print(f"\nStage 2: {holm_v} significant hard-bound violations (Holm; {total_v} rows > 3 SE); "
          f"aligned inside Tier N {ins}/{al}; "
          f"{'PASS' if verdict['passes'] else 'CHECK'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
