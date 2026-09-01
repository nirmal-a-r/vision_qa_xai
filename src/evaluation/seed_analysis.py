"""
seed_analysis.py
================
Aggregate repeated-seed runs into the statistics a reviewer will ask for.

Two things this does that a plain mean-and-std does not:

**Pairs by seed.** The CCE ablation compares two encodings trained from the same
initialisation and the same data order. Those runs are *paired*, so comparing
group means throws away the pairing and inflates the variance with between-seed
noise that cancels exactly. A paired test on the per-seed differences is both
more powerful and the correct model of how the experiment was run.

**Refuses to over-claim at n=3.** Three seeds is enough to report a mean and a
spread; it is not enough for a t-test to mean much, and a p-value from three
paired differences deserves to be labelled as such rather than quoted as if it
came from thirty. The Wilcoxon signed-rank test cannot even reach p < 0.05 at
n = 3 (its minimum attainable two-sided p is 0.25), so the effect size and the
per-seed signs carry the argument, not the p-value.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np


def load_runs(paths=("runs/results_yolov8s.json", "runs/results_rtdetr.json")):
    rows = []
    for p in paths:
        if os.path.exists(p):
            rows += json.load(open(p))
    # Guard against interrupted runs sneaking back in; train_baselines refuses to
    # record them now, but older files may still contain some.
    return [r for r in rows if r.get("train_seconds", 0) > 0 and r.get("epochs", 0) > 0]


def per_config(rows, metric="test_mAP50"):
    """(model, dataset, encoding) -> {seed: value}"""
    out = defaultdict(dict)
    for r in rows:
        key = (r["model"], r["dataset"], r.get("encoding", "baseline"))
        out[key][r["seed"]] = float(r[metric])
    return out


def summarise(rows, metric="test_mAP50"):
    """mean / std / n per configuration, sorted for stable display."""
    cfg = per_config(rows, metric)
    recs = []
    for (model, ds, enc), by_seed in sorted(cfg.items()):
        v = np.array(list(by_seed.values()), dtype=float)
        recs.append(dict(model=model, dataset=ds, encoding=enc, n_seeds=v.size,
                         mean=round(float(v.mean()), 4),
                         std=round(float(v.std(ddof=1)), 4) if v.size > 1 else None,
                         seeds=sorted(by_seed)))
    return recs


def paired_ablation(rows, metric="test_mAP50", a="baseline", b="cce"):
    """Per-seed paired comparison of two encodings, per (model, dataset).

    Only seeds present in BOTH arms are used - an unpaired seed would silently
    turn the paired comparison back into a two-sample one.
    """
    cfg = per_config(rows, metric)
    out = []
    keys = {(m, d) for (m, d, e) in cfg}
    for (model, ds) in sorted(keys):
        A = cfg.get((model, ds, a), {})
        B = cfg.get((model, ds, b), {})
        shared = sorted(set(A) & set(B))
        if not shared:
            continue
        diffs = np.array([B[s] - A[s] for s in shared], dtype=float)
        rec = dict(model=model, dataset=ds, n_pairs=len(shared),
                   mean_a=round(float(np.mean([A[s] for s in shared])), 4),
                   mean_b=round(float(np.mean([B[s] for s in shared])), 4),
                   mean_diff=round(float(diffs.mean()), 4),
                   diff_pp=round(float(diffs.mean() * 100), 2),
                   n_better=int((diffs > 0).sum()), n_worse=int((diffs < 0).sum()))
        if len(shared) >= 2:
            rec["std_diff"] = round(float(diffs.std(ddof=1)), 4)
            # Cohen's dz for paired differences; undefined if every diff is equal
            sd = diffs.std(ddof=1)
            rec["dz"] = round(float(diffs.mean() / sd), 3) if sd > 1e-12 else None
        if len(shared) >= 3:
            try:
                from scipy.stats import wilcoxon
                rec["wilcoxon_p"] = round(float(wilcoxon(diffs).pvalue), 4)
                # At n=3 the smallest attainable two-sided p is 0.25, so this can
                # never reach significance however large the effect. Recorded to
                # be transparent about the ceiling rather than to be interpreted.
                rec["p_floor_at_n"] = 0.25 if len(shared) == 3 else None
            except Exception:
                rec["wilcoxon_p"] = None
        out.append(rec)
    return out


def pooled_effect(pairs):
    """Effect pooled across datasets, with the per-dataset signs kept visible.

    Sign agreement matters more than the pooled mean at this sample size: three
    datasets all moving the same way is weak-but-real evidence, whereas a pooled
    mean can be dominated by one dataset with a large swing.
    """
    if not pairs:
        return {}
    d = np.array([p["mean_diff"] for p in pairs], dtype=float)
    return dict(n_datasets=len(pairs),
                pooled_diff_pp=round(float(d.mean() * 100), 2),
                datasets_better=int((d > 0).sum()),
                datasets_worse=int((d < 0).sum()),
                verdict=("consistent gain" if (d > 0).all() else
                         "consistent loss" if (d < 0).all() else
                         "mixed - no consistent direction"))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="test_mAP50")
    a = ap.parse_args()

    rows = load_runs()
    print(f"{len(rows)} valid runs\n")

    print("PER-CONFIGURATION (mean +/- std across seeds)")
    for r in summarise(rows, a.metric):
        sd = f"+/- {r['std']:.4f}" if r["std"] is not None else "  (1 seed) "
        print(f"  {r['model']:12s} {r['dataset']:14s} {r['encoding']:9s} "
              f"{r['mean']:.4f} {sd}  n={r['n_seeds']}")

    print("\nPAIRED CCE ABLATION (per seed, baseline -> cce)")
    pairs = paired_ablation(rows, a.metric)
    if not pairs:
        print("  no paired seeds yet")
    for p in pairs:
        extra = ""
        if "wilcoxon_p" in p and p["wilcoxon_p"] is not None:
            extra = f"  wilcoxon p={p['wilcoxon_p']}"
            if p.get("p_floor_at_n"):
                extra += f" (floor {p['p_floor_at_n']} at n=3)"
        print(f"  {p['model']:12s} {p['dataset']:14s} n={p['n_pairs']} "
              f"{p['mean_a']:.4f} -> {p['mean_b']:.4f}  {p['diff_pp']:+.2f}pp "
              f"({p['n_better']}+/{p['n_worse']}-){extra}")
    pe = pooled_effect(pairs)
    if pe:
        print(f"\n  pooled: {pe['pooled_diff_pp']:+.2f}pp over {pe['n_datasets']} datasets "
              f"({pe['datasets_better']} better / {pe['datasets_worse']} worse) "
              f"-> {pe['verdict']}")
