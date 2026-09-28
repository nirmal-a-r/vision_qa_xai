"""
run_stage1.py - Stage 1 of the experimental protocol (project document, Section 12):
the best-case test and the project's go / no-go gate (Section 15, week 2).

Held-out REAL defects play the synthetic role (``real_as_synthetic``), so the
synthetic data is as good as it can ever be (eps ~ 0). If SPERC cannot beat CRC
here, no diffusion generator will rescue it, and the plan switches to the fallback
paper (Section 14). Needs only cached detections - no GPU.

Decision rule (fixed before looking at results; the plan's "no review-load gain
over CRC at equal Tier H cap = stop", made operational):

  Both methods must meet the same plant constraints: operate at the target escape
  rate, and carry a hard (distribution-free) escape bound no worse than the
  plant's alpha_max. CRC at the target satisfies both (its hard bound IS the
  target). SPERC run at alpha = target - beta (Tier N = target) satisfies them
  when its Tier H cap <= alpha_max. At equal m and on the same draws:

  GO     if for some m in {5, 10, 15, 25}, some beta and some alpha_max in the
         configured list, SPERC (i) issues in >= 80% of draws, (ii) has mean escape
         <= target + 3 SE, (iii) has Tier H cap <= alpha_max and (iv) sends fewer
         parts to review than CRC by more than 2 paired standard errors.
  NO-GO  otherwise.

The C3 comparison (CRC run AT SPERC's Tier H cap, i.e. the same hard guarantee
but no operating target) is reported alongside; theory predicts no gain there.

    python scripts/run_stage1.py                                     # new detectors (runs/preds)
    python scripts/run_stage1.py --preds_dir archive/v1_results/preds --model yolov8s \
           --out_dir runs/stage1_v1preview                          # preview on the archived models
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_coldstart_sweep import run_tag, write_csv   # noqa: E402


def decide(rows, target, alpha_max_values, m_values, min_issued=0.8):
    """Apply the Stage-1 decision rule to sweep rows (one dataset/kind/source set)."""
    evidence, c3 = [], []
    for r in rows:
        if r["method"] != "sperc_tierN" or r["m"] not in m_values:
            continue
        g, se = r.get("review_gain_vs_crc"), r.get("review_gain_vs_crc_se") or 0.0
        cap = r.get("tier_h_cap_max", r["tier_h_cap"])
        for amax in alpha_max_values:
            ok = (g is not None and cap is not None and cap <= amax + 1e-12
                  and r["issued"] >= min_issued
                  and r["escape"] <= target + 3 * (r["escape_se"] or 0)
                  and g > 2 * se and g > 0)
            evidence.append({"dataset": r["dataset"], "kind": r["kind"], "m": r["m"],
                             "beta": r["beta"], "alpha_max": amax, "tier_h_cap": cap,
                             "issued": r["issued"], "escape": r["escape"],
                             "escape_se": r["escape_se"], "review_sperc": r["review"],
                             "review_gain_vs_crc": g, "gain_se": se, "passes": bool(ok)})
        if r.get("review_gain_vs_crc_at_cap") is not None:
            c3.append({"dataset": r["dataset"], "kind": r["kind"], "m": r["m"], "beta": r["beta"],
                       "tier_h_cap": cap, "review_gain_vs_crc_at_cap": r["review_gain_vs_crc_at_cap"],
                       "se": r.get("review_gain_vs_crc_at_cap_se")})
    passing = [e for e in evidence if e["passes"]]
    return {"verdict": "GO" if passing else "NO-GO", "n_passing": len(passing),
            "passing": passing, "evidence": evidence, "c3_same_hard_bound": c3}


def main(argv=None):
    try:
        from src.utils.winenv import setup_console
        setup_console()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "config.yaml"))
    ap.add_argument("--datasets", default=None, help="default: config stage1.datasets (gc10)")
    ap.add_argument("--model", default=None, help="detector cache tag; default: config headline_model")
    ap.add_argument("--preds_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_draws", type=int, default=None)
    a = ap.parse_args(argv)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    s1 = cfg.get("stage1", {})
    datasets = a.datasets.split(",") if a.datasets else s1.get("datasets", ["gc10"])
    tag = a.model or cfg.get("headline_model", "rtdetr-l")
    preds_dir = a.preds_dir or os.path.join(ROOT, cfg["paths"]["preds_dir"])
    out_dir = a.out_dir or os.path.join(ROOT, "runs", "stage1")
    m_values = list(s1.get("m_values", [5, 10, 15, 25]))
    n_draws = a.n_draws or s1.get("n_draws", 500)
    target = cfg["certificate"]["target_escape"]
    amaxs = cfg["coldstart"].get("mstar_alpha_max_values", [cfg["certificate"]["alpha_max"]])

    rows = run_tag(tag, cfg, datasets, ["real_as_synthetic"], False, n_draws, preds_dir,
                   out_dir, m_values + ["full"])
    if not rows:
        print(f"Stage 1: no cached {tag} predictions for {datasets} in {preds_dir}")
        return 1
    out = {"stage": 1, "detector": tag, "preds_dir": os.path.relpath(preds_dir, ROOT),
           "datasets": datasets, "m_values": m_values, "n_draws": n_draws, "target": target,
           "alpha_max_values": amaxs, "per_dataset": {}}
    for ds in datasets:
        for kind in cfg["escape_kinds"]:
            sub = [r for r in rows if r["dataset"] == ds and r["kind"] == kind]
            if sub:
                out["per_dataset"][f"{ds}/{kind}"] = decide(sub, target, amaxs, m_values)
    part = [v for k, v in out["per_dataset"].items() if k.endswith("/part")]
    out["verdict"] = "GO" if any(v["verdict"] == "GO" for v in part) else "NO-GO"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage1_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    ev = [dict(key=k, **e) for k, v in out["per_dataset"].items() for e in v["evidence"]]
    if ev:
        write_csv(ev, os.path.join(out_dir, "stage1_evidence.csv"))
    print("\n" + "=" * 72)
    for k, v in out["per_dataset"].items():
        best = max(v["evidence"], key=lambda e: (e["passes"], e["review_gain_vs_crc"] or -9), default=None)
        extra = (f"  best: m={best['m']} beta={best['beta']} alpha_max={best['alpha_max']} "
                 f"gain={best['review_gain_vs_crc']:+.3f} (se {best['gain_se']:.3f}), "
                 f"escape {best['escape']:.3f}, cap {best['tier_h_cap']:.3f}") if best else ""
        print(f"  {k:28s} {v['verdict']:6s} ({v['n_passing']} passing settings){extra}")
    print(f"STAGE 1 VERDICT (part escape, primary event): {out['verdict']}")
    print(f"wrote {os.path.join(out_dir, 'stage1_verdict.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
