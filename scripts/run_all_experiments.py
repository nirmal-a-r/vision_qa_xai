"""
run_all_experiments.py
======================
Everything downstream of training, in one reproducible pass.

Order matters and is enforced here: predictions are cached once, then risk
calibration, drift, triage and the agent all read that same cache. Recomputing
detector outputs per analysis would let the analyses silently disagree with each
other after any change to the inference path.

Outputs (consumed by the notebook):
    runs/preds_<dataset>_<model>.json
    runs/risk_results.json
    runs/faithfulness_results.json
    runs/triage_results.json
    runs/agent_log.json
"""

from __future__ import annotations

import os, sys, json, glob, argparse
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.risk.conformal import (conformal_risk_control, escape_threshold_grid,
                                build_calibration_losses, box_iou_matrix,
                                mondrian_risk_control, RiskNotAchievable)
from src.risk.triage import (build_grid, fit_triage_policy, evaluate_policy,
                             NoCertifiableConfig)

DATASETS = ["neu", "gc10", "pcb", "magnetic_tile", "kolektor"]
ALPHAS = [0.01, 0.05, 0.10, 0.20]
IOU = 0.5


# ---------------------------------------------------------------- predictions
def dump_predictions(dataset, weights, imgsz, split, out_path, conf=0.01):
    """Cache detector output at a very low confidence floor.

    The floor must be far below any threshold calibration might select: if we
    pre-filtered at, say, 0.25, conformal could never certify anything below it
    and the guarantee would silently be conditioned on an arbitrary choice made
    here rather than on the calibration data.
    """
    from ultralytics import YOLO, RTDETR
    M = RTDETR if "rtdetr" in os.path.basename(weights).lower() else YOLO
    model = M(weights)

    img_dir = f"data/yolo/{dataset}/images/{split}"
    lbl_dir = f"data/yolo/{dataset}/labels/{split}"
    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    recs = []
    B = 16
    for i in range(0, len(files), B):
        batch = files[i:i + B]
        for f, r in zip(batch, model.predict(batch, imgsz=imgsz, conf=conf,
                                             verbose=False, device=0)):
            h, w = r.orig_shape
            stem = os.path.splitext(os.path.basename(f))[0]
            gt_b, gt_c = [], []
            lp = os.path.join(lbl_dir, stem + ".txt")
            if os.path.exists(lp):
                for line in open(lp):
                    p = line.split()
                    if len(p) == 5:
                        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                        gt_b.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                                     (cx + bw / 2) * w, (cy + bh / 2) * h])
                        gt_c.append(c)
            bx = r.boxes
            recs.append({
                "id": stem,
                "gt_boxes": [[round(v, 2) for v in b] for b in gt_b],
                "gt_labels": gt_c,
                "pred_boxes": [[round(v, 2) for v in b] for b in bx.xyxy.cpu().numpy().tolist()],
                "pred_scores": [round(v, 5) for v in bx.conf.cpu().numpy().tolist()],
                "pred_labels": [int(v) for v in bx.cls.cpu().numpy().tolist()],
            })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(recs, open(out_path, "w"))
    print(f"  [{dataset}/{split}] {len(recs)} images cached -> {out_path}", flush=True)
    return recs


# ---------------------------------------------------------------- risk
def risk_for_dataset(cal, test, n_trials=60, seed=0):
    """Single-split calibration plus a repeated-split estimate of E[risk].

    The repeated estimate is the honest test of the theorem: CRC bounds the
    expectation over calibration draws, so one split says little on its own.
    """
    lam = escape_threshold_grid(201)
    Lc, Gc, _ = build_calibration_losses(cal, lam, IOU)
    Lt, Gt, _ = build_calibration_losses(test, lam, IOU)

    single, repeated = [], {}
    for a in ALPHAS:
        try:
            l = conformal_risk_control(Lc, lam, a)
            j = int(np.flatnonzero(lam == l)[0])
            single.append(dict(alpha=a, **{"lambda": float(l)}, error=None,
                               escape_cal=float(Lc[:, j].mean()),
                               escape_test=float(Lt[:, j].mean()),
                               covered=bool(Lt[:, j].mean() <= a)))
        except RiskNotAchievable as e:
            single.append(dict(alpha=a, **{"lambda": None}, error=str(e)[:60]))

    rng = np.random.default_rng(seed)
    allL = np.vstack([Lc, Lt])
    n_cal = len(Lc)
    for a in ALPHAS:
        esc, ref = [], 0
        for _ in range(n_trials):
            idx = rng.permutation(len(allL))
            A, Bt = allL[idx[:n_cal]], allL[idx[n_cal:]]
            try:
                l = conformal_risk_control(A, lam, a)
                j = int(np.flatnonzero(lam == l)[0])
                esc.append(float(Bt[:, j].mean()))
            except RiskNotAchievable:
                ref += 1
        repeated[str(a)] = dict(
            mean_escape_test=float(np.mean(esc)) if esc else 0.0,
            std=float(np.std(esc)) if esc else 0.0,
            frac_trials_covered=float(np.mean([e <= a for e in esc])) if esc else 1.0,
            n_trials=n_trials, n_refused=ref, refusal_rate=ref / n_trials,
            n_issued=len(esc))
    return dict(single_split=single, repeated=repeated,
                n_cal=len(Lc), n_test=len(Lt))


def drift_experiment(cal, test, seed=0):
    """Induce covariate shift by biasing the test block toward hard images.

    Hardness is proxied by the top detection score: resampling toward
    low-scoring images simulates a batch the detector finds unfamiliar, which is
    what a new coil or a dimmed lamp looks like downstream. Exchangeability is
    deliberately broken so the degradation can be measured rather than assumed.
    """
    lam = escape_threshold_grid(201)
    Lc, _, _ = build_calibration_losses(cal, lam, IOU)
    rng = np.random.default_rng(seed)
    hard = np.array([max(r["pred_scores"]) if r["pred_scores"] else 0.0 for r in test])
    keep = [i for i, r in enumerate(test) if len(r["gt_boxes"]) > 0]
    hard = hard[keep]
    order = np.argsort(hard)                       # hardest first
    out = {}
    for a in (0.05, 0.10):
        try:
            l = conformal_risk_control(Lc, lam, a)
        except RiskNotAchievable:
            continue
        j = int(np.flatnonzero(lam == l)[0])
        row = {}
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            k = max(5, int(len(order) * (1 - 0.6 * frac)))
            sub = [test[keep[i]] for i in order[:k]]
            Ls, _, _ = build_calibration_losses(sub, lam, IOU)
            row[f"shift_{frac:.2f}"] = round(float(Ls[:, j].mean()), 4)
        out[str(a)] = row
    return out


# ---------------------------------------------------------------- triage
def triage_for_dataset(cal, test, alpha=0.05, delta=0.10, max_false_scrap=0.20):
    def feats(recs):
        s = np.array([max(r["pred_scores"]) if r["pred_scores"] else 0.0 for r in recs])
        d = np.array([len(r["gt_boxes"]) > 0 for r in recs])
        # Faithfulness proxy when no audited map is cached: score concentration.
        # Documented as a proxy rather than presented as the audited metric.
        f = np.array([(np.max(r["pred_scores"]) / (np.sum(r["pred_scores"]) + 1e-9))
                      if r["pred_scores"] else 0.0 for r in recs])
        f = np.clip(f / (f.max() + 1e-9), 0, 1)
        return s, f, d

    sc, fc, dc = feats(cal)
    st, ft, dt = feats(test)
    if dc.sum() == 0 or (~dc).sum() == 0:
        return {"skipped": "dataset has no negatives; triage is not evaluable"}
    grid = build_grid(np.linspace(0.02, 0.60, 15), np.linspace(0.30, 0.95, 14),
                      np.array([0.0, 0.3, 0.5]))
    try:
        sol = fit_triage_policy(sc, fc, dc, grid, alpha, delta,
                                max_false_scrap=max_false_scrap)
    except NoCertifiableConfig as e:
        return {"error": str(e)[:160], "alpha": alpha}
    ev = evaluate_policy(sol.config, st, ft, dt)
    return {**ev, "alpha": alpha, "delta": delta,
            "n_certified": sol.n_certified, "n_configs": sol.n_configs,
            "config": {"lambda_lo": sol.config.lambda_lo,
                       "lambda_hi": sol.config.lambda_hi, "phi": sol.config.phi}}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip_predict", action="store_true")
    a = ap.parse_args()

    results = json.load(open("runs/results_yolov8s.json")) if os.path.exists("runs/results_yolov8s.json") else []
    if os.path.exists("runs/results_rtdetr.json"):
        results += json.load(open("runs/results_rtdetr.json"))
    wmap = {(r["model"], r["dataset"], r.get("encoding", "baseline")): r for r in results}

    risk_all, triage_all = {}, {}
    for name in a.datasets.split(","):
        name = name.strip()
        key = (a.model, name, "baseline")
        if key not in wmap:
            print(f"[{name}] no trained {a.model}, skipping"); continue
        rec = wmap[key]
        tag = f"{name}_{a.model.replace('.pt','')}"
        paths = {}
        for split in ("cal", "test"):
            p = f"runs/preds_{tag}_{split}.json"
            paths[split] = p
            if not (a.skip_predict and os.path.exists(p)):
                if not os.path.exists(rec["weights"]):
                    print(f"[{name}] weights missing: {rec['weights']}"); paths = None; break
                dump_predictions(name, rec["weights"], rec["imgsz"], split, p)
        if not paths:
            continue
        cal = json.load(open(paths["cal"])); test = json.load(open(paths["test"]))
        try:
            r = risk_for_dataset(cal, test)
            r["drift"] = drift_experiment(cal, test)
            risk_all[tag] = r
            print(f"[{name}] risk done  (cal={r['n_cal']} test={r['n_test']})", flush=True)
        except Exception as e:
            print(f"[{name}] risk FAILED: {type(e).__name__}: {e}", flush=True)
        try:
            triage_all[tag] = triage_for_dataset(cal, test)
            print(f"[{name}] triage done", flush=True)
        except Exception as e:
            print(f"[{name}] triage FAILED: {type(e).__name__}: {e}", flush=True)

    if risk_all:
        json.dump(risk_all, open("runs/risk_results.json", "w"), indent=1)
        print("wrote runs/risk_results.json")
    if triage_all:
        json.dump(triage_all, open("runs/triage_results.json", "w"), indent=1)
        print("wrote runs/triage_results.json")


if __name__ == "__main__":
    main()
