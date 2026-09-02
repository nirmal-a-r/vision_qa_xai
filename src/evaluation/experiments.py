"""
experiments.py
==============
Everything downstream of training, in one reproducible pass.

Order matters and is enforced here: predictions are cached once, then risk
calibration, drift, triage and faithfulness all read that same cache.
Recomputing detector outputs per analysis would let the analyses silently
disagree with each other after any change to the inference path.

**One cache format.** This module previously existed as two scripts that wrote
two different caches: a flat runs/preds_<ds>_<model>_<split>.json holding a bare
list, and a richer runs/preds/<ds>_<model>_<split>.json holding a dict with
file_name per record. Only the second can drive the saliency work, since
faithfulness has to re-open the image; only the first had ever been generated for
the two primary datasets. The analyses therefore covered different dataset sets
depending on which one they happened to read. The rich layout is now the single
canonical one, and load_preds still accepts the legacy flat file so existing
caches keep working.

Outputs (consumed by the notebook):
    runs/preds/<dataset>_<model>_<split>.json
    runs/risk_results.json
    runs/triage_results.json
"""

from __future__ import annotations

import os
import json
import glob

import numpy as np

from src.risk.conformal import (conformal_risk_control, escape_threshold_grid,
                                build_calibration_losses, RiskNotAchievable)
from src.risk.triage import (build_grid, fit_triage_policy, evaluate_policy,
                             NoCertifiableConfig)

DATASETS = ["neu", "gc10", "pcb", "magnetic_tile", "kolektor"]
ALPHAS = [0.01, 0.05, 0.10, 0.20]
IOU = 0.5


def preds_path(dataset, model="yolov8s", split="test"):
    """Canonical cache location."""
    return os.path.join("runs", "preds", f"{dataset}_{model}_{split}.json")


def load_preds(path):
    """Return a list of records from either cache layout.

    The legacy flat file stores a bare list whose records key the image as 'id'
    and carry no 'file_name'. Normalising here means every consumer sees one
    shape, and anything needing 'file_name' can check for it explicitly rather
    than crashing halfway through a long loop.
    """
    with open(path) as f:
        d = json.load(f)
    recs = d["records"] if isinstance(d, dict) else d
    for r in recs:
        if "image_id" not in r:
            r["image_id"] = r.get("id")
    return recs


# ---------------------------------------------------------------- predictions
def dump_predictions(dataset, weights, imgsz, split, out_path=None,
                     conf=0.01, yolo_dir="data/yolo", device=0):
    """Cache detector output at a very low confidence floor.

    The floor must be far below any threshold calibration might select: if we
    pre-filtered at, say, 0.25, conformal could never certify anything below it
    and the guarantee would silently be conditioned on an arbitrary choice made
    here rather than on the calibration data.

    file_name is recorded so the saliency pass can re-open the exact image this
    row describes instead of re-deriving a path and hoping it matches.
    """
    from ultralytics import YOLO, RTDETR
    base = os.path.basename(weights).lower()
    M = RTDETR if "rtdetr" in base else YOLO
    model = M(weights)

    if out_path is None:
        tag = "rtdetr" if "rtdetr" in base else "yolov8s"
        out_path = preds_path(dataset, tag, split)

    img_dir = os.path.join(yolo_dir, dataset, "images", split)
    lbl_dir = os.path.join(yolo_dir, dataset, "labels", split)
    files = sorted(glob.glob(os.path.join(img_dir, "*")))
    recs = []
    B = 16
    for i in range(0, len(files), B):
        batch = files[i:i + B]
        for f, r in zip(batch, model.predict(batch, imgsz=imgsz, conf=conf,
                                             verbose=False, device=device)):
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
                "image_id": stem,
                "id": stem,
                "file_name": os.path.abspath(f),
                "orig_hw": [int(h), int(w)],
                "gt_boxes": [[round(v, 2) for v in b] for b in gt_b],
                "gt_labels": gt_c,
                "pred_boxes": [[round(v, 2) for v in b]
                               for b in bx.xyxy.cpu().numpy().tolist()],
                "pred_scores": [round(v, 5) for v in bx.conf.cpu().numpy().tolist()],
                "pred_labels": [int(v) for v in bx.cls.cpu().numpy().tolist()],
            })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"split": split, "weights": weights, "imgsz": imgsz,
                   "records": recs}, f)
    print(f"  [{dataset}/{split}] {len(recs)} images cached -> {out_path}", flush=True)
    return recs


# ---------------------------------------------------------------- risk
def risk_for_dataset(cal, test, n_trials=60, seed=0):
    """Single-split calibration plus a repeated-split estimate of E[risk].

    The repeated estimate is the honest test of the theorem: CRC bounds the
    expectation over calibration draws, so one split says little on its own.

    A refusal contributes 0 to the escape mean, because a refused batch is
    handed to human review and escapes nothing. Averaging only over the splits
    that issued a certificate would be a selection on the calibration draw, and
    is what previously made a satisfied guarantee read as violated.
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
        esc, cond, ref = [], [], 0
        for _ in range(n_trials):
            idx = rng.permutation(len(allL))
            A, Bt = allL[idx[:n_cal]], allL[idx[n_cal:]]
            try:
                l = conformal_risk_control(A, lam, a)
                j = int(np.flatnonzero(lam == l)[0])
                v = float(Bt[:, j].mean())
                esc.append(v)
                cond.append(v)
            except RiskNotAchievable:
                ref += 1
                esc.append(0.0)
        repeated[str(a)] = dict(
            # Unconditional over all draws: this is the quantity CRC bounds and
            # the only one that should be compared against alpha.
            mean_escape_test=float(np.mean(esc)) if esc else 0.0,
            std=float(np.std(esc)) if esc else 0.0,
            # Conditional on issuing. Reported for transparency, NOT the
            # guarantee; it is high exactly when refusal is doing its job.
            mean_escape_issued=float(np.mean(cond)) if cond else None,
            frac_trials_covered=float(np.mean([e <= a for e in esc])) if esc else 1.0,
            n_trials=n_trials, n_refused=ref, refusal_rate=ref / n_trials,
            n_issued=len(cond))
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
    hard = np.array([max(r["pred_scores"]) if r["pred_scores"] else 0.0 for r in test])
    keep = [i for i, r in enumerate(test) if len(r["gt_boxes"]) > 0]
    if not keep:
        return {}
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
def triage_for_dataset(cal, test, alpha=0.05, delta=0.10, max_false_scrap=0.20,
                       faith_cal=None, faith_test=None):
    """Fit and evaluate the auto-accept / review / auto-scrap policy.

    faith_cal / faith_test accept audited faithfulness keyed by image_id. When
    absent, the score-concentration proxy is used AND the result records
    faithfulness_source='proxy'. The earlier version substituted the proxy
    silently and then reported the outcome as though the audited metric had been
    used, which is how the faithfulness gate came to look meaningful when it was
    never actually fed.
    """
    def feats(recs, faith):
        s = np.array([max(r["pred_scores"]) if r["pred_scores"] else 0.0 for r in recs])
        d = np.array([len(r["gt_boxes"]) > 0 for r in recs])
        used = False
        if faith:
            f = np.array([faith.get(str(r.get("image_id")), np.nan) for r in recs],
                         dtype=float)
            if not np.isnan(f).all():
                f = np.where(np.isnan(f), np.nanmedian(f), f)
                used = True
        if not used:
            # Score concentration, used only when no audited map is cached.
            f = np.array([(np.max(r["pred_scores"]) / (np.sum(r["pred_scores"]) + 1e-9))
                          if r["pred_scores"] else 0.0 for r in recs])
            f = np.clip(f / (f.max() + 1e-9), 0, 1)
        return s, f, d, used

    sc, fc, dc, used_cal = feats(cal, faith_cal)
    st, ft, dt, used_test = feats(test, faith_test)
    source = "audited" if (used_cal and used_test) else "proxy"
    if dc.sum() == 0 or (~dc).sum() == 0:
        return {"skipped": "dataset has no negatives; triage is not evaluable"}
    grid = build_grid(np.linspace(0.02, 0.60, 15), np.linspace(0.30, 0.95, 14),
                      np.array([0.0, 0.3, 0.5]))
    try:
        sol = fit_triage_policy(sc, fc, dc, grid, alpha, delta,
                                max_false_scrap=max_false_scrap)
    except NoCertifiableConfig as e:
        return {"error": str(e)[:160], "alpha": alpha,
                "faithfulness_source": source}
    ev = evaluate_policy(sol.config, st, ft, dt)
    return {**ev, "alpha": alpha, "delta": delta,
            "n_certified": sol.n_certified, "n_configs": sol.n_configs,
            "faithfulness_source": source,
            "config": {"lambda_lo": sol.config.lambda_lo,
                       "lambda_hi": sol.config.lambda_hi, "phi": sol.config.phi}}


# ---------------------------------------------------------------- driver
def load_faithfulness(dataset, key="faithfulness"):
    """image_id -> composite faithfulness, or None when not computed."""
    p = f"runs/faithfulness_{dataset}.json"
    if not os.path.exists(p):
        return None
    try:
        rows = json.load(open(p))
    except Exception:
        return None
    out = {str(r["image_id"]): float(r[key]) for r in rows if key in r}
    return out or None


def run_all(model="yolov8s.pt", datasets=None, skip_predict=True,
            n_trials=60, verbose=True):
    """Predictions -> risk -> drift -> triage for every trained dataset."""
    datasets = datasets or DATASETS
    if isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(",")]

    results = []
    for p in ("runs/results_yolov8s.json", "runs/results_rtdetr.json"):
        if os.path.exists(p):
            results += json.load(open(p))
    # Interrupted runs record epochs=0; they must not supply weights here.
    results = [r for r in results if r.get("epochs", 0) > 0]
    wmap = {(r["model"], r["dataset"], r.get("encoding") or "baseline"): r
            for r in results}

    tag_of = "rtdetr" if "rtdetr" in model.lower() else "yolov8s"
    risk_all, triage_all = {}, {}
    for name in datasets:
        rec = wmap.get((model, name, "baseline"))
        if rec is None:
            if verbose:
                print(f"[{name}] no trained {model}, skipping")
            continue
        tag = f"{name}_{tag_of}"
        paths, ok = {}, True
        for split in ("cal", "test"):
            p = preds_path(name, tag_of, split)
            paths[split] = p
            if skip_predict and os.path.exists(p):
                continue
            if not os.path.exists(rec["weights"]):
                if verbose:
                    print(f"[{name}] weights missing: {rec['weights']}")
                ok = False
                break
            dump_predictions(name, rec["weights"], rec["imgsz"], split, p)
        if not ok:
            continue

        cal, test = load_preds(paths["cal"]), load_preds(paths["test"])
        faith = load_faithfulness(name)
        try:
            r = risk_for_dataset(cal, test, n_trials=n_trials)
            r["drift"] = drift_experiment(cal, test)
            risk_all[tag] = r
            if verbose:
                print(f"[{name}] risk done  (cal={r['n_cal']} test={r['n_test']})",
                      flush=True)
        except Exception as e:
            print(f"[{name}] risk FAILED: {type(e).__name__}: {e}", flush=True)
        try:
            triage_all[tag] = triage_for_dataset(cal, test, faith_cal=faith,
                                                 faith_test=faith)
            if verbose:
                print(f"[{name}] triage done "
                      f"({triage_all[tag].get('faithfulness_source', '-')})", flush=True)
        except Exception as e:
            print(f"[{name}] triage FAILED: {type(e).__name__}: {e}", flush=True)

    if risk_all:
        json.dump(risk_all, open("runs/risk_results.json", "w"), indent=1)
        print("wrote runs/risk_results.json")
    if triage_all:
        json.dump(triage_all, open("runs/triage_results.json", "w"), indent=1)
        print("wrote runs/triage_results.json")
    return risk_all, triage_all


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip_predict", action="store_true")
    ap.add_argument("--n_trials", type=int, default=60)
    a = ap.parse_args()
    run_all(a.model, a.datasets, a.skip_predict, a.n_trials)
