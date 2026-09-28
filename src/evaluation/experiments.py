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

from src.risk.escape import KINDS, crc_escape_threshold, escape_scores
from src.risk.triage import (build_grid, fit_triage_policy, evaluate_policy,
                             NoCertifiableConfig)

# The plan's five datasets (Section 9); DAGM stays available via --datasets.
DATASETS = ["kolektor", "magnetic_tile", "neu", "gc10", "pcb"]
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
                     conf=0.001, yolo_dir="data/yolo", device=0):
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
def _escape_arrays(records, kind, iou=IOU):
    return escape_scores(records, kind, iou)


def risk_for_dataset(cal, test, n_trials=200, seed=0, kinds=KINDS):
    """CRC on the two 0/1 escape events (src/risk/escape.py), full calibration pool.

    Single split plus a repeated-split estimate of E[escape]: CRC bounds the
    expectation over calibration draws, so one split says little on its own. Each
    repetition re-partitions the held-out defective images into calibration and
    test blocks of the original sizes.

    A refusal contributes 0 to the escape mean, because a refused batch is handed
    to human review and escapes nothing. Averaging only over the splits that
    issued a certificate would be a selection on the calibration draw, and is what
    previously made a satisfied guarantee read as violated.
    """
    out = {"n_cal_parts": len(cal), "n_test_parts": len(test), "kinds": {}}
    rng = np.random.default_rng(seed)
    for kind in kinds:
        cc, ct = _escape_arrays(cal, kind), _escape_arrays(test, kind)
        single, repeated = [], {}
        for a in ALPHAS:
            lam = crc_escape_threshold(cc, a)
            issued = bool(np.isfinite(lam))
            single.append(dict(alpha=a, issued=issued,
                               threshold=float(lam) if issued else None,
                               escape_test=float(np.mean(ct < lam)) if ct.size else None))
        allc = np.concatenate([cc, ct])
        n_c = cc.size
        for a in ALPHAS:
            esc, cond, ref = [], [], 0
            for _ in range(n_trials):
                idx = rng.permutation(allc.size)
                A, B = allc[idx[:n_c]], allc[idx[n_c:]]
                lam = crc_escape_threshold(A, a)
                if not np.isfinite(lam):
                    ref += 1
                    esc.append(0.0)
                    continue
                v = float(np.mean(B < lam))
                esc.append(v)
                cond.append(v)
            e = np.asarray(esc)
            repeated[str(a)] = dict(
                mean_escape_test=float(e.mean()),                 # what CRC bounds
                se=float(e.std(ddof=1) / np.sqrt(e.size)) if e.size > 1 else 0.0,
                mean_escape_issued=float(np.mean(cond)) if cond else None,   # NOT the guarantee
                holds=bool(e.mean() <= a + 3 * (e.std(ddof=1) / np.sqrt(e.size) if e.size > 1 else 0)),
                n_trials=n_trials, n_refused=ref, refusal_rate=ref / n_trials,
                n_issued=len(cond))
        out["kinds"][kind] = dict(single_split=single, repeated=repeated,
                                  n_cal_defective=int(cc.size), n_test_defective=int(ct.size),
                                  alpha_floor=float(1.0 / (cc.size + 1)))
    return out


def drift_experiment(cal, test, seed=0):
    """Induce covariate shift by biasing the test block toward hard images.

    Part escape (the primary event). Hardness is proxied by the top detection
    score: resampling toward low-scoring images simulates a batch the detector
    finds unfamiliar, which is what a new coil or a dimmed lamp looks like
    downstream. Exchangeability is deliberately broken so the degradation can be
    measured rather than assumed; the drift monitor (src/risk/adaptive.py) is
    what a plant runs against it.
    """
    cc = _escape_arrays(cal, "part")
    ct = _escape_arrays(test, "part")
    if ct.size == 0:
        return {}
    order = np.argsort(np.where(np.isfinite(ct), ct, -1.0))    # hardest first
    out = {}
    for a in (0.05, 0.10):
        lam = crc_escape_threshold(cc, a)
        if not np.isfinite(lam):
            continue
        row = {}
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            k = max(5, int(len(order) * (1 - 0.6 * frac)))
            row[f"shift_{frac:.2f}"] = round(float(np.mean(ct[order[:k]] < lam)), 4)
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
    # The faithfulness gate is dropped from the paper's claims (project document,
    # Sections 3 and 11.1: it was vacuous as built), so phi is fixed at 0 here and
    # the policy is the plain two-threshold accept / review / reject rule.
    grid = build_grid(np.linspace(0.02, 0.60, 15), np.linspace(0.30, 0.95, 14),
                      np.array([0.0]))
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
            n_trials=200, verbose=True):
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

    # Same tag as dump_ultralytics_preds.py writes (weights stem): yolov8s, rtdetr-l.
    tag_of = os.path.basename(model).replace(".pt", "")
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
                print(f"[{name}] risk done  (cal={r['n_cal_parts']} test={r['n_test_parts']})",
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

    # merge, so running the two detectors one after the other keeps both
    for path, new in (("runs/risk_results.json", risk_all), ("runs/triage_results.json", triage_all)):
        if not new:
            continue
        old = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    old = json.load(f)
            except Exception:
                old = {}
        old.update(new)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(old, f, indent=1)
        print(f"wrote {path} ({len(new)} entries updated)")
    return risk_all, triage_all


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip_predict", action="store_true")
    ap.add_argument("--n_trials", type=int, default=200)
    a = ap.parse_args()
    run_all(a.model, a.datasets, a.skip_predict, a.n_trials)
