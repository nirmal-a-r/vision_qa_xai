"""
train_baselines.py
==================
Per-dataset detectors, one shared risk-control framework.

Per-dataset rather than one merged model: the five sources span unrelated
domains with disjoint label spaces (28 classes, no shared taxonomy), so a single
merged detector would confound "does the method work" with "can one backbone
serve five industries". Training separately also makes the central claim
stronger, because the risk-control layer then has to hold across five
independently-trained models rather than one.

Input resolution is set per dataset from the actual defect scale, not a global
default. PCB defects average 2.6% x 3.2% of a 3034x1586 frame; at the 224 px
this project used originally, a defect is under 8 px and simply is not
recoverable. That single choice is worth more mAP than any architectural
tweak here.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse

# Ultralytics renders a live progress bar per batch. Redirected to a log file it
# becomes megabytes of escape codes and buries the epoch metrics. Disabling tqdm
# alone is the right fix - keep verbose=True so the per-epoch metric lines,
# which are what we actually need to follow a long run, still come through.
os.environ.setdefault("TQDM_DISABLE", "1")

# Per-dataset training config, keyed to native resolution and defect scale.
CONFIGS = {
    "neu":           dict(imgsz=256,  batch=32, epochs=150, native="200x200",   note="defects are large relative to frame"),
    "pcb":           dict(imgsz=1024, batch=4,  epochs=150, native="3034x1586", note="tiny defects, needs high res"),
    "gc10":          dict(imgsz=800,  batch=8,  epochs=120, native="2048x1000", note="wide strip images"),
    "magnetic_tile": dict(imgsz=384,  batch=16, epochs=150, native="~373x248",  note="small set, many negatives"),
    "kolektor":      dict(imgsz=640,  batch=8,  epochs=100, native="~640x230",  note="mostly negatives"),
}


def _load_model(model_name):
    """RT-DETR for the main method, YOLO only as a comparison baseline.

    RT-DETR is the primary detector for a reason beyond accuracy: it is
    NMS-free. NMS is a non-differentiable, score-dependent filter that
    reshapes the confidence distribution in a way that varies with box
    density, and conformal calibration is a statement about exactly that
    distribution. An end-to-end set-prediction head emits one score per object
    with no post-hoc suppression, so the calibration target is the model's own
    output rather than an artefact of the filter applied after it.
    """
    from ultralytics import YOLO, RTDETR
    return (RTDETR if "rtdetr" in model_name.lower() else YOLO)(model_name)


# RT-DETR measured at 46.4 s/epoch vs YOLOv8s at 6.6 s/epoch on NEU@256 on a
# clean GPU - a 7x gap, so the schedule has to be budgeted, not copied across.
# At the YOLO settings RT-DETR would need ~20 h for the five datasets. These
# overrides bring it to ~6.5 h by trimming epochs and easing resolution on the
# three high-resolution sets, while leaving PCB highest (its defects are 2.6%
# of frame and resolution is the binding constraint there, not epochs).
RTDETR_OVERRIDES = {
    "neu":           dict(imgsz=256, batch=8, epochs=100),
    "magnetic_tile": dict(imgsz=384, batch=8, epochs=100),
    "kolektor":      dict(imgsz=512, batch=4, epochs=60),
    "gc10":          dict(imgsz=640, batch=4, epochs=60),
    "pcb":           dict(imgsz=800, batch=2, epochs=80),
}


def train_one(name, data_yaml, model_name, out_dir, seed=0, device=0, override=None,
              run_tag="baseline"):
    cfg = dict(CONFIGS[name])
    if "rtdetr" in model_name.lower():
        cfg.update(RTDETR_OVERRIDES.get(name, {}))
    if override:
        cfg.update(override)

    print(f"\n{'=' * 70}\n[{name}] {model_name} imgsz={cfg['imgsz']} "
          f"batch={cfg['batch']} epochs={cfg['epochs']} seed={seed}\n"
          f"  native {cfg['native']} - {cfg['note']}\n{'=' * 70}", flush=True)

    t0 = time.time()
    model = _load_model(model_name)
    model.train(
        data=os.path.abspath(data_yaml),
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        epochs=cfg["epochs"],
        seed=seed,
        device=device,
        project=out_dir,
        name=f"{name}_{model_name.replace('.pt', '')}_{run_tag}_s{seed}",
        exist_ok=True,
        patience=40,
        workers=0,             # Windows: worker spawn re-imports, keep in-process
        val=True,
        plots=True,
        deterministic=False,
        # Augmentation. CLAHE is applied upstream as an offline pass (see
        # src/data/photometric.py); these are the geometric/photometric ones
        # ultralytics applies online.
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.4, shear=2.0,
        fliplr=0.5, flipud=0.2,
        mosaic=1.0, close_mosaic=15,
        verbose=True,
    )
    dt = time.time() - t0

    # A killed run still returns from model.train() and still evaluates, so it
    # lands in results_*.json looking like a finished experiment - and the
    # resume check then treats it as done and never retrains it. Two runs
    # reached the results file this way (gc10+cce at 5/120 epochs -> 0.3765,
    # neu rtdetr+cce at 2/100 -> 0.0614) and dragged the CCE ablation mean down
    # by 18pp. Verify the trainer actually walked the schedule before recording.
    completed = None
    try:
        import csv as _csv
        _rf = os.path.join(str(getattr(model.trainer, "save_dir", "")), "results.csv")
        if os.path.exists(_rf):
            completed = sum(1 for _ in _csv.DictReader(open(_rf)))
    except Exception:
        completed = None
    if completed is not None:
        # `patience` can stop early legitimately, so allow that; anything that
        # covered less than a third of the schedule was interrupted.
        if completed < max(3, cfg["epochs"] // 3):
            raise RuntimeError(
                f"[{name}] run did not complete: {completed}/{cfg['epochs']} epochs "
                f"in {dt/60:.1f} min. Refusing to record a partial result - it would "
                f"be indistinguishable from a converged one in results_*.json."
            )

    # Evaluate on the held-out TEST block (ultralytics 'val' points at our
    # calibration block, which must not be used to report accuracy).
    metrics = model.val(data=os.path.abspath(data_yaml), split="test",
                        imgsz=cfg["imgsz"], device=device, workers=0, verbose=False)
    res = {
        "dataset": name, "model": model_name, "seed": seed,
        "imgsz": cfg["imgsz"], "epochs": cfg["epochs"],
        "train_seconds": round(dt, 1),
        "test_mAP50": float(metrics.box.map50),
        "test_mAP50_95": float(metrics.box.map),
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        # Ask the trainer where it saved rather than reconstructing the path:
        # ultralytics prepends its own configured runs_dir to `project`, so
        # project="runs/detect" actually lands in runs/detect/runs/detect/...
        "weights": str(getattr(model.trainer, "best", "")) or os.path.join(
            str(getattr(model.trainer, "save_dir", out_dir)), "weights", "best.pt"),
    }
    print(f"[{name}] DONE in {dt/60:.1f} min | "
          f"test mAP@0.5={res['test_mAP50']:.4f} mAP@[.5:.95]={res['test_mAP50_95']:.4f}",
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_dir", default="data/yolo")
    ap.add_argument("--out_dir", default="runs/detect")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--datasets", default="neu,magnetic_tile,kolektor,gc10,pcb")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--results", default="runs/baseline_results.json")
    # Encoding is part of the experiment identity, not a path detail: the same
    # model+dataset+seed trained on grey-x3 and on CCE are two different runs
    # and must not collide in the resume check.
    ap.add_argument("--encoding", default="baseline", help="baseline | cce")
    a = ap.parse_args()

    all_res = []
    if os.path.exists(a.results):
        with open(a.results) as f:
            all_res = json.load(f)

    for seed in [int(s) for s in a.seeds.split(",")]:
        for name in a.datasets.split(","):
            name = name.strip()
            data_yaml = os.path.join(a.yolo_dir, name, "data.yaml")
            if not os.path.exists(data_yaml):
                print(f"[{name}] skipped: no {data_yaml}")
                continue
            if any(r["dataset"] == name and r["seed"] == seed and r["model"] == a.model
                   and r.get("encoding", "baseline") == a.encoding
                   for r in all_res):
                print(f"[{name}] seed {seed} already done, skipping")
                continue
            try:
                r = train_one(name, data_yaml, a.model, a.out_dir, seed=seed,
                              run_tag=a.encoding)
                r["encoding"] = a.encoding
                all_res.append(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            os.makedirs(os.path.dirname(a.results), exist_ok=True)
            with open(a.results, "w") as f:
                json.dump(all_res, f, indent=2)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'dataset':16s} {'model':12s} {'seed':>4s} {'mAP@0.5':>8s} {'mAP@.5:.95':>11s} {'min':>6s}")
    for r in all_res:
        print(f"{r['dataset']:16s} {r['model']:12s} {r['seed']:4d} "
              f"{r['test_mAP50']:8.4f} {r['test_mAP50_95']:11.4f} {r['train_seconds']/60:6.1f}")


if __name__ == "__main__":
    main()
