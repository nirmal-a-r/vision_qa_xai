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

Windows / RTX 5060 (8 GB) settings (project document, Sections 10.3-10.4), from
configs/config.yaml ``training``:

* ``vram_profile``: ``8gb`` uses the document's reduced batch profile - YOLOv8s
  batch 8 at imgsz 256-384, 4 at 512-640, 2 at 800 and above; RT-DETR-L half of
  that. ``full`` uses the larger per-dataset batches below. ``auto`` (default)
  picks ``8gb`` when the GPU has less than 12 GB. Ultralytics accumulates
  gradients to a nominal batch of 64 either way, so the optimisation schedule
  is the same; only memory and speed change. The batch actually used is recorded.
* out-of-memory: the run is retried at half the batch (down to 1) instead of
  failing, and the batch that succeeded is recorded.
* ``workers: 0`` - Windows spawns DataLoader workers by re-importing the script;
  in-process loading is the safe default. ``cache: ram`` decodes every training
  image once into RAM (a few GB here; Ultralytics checks free memory first),
  which removes most of the cost of workers = 0.
* ``project`` is an absolute path (Ultralytics otherwise nests runs/detect/runs/
  detect/..., which eats into Windows' 260-character path limit), and weights are
  recorded relative to the working folder so the project can be moved.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

# Ultralytics renders a live progress bar per batch. Redirected to a log file it
# becomes megabytes of escape codes and buries the epoch metrics. Disabling tqdm
# alone is the right fix - keep verbose=True so the per-epoch metric lines,
# which are what we actually need to follow a long run, still come through.
os.environ.setdefault("TQDM_DISABLE", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Per-dataset training config, keyed to native resolution and defect scale.
# `batch` here is the FULL profile (24 GB-class GPUs); see batch_for().
CONFIGS = {
    "neu":           dict(imgsz=256,  batch=32, epochs=150, native="200x200",   note="defects are large relative to frame"),
    "pcb":           dict(imgsz=1024, batch=4,  epochs=150, native="3034x1586", note="tiny defects, needs high res"),
    "gc10":          dict(imgsz=800,  batch=8,  epochs=120, native="2048x1000", note="wide strip images"),
    "magnetic_tile": dict(imgsz=384,  batch=16, epochs=150, native="~373x248",  note="small set, many negatives"),
    "kolektor":      dict(imgsz=640,  batch=8,  epochs=100, native="~640x230",  note="mostly negatives"),
    "dagm":          dict(imgsz=512,  batch=16, epochs=100, native="512x512",   note="synthetic textures, mostly negatives"),
    "mvtec":         dict(imgsz=512,  batch=8,  epochs=100, native="700-1024 sq", note="bridge dataset, 15 categories"),
}

# RT-DETR measured at 46.4 s/epoch vs YOLOv8s at 6.6 s/epoch on NEU@256 on a
# clean GPU - a 7x gap, so the schedule has to be budgeted, not copied across.
# These overrides trim epochs and ease resolution on the three high-resolution
# sets, while leaving PCB highest (its defects are 2.6% of frame and resolution
# is the binding constraint there, not epochs).
RTDETR_OVERRIDES = {
    "neu":           dict(imgsz=256, batch=8, epochs=100),
    "magnetic_tile": dict(imgsz=384, batch=8, epochs=100),
    "kolektor":      dict(imgsz=512, batch=4, epochs=60),
    "gc10":          dict(imgsz=640, batch=4, epochs=60),
    "pcb":           dict(imgsz=800, batch=2, epochs=80),
    "dagm":          dict(imgsz=512, batch=4, epochs=60),
    "mvtec":         dict(imgsz=512, batch=4, epochs=60),
}


def training_settings():
    """configs/config.yaml ``training`` with defaults."""
    s = {"vram_profile": "auto", "workers": 0, "cache": "ram", "patience": 40}
    p = os.path.join(ROOT, "configs", "config.yaml")
    if os.path.exists(p):
        try:
            import yaml
            with open(p) as f:
                s.update((yaml.safe_load(f) or {}).get("training", {}) or {})
        except Exception:
            pass
    return s


def batch_8gb(model_name: str, imgsz: int) -> int:
    """The document's reduced profile for an 8 GB GPU (Section 10.3)."""
    b = 8 if imgsz <= 384 else 4 if imgsz <= 640 else 2
    if "rtdetr" in model_name.lower():
        b = max(1, b // 2)                   # "start at half the YOLOv8s batch"
    return b


def resolve_profile(profile: str, device) -> str:
    if profile in ("8gb", "full"):
        return profile
    if str(device) == "cpu":
        return "8gb"
    try:
        sys.path.insert(0, ROOT)
        from src.utils.winenv import gpu_memory_mb
        total, _ = gpu_memory_mb(int(device) if str(device).isdigit() else 0)
    except Exception:
        total = None
    return "full" if (total is not None and total >= 12 * 1024) else "8gb"


def batch_for(model_name: str, cfg: dict, profile: str) -> int:
    return int(cfg["batch"]) if profile == "full" else batch_8gb(model_name, int(cfg["imgsz"]))


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
    # Use the pretrained weights shipped at the repository root when the working
    # directory is elsewhere (e.g. a smoke-test workspace), instead of downloading.
    path = model_name
    if not os.path.exists(path) and os.path.exists(os.path.join(ROOT, model_name)):
        path = os.path.join(ROOT, model_name)
    return (RTDETR if "rtdetr" in model_name.lower() else YOLO)(path)


def _free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_oom(e: BaseException) -> bool:
    s = f"{type(e).__name__}: {e}".lower()
    return "out of memory" in s or "outofmemory" in s or "cudnn_status_alloc_failed" in s


def _portable(path: str) -> str:
    """Path relative to the working folder when it lies inside it (movable project)."""
    try:
        rel = os.path.relpath(path)
        return path if rel.startswith("..") else rel
    except ValueError:                        # another drive on Windows
        return path


def train_one(name, data_yaml, model_name, out_dir, seed=0, device=0, override=None,
              run_tag="baseline", settings=None):
    settings = settings or training_settings()
    cfg = dict(CONFIGS[name])
    if "rtdetr" in model_name.lower():
        cfg.update(RTDETR_OVERRIDES.get(name, {}))
    profile = resolve_profile(settings.get("vram_profile", "auto"), device)
    cfg["batch"] = batch_for(model_name, cfg, profile)
    if override:
        cfg.update(override)
    workers = int(settings.get("workers", 0))
    cache = settings.get("cache", "ram")
    cache = False if cache in (None, False, "none", "false") else cache
    run_name = f"{name}_{model_name.replace('.pt', '')}_{run_tag}_s{seed}"
    project = os.path.abspath(out_dir)

    t0 = time.time()
    batch = int(cfg["batch"])
    while True:
        print(f"\n{'=' * 70}\n[{name}] {model_name} imgsz={cfg['imgsz']} batch={batch} "
              f"({profile} profile) epochs={cfg['epochs']} seed={seed} workers={workers} "
              f"cache={cache}\n  native {cfg['native']} - {cfg['note']}\n{'=' * 70}", flush=True)
        model = _load_model(model_name)
        try:
            model.train(
                data=os.path.abspath(data_yaml),
                imgsz=cfg["imgsz"],
                batch=batch,
                epochs=cfg["epochs"],
                seed=seed,
                device=device,
                project=project,
                name=run_name,
                exist_ok=True,
                patience=int(settings.get("patience", 40)),
                workers=workers,       # Windows: worker spawn re-imports, keep in-process
                cache=cache,
                amp=True,
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
            break
        except Exception as e:
            if _is_oom(e) and batch > 1:
                print(f"[{name}] out of GPU memory at batch {batch}; retrying at {batch // 2}", flush=True)
                del model
                _free_gpu()
                batch //= 2
                continue
            raise
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
            with open(_rf) as f:
                completed = sum(1 for _ in _csv.DictReader(f))
    except Exception:
        completed = None
    if completed is not None:
        # `patience` can stop early legitimately, so allow that; anything that
        # covered less than a third of the schedule was interrupted.
        if completed < min(cfg["epochs"], max(3, cfg["epochs"] // 3)):
            raise RuntimeError(
                f"[{name}] run did not complete: {completed}/{cfg['epochs']} epochs "
                f"in {dt/60:.1f} min. Refusing to record a partial result - it would "
                f"be indistinguishable from a converged one in results_*.json."
            )

    # Evaluate on the held-out TEST block. Ultralytics 'val' is the inner
    # validation slice used for checkpoint selection, so it must not be used to
    # report accuracy, and the calibration block is never shown to training.
    on_gpu = str(device) != "cpu"
    metrics = model.val(data=os.path.abspath(data_yaml), split="test", imgsz=cfg["imgsz"],
                        batch=max(1, 2 * batch), half=on_gpu, device=device, workers=0,
                        verbose=False)
    weights = str(getattr(model.trainer, "best", "")) or os.path.join(
        str(getattr(model.trainer, "save_dir", out_dir)), "weights", "best.pt")
    res = {
        "dataset": name, "model": model_name, "seed": seed,
        "imgsz": cfg["imgsz"], "epochs": cfg["epochs"], "epochs_completed": completed,
        "batch": batch, "vram_profile": profile, "cache": str(cache), "workers": workers,
        "train_seconds": round(dt, 1),
        "test_mAP50": float(metrics.box.map50),
        "test_mAP50_95": float(metrics.box.map),
        "test_precision": float(metrics.box.mp),
        "test_recall": float(metrics.box.mr),
        # Ask the trainer where it saved rather than reconstructing the path.
        "weights": _portable(weights),
    }
    print(f"[{name}] DONE in {dt/60:.1f} min | "
          f"test mAP@0.5={res['test_mAP50']:.4f} mAP@[.5:.95]={res['test_mAP50_95']:.4f}",
          flush=True)
    del model
    _free_gpu()
    return res


def main():
    sys.path.insert(0, ROOT)
    try:
        from src.utils.winenv import setup_console
        setup_console()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_dir", default="data/yolo")
    ap.add_argument("--out_dir", default="runs/detect")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--datasets", default="kolektor,magnetic_tile,neu,gc10,pcb")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--results", default="runs/baseline_results.json")
    # Encoding is part of the experiment identity, not a path detail: the same
    # model+dataset+seed trained on grey-x3 and on CCE are two different runs
    # and must not collide in the resume check.
    ap.add_argument("--encoding", default="baseline", help="baseline | cce | p2_<source>")
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    ap.add_argument("--vram_profile", default=None, help="auto | 8gb | full (default: config)")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: config, 0)")
    ap.add_argument("--cache", default=None, help="ram | disk | none (default: config)")
    # Optional overrides of the per-dataset schedule, for quick smoke tests only.
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    a = ap.parse_args()
    device = int(a.device) if a.device.isdigit() else a.device
    override = {k: v for k, v in (("epochs", a.epochs), ("imgsz", a.imgsz), ("batch", a.batch))
                if v is not None} or None
    settings = training_settings()
    for k in ("vram_profile", "workers", "cache"):
        if getattr(a, k) is not None:
            settings[k] = getattr(a, k)

    def load_results():
        if os.path.exists(a.results):
            with open(a.results) as f:
                return json.load(f)
        return []

    for seed in [int(s) for s in a.seeds.split(",")]:
        for name in a.datasets.split(","):
            name = name.strip()
            data_yaml = os.path.join(a.yolo_dir, name, "data.yaml")
            if not os.path.exists(data_yaml):
                print(f"[{name}] skipped: no {data_yaml}")
                continue
            if any(r["dataset"] == name and r["seed"] == seed and r["model"] == a.model
                   and r.get("encoding", "baseline") == a.encoding
                   for r in load_results()):
                print(f"[{name}] {a.model} {a.encoding} seed {seed} already done, skipping")
                continue
            try:
                r = train_one(name, data_yaml, a.model, a.out_dir, seed=seed, device=device,
                              override=override, run_tag=a.encoding, settings=settings)
                r["encoding"] = a.encoding
                all_res = load_results()          # re-read: never overwrite another run's row
                all_res.append(r)
                os.makedirs(os.path.dirname(a.results) or ".", exist_ok=True)
                tmp = a.results + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(all_res, f, indent=2)
                os.replace(tmp, a.results)        # atomic on Windows and POSIX
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
                _free_gpu()

    all_res = load_results()
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'dataset':16s} {'model':12s} {'enc':22s} {'seed':>4s} {'batch':>5s} {'mAP@0.5':>8s} {'mAP@.5:.95':>11s} {'min':>6s}")
    for r in all_res:
        print(f"{r['dataset']:16s} {r['model']:12s} {str(r.get('encoding', 'baseline')):22s} {r['seed']:4d} "
              f"{str(r.get('batch', '-')):>5s} {r['test_mAP50']:8.4f} {r['test_mAP50_95']:11.4f} "
              f"{r['train_seconds']/60:6.1f}")


if __name__ == "__main__":
    main()
