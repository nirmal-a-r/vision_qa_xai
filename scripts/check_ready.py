"""
check_ready.py - is this machine ready for the final run? Run it before
`scripts/run_pipeline.py`; it changes nothing, it only checks.

    python scripts/check_ready.py
    python scripts/check_ready.py --data_root "D:/datasets/raw"     # data elsewhere

Checks: Python packages, PyTorch + CUDA + RTX 50xx (sm_120) support, free disk,
pretrained weights, and every dataset folder with its expected file counts.
Exit code 0 = ready, 1 = something must be fixed (each problem is printed).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (folder under data/raw, what to count, minimum expected, how to count)
DATASETS = {
    "kolektor":      ("KOLEKTORSDD2", 3335, "png images (masks excluded)"),
    "magnetic_tile": ("MAGNETIC-TILE", 1344, "jpg images"),
    "neu":           ("NEU-DET", 1800, "jpg images"),
    "gc10":          ("GC10-DET", 2294, "xml annotations"),
    "pcb":           ("PCB-DEFECTS/PCB_DATASET", 693, "xml annotations"),
    "mvtec":         ("MVTEC-AD", 5354, "png images (masks excluded)"),   # Stage 2 bridge (optional)
    "dagm":          ("DAGM2007/DAGM_KaggleUpload", 16100, "png images (labels excluded)"),  # optional extra
}
PRIMARY = ("kolektor", "magnetic_tile")
REQUIRED = ("kolektor", "magnetic_tile", "neu", "gc10", "pcb")      # the plan's five (Section 9)


def _count(path, kind):
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            fl = f.lower()
            if kind.startswith("png images"):
                if fl.endswith(".png") and not fl.endswith(("_gt.png", "_mask.png")) and "_label" not in fl:
                    n += 1
            elif kind.startswith("jpg images"):
                n += fl.endswith((".jpg", ".jpeg"))
            elif kind.startswith("xml"):
                n += fl.endswith(".xml")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data", "raw"))
    a = ap.parse_args()
    problems, warnings = [], []
    ok = lambda m: print(f"  [ok]   {m}")            # noqa: E731
    bad = lambda m: (print(f"  [FIX]  {m}"), problems.append(m))    # noqa: E731
    warn = lambda m: (print(f"  [warn] {m}"), warnings.append(m))   # noqa: E731

    print("\n1. Python and packages")
    print(f"  python {sys.version.split()[0]} at {sys.executable}")
    if "vqaenv" not in sys.executable.replace("\\", "/"):
        warn("not running from vqaenv - use vqaenv\\Scripts\\python.exe (README, Section 4)")
    for mod in ("numpy", "scipy", "pandas", "matplotlib", "cv2", "yaml", "ultralytics",
                "torchvision", "pycocotools", "nbformat", "nbclient", "ipykernel", "pytest",
                "diffusers", "transformers", "accelerate"):
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}")
        except Exception:
            optional = mod in ("pytest", "diffusers", "transformers", "accelerate", "pycocotools")
            (warn if optional else bad)(
                f"{mod} missing - pip install -r requirements.txt"
                + (" (only the training-free diffusion generator needs it)" if mod in ("diffusers",
                   "transformers", "accelerate") else ""))

    print("\n2. PyTorch and GPU")
    try:
        import torch
        ok(f"torch {torch.__version__} (CUDA build {torch.version.cuda})")
        if not torch.cuda.is_available():
            bad("CUDA not available - install the cu128 PyTorch build (see README) and a current NVIDIA driver")
        else:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            arch = torch.cuda.get_arch_list()
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30
            ok(f"GPU {name}, compute capability {cap[0]}.{cap[1]}, {mem:.1f} GB")
            if cap[0] >= 12 and not any(x.startswith("sm_12") for x in arch):
                bad(f"this PyTorch build lacks sm_120 kernels for {name}: {arch}")
            else:
                ok(f"architectures in this build: {', '.join(arch)}")
            x = torch.randn(256, 256, device="cuda")
            _ = (x @ x).sum().item()
            ok("small CUDA matmul ran")
    except Exception as e:
        bad(f"torch unusable: {type(e).__name__}: {e}")

    print("\n3. Windows / laptop (Section 10.4)")
    sys.path.insert(0, ROOT)
    from src.utils.winenv import (IS_WINDOWS, gpu_memory_mb, in_cloud_sync_folder,
                                  long_paths_enabled, on_ac_power)
    sync = in_cloud_sync_folder(ROOT)
    (bad if sync else ok)(f"project folder is {'inside ' + sync + ' - move it out (sync locks files mid-run)' if sync else 'not in a cloud-synced folder'}")
    if IS_WINDOWS:
        lp = long_paths_enabled()
        (ok if lp else warn)("Windows long paths " + ("enabled" if lp else
             "DISABLED - enable them (Group Policy / registry LongPathsEnabled = 1) if a path error appears"))
        ac = on_ac_power()
        if ac is not None:
            (ok if ac else warn)("on AC power" if ac else "on battery - plug in for long GPU runs")
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = _MS(); m.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            tot, av = m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
            (ok if av >= 8 else warn)(f"RAM {tot:.0f} GB total, {av:.0f} GB free "
                                      "(training caches images in RAM; config training.cache)")
        except Exception:
            pass
        try:
            import subprocess
            r = subprocess.run(["wsl.exe", "--status"], capture_output=True, timeout=20)
            (ok if r.returncode == 0 else warn)(
                "WSL2 available (for AnomalyDiffusion)" if r.returncode == 0 else
                "WSL2 not set up - only needed for AnomalyDiffusion (wsl --install)")
        except Exception:
            warn("WSL2 not found - only needed for AnomalyDiffusion (wsl --install)")
        print("  [tip]  Windows Defender real-time scanning slows reading many small images;"
              " adding the project folder as an exclusion speeds up training")
    total, free = gpu_memory_mb(0)
    if total:
        (ok if free >= 5500 else warn)(f"GPU memory {free} MB free of {total} MB"
                                       + ("" if free >= 5500 else " - close GPU-heavy apps before the run"))
        try:
            import subprocess
            drv = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                           "--format=csv,noheader"], timeout=20).decode().strip()
            ok(f"NVIDIA driver {drv}")
        except Exception:
            pass
    try:
        from src.evaluation.train_baselines import resolve_profile, training_settings
        st = training_settings()
        ok(f"training profile: {resolve_profile(st.get('vram_profile', 'auto'), 0)} "
           f"(workers={st.get('workers')}, cache={st.get('cache')})")
    except Exception as e:
        warn(f"could not read training settings: {e}")

    print("\n4. Disk and weights")
    free = shutil.disk_usage(ROOT).free / 2**30
    (ok if free >= 30 else warn)(f"{free:.0f} GB free (training + caches need roughly 20-30 GB)")
    for w in ("yolov8s.pt", "rtdetr-l.pt"):
        p = os.path.join(ROOT, w)
        (ok if os.path.exists(p) else warn)(f"{w} {'present' if os.path.exists(p) else 'absent - Ultralytics will download it'}")

    print(f"\n5. Datasets in {a.data_root}")
    for key, (sub, need, kind) in DATASETS.items():
        p = os.path.join(a.data_root, sub)
        if not os.path.isdir(p):
            (bad if key in REQUIRED else warn)(f"{key}: folder {sub} not found"
                                               + ("" if key in REQUIRED else " (optional)"))
            continue
        if key == "dagm":       # only the top-level Class1..Class10 folders are read
            classes = [d for d in os.listdir(p) if d.lower().startswith("class")]
            n = sum(_count(os.path.join(p, d), kind) for d in classes)
            extra = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d)) and d not in classes]
            if extra:
                warn(f"dagm: extra folder(s) {extra} are ignored (a nested duplicate copy can be deleted)")
        else:
            n = _count(p, kind)
        msg = f"{key}: {n} {kind} (expected >= {need})"
        (ok if n >= need else (bad if key in REQUIRED else warn))(msg)

    print("\n6. Stale artefacts")
    import json
    for f in ("runs/results_yolov8s.json", "runs/results_rtdetr.json"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            warn(f"{f} exists with {len(json.load(open(p)))} runs; train_baselines skips any "
                 "(dataset, model, seed) listed there - delete it if those runs predate the "
                 "2026-09-23 split fix")
    y = os.path.join(ROOT, "data", "yolo", "kolektor", "data.yaml")
    if os.path.exists(y) and "images/cal" in open(y).read().split("val:")[-1].splitlines()[0]:
        bad("data/yolo was built with the old splits (val = calibration); delete data/yolo and data/processed")

    # splits built with the allocation the config asks for? (Section 9.1)
    try:
        import yaml
        sys.path.insert(0, ROOT)
        from src.data.splits_and_yolo import split_settings, defect_fractions_for
        clean_f, defect_f, mode, _, _ = split_settings()
        for key in REQUIRED:
            sp = os.path.join(ROOT, "data", "processed", "splits", f"{key}_cal.json")
            if not os.path.exists(sp):
                continue
            n = {}
            for split in ("train", "val", "cal", "test"):
                with open(os.path.join(ROOT, "data", "processed", "splits", f"{key}_{split}.json")) as f:
                    n[split] = sum(1 for im in json.load(f)["images"] if im.get("n_boxes", 1) > 0)
            tot = sum(n.values())
            want = defect_fractions_for(key, clean_f, defect_f, mode)[1]
            got = n["cal"] / max(tot, 1)
            (ok if abs(got - want) < 0.03 else bad)(
                f"{key}: {got:.0%} of defective images in cal (config wants {want:.0%})"
                + ("" if abs(got - want) < 0.03 else " - delete data/processed/splits and "
                   f"data/yolo/{key} so the pipeline rebuilds them"))
    except Exception as e:
        warn(f"could not check split allocation: {type(e).__name__}: {e}")

    print("\n" + ("READY: no blocking problems." if not problems else
                  f"NOT READY: {len(problems)} problem(s) to fix above.")
          + (f" {len(warnings)} warning(s)." if warnings else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
