"""
run_pipeline.py
===============
The single entry point that takes the repo from a clean clone to every artefact
the notebook renders.

    python scripts/run_pipeline.py                # everything, resumable
    python scripts/run_pipeline.py --skip-train   # analysis + notebook only
    python scripts/run_pipeline.py --wait-for-gpu # block until the GPU is free

Stages run in dependency order and each is individually resumable, because the
expensive ones are hours long and a machine that reboots mid-run should not cost
the whole sweep:

    1  prepare      raw datasets -> COCO -> stratified train/cal/test -> YOLO trees
    2  encode       Complementary Channel Encoding copies of those trees
    3  train        detectors, per dataset, baseline and CCE arms
    4  analyse      cache predictions once, then risk / drift / triage / agent
    5  notebook     execute the notebook end to end and save the run copy

`--wait-for-gpu` exists because this machine shares one 8 GB card with another
project. Two trainings on one card OOM each other, and the per-dataset exception
handler would swallow the failure and silently skip a dataset - so the pipeline
waits for a genuinely free card instead of racing for it.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = os.path.join(ROOT, "vqaenv", "Scripts", "python.exe")
if not os.path.exists(PY):
    PY = sys.executable


def run(cmd, label):
    print(f"\n{'=' * 72}\n>>> {label}\n{'=' * 72}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, shell=isinstance(cmd, str))
    print(f"<<< {label}: rc={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rc


# ---------------------------------------------------------------------------
# GPU gating
# ---------------------------------------------------------------------------

def gpu_busy_mb(threshold_mb=1500):
    """Used VRAM in MB, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def wait_for_gpu(threshold_mb=1500, poll_s=120, max_wait_h=24):
    """Block until VRAM use drops below threshold.

    Threshold rather than zero: the desktop compositor and any idle CUDA context
    hold a few hundred MB permanently, so waiting for exactly 0 would wait
    forever.
    """
    deadline = time.time() + max_wait_h * 3600
    while time.time() < deadline:
        used = gpu_busy_mb()
        if used is None:
            print("  nvidia-smi unavailable; proceeding without GPU gating", flush=True)
            return True
        if used < threshold_mb:
            print(f"  GPU free ({used} MB used) - starting", flush=True)
            return True
        print(f"  GPU busy ({used} MB used), waiting {poll_s}s ...", flush=True)
        time.sleep(poll_s)
    print("  gave up waiting for the GPU", flush=True)
    return False


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_prepare():
    if os.path.exists("data/yolo/neu/data.yaml"):
        print("  datasets already prepared, skipping")
        return 0
    rc = run([PY, "-m", "src.data.prepare_datasets"], "1a  raw -> COCO")
    if rc:
        return rc
    return run([PY, "-m", "src.data.splits_and_yolo"], "1b  splits -> YOLO trees")


def stage_encode():
    if os.path.exists("data/yolo_cce/neu/data.yaml"):
        print("  CCE trees already present, skipping")
        return 0
    return run([PY, "-m", "src.data.build_cce_dataset"], "2  Complementary Channel Encoding")


def stage_train(datasets):
    ds = ",".join(datasets)
    rc = 0
    rc |= run([PY, "-m", "src.evaluation.train_baselines", "--model", "yolov8s.pt",
               "--datasets", ds, "--results", "runs/results_yolov8s.json"],
              "3a  YOLOv8s baseline")
    rc |= run([PY, "-m", "src.evaluation.train_baselines", "--model", "yolov8s.pt",
               "--encoding", "cce", "--yolo_dir", "data/yolo_cce",
               "--datasets", ds, "--results", "runs/results_yolov8s.json"],
              "3b  YOLOv8s + CCE")
    rc |= run([PY, "-m", "src.evaluation.train_baselines", "--model", "rtdetr-l.pt",
               "--datasets", ds, "--results", "runs/results_rtdetr.json"],
              "3c  RT-DETR baseline")
    rc |= run([PY, "-m", "src.evaluation.train_baselines", "--model", "rtdetr-l.pt",
               "--encoding", "cce", "--yolo_dir", "data/yolo_cce",
               "--datasets", ds, "--results", "runs/results_rtdetr.json"],
              "3d  RT-DETR + CCE")
    return rc


def stage_analyse():
    return run([PY, "scripts/run_all_experiments.py"], "4  risk / drift / triage / agent")


def stage_notebook(timeout_s=7200):
    nb = "notebook/VisionQA_RiskControlled_Inspection.ipynb"
    out = "notebook/VisionQA_RiskControlled_Inspection_executed.ipynb"
    code = (
        "import nbformat, io\n"
        "from nbclient import NotebookClient\n"
        f"nb = nbformat.read(io.open(r'{nb}', encoding='utf-8'), as_version=4)\n"
        f"c = NotebookClient(nb, timeout={timeout_s}, kernel_name='python3',\n"
        f"                   resources={{'metadata': {{'path': r'{ROOT}'}}}}, allow_errors=True)\n"
        "c.execute()\n"
        f"nbformat.write(nb, io.open(r'{out}', 'w', encoding='utf-8'))\n"
        "errs = [(i, o) for i, cell in enumerate(nb.cells) if cell.cell_type=='code'\n"
        "        for o in cell.get('outputs', []) if o.output_type=='error']\n"
        "print(f'executed; {len(errs)} cell error(s)')\n"
        "for i, o in errs[:5]:\n"
        "    print(' cell', i, o.get('ename'), str(o.get('evalue'))[:160])\n"
    )
    return run([PY, "-c", code], "5  execute notebook")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="neu,magnetic_tile,kolektor,gc10,pcb")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-notebook", action="store_true")
    ap.add_argument("--wait-for-gpu", action="store_true")
    ap.add_argument("--gpu-threshold-mb", type=int, default=1500)
    a = ap.parse_args()

    t0 = time.time()
    if a.wait_for_gpu and not a.skip_train:
        print("waiting for a free GPU before training ...", flush=True)
        wait_for_gpu(a.gpu_threshold_mb)

    datasets = [d.strip() for d in a.datasets.split(",")]
    if stage_prepare():
        return 1
    if stage_encode():
        return 1
    if not a.skip_train:
        stage_train(datasets)          # per-dataset failures are tolerated inside
    if stage_analyse():
        return 1
    if not a.skip_notebook:
        stage_notebook()

    print(f"\nPIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min")
    for f in ("runs/results_yolov8s.json", "runs/results_rtdetr.json",
              "runs/risk_results.json", "runs/triage_results.json"):
        if os.path.exists(f):
            n = len(json.load(open(f)))
            print(f"  {f}: {n} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
