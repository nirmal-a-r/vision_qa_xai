"""
scripts/run_overnight.py
========================
Full overnight pipeline runner:
1. Trains multi-seed baseline & CCE models across target datasets (neu, magnetic_tile, kolektor)
2. Runs downstream risk calibration, triage, and drift analyses (run_all_experiments.py)
3. Computes multi-seed statistical summaries
4. Builds the certified inspection notebook
"""

import os
import sys
import time
import subprocess
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_EXE = sys.executable
DATASETS = "neu,magnetic_tile,kolektor"
RESULTS_FILE = "runs/results_yolov8s.json"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] ========================================================")
    print(f"[{now}] {msg}")
    print(f"[{now}] ========================================================\n", flush=True)

def run_cmd(args):
    t0 = time.time()
    log(f"RUNNING: {' '.join(args)}")
    ret = subprocess.call(args, cwd=ROOT_DIR)
    dt = time.time() - t0
    if ret != 0:
        log(f"WARNING: Command exited with code {ret} in {dt/60:.2f} min")
    else:
        log(f"COMPLETED successfully in {dt/60:.2f} min")
    return ret

def train_seed(seed):
    log(f"STARTING SEED {seed} : Baseline Encoding")
    run_cmd([PYTHON_EXE, "-u", "-m", "src.evaluation.train_baselines",
             "--model", "yolov8s.pt",
             "--datasets", DATASETS,
             "--seeds", str(seed),
             "--results", RESULTS_FILE])

    log(f"STARTING SEED {seed} : CCE Encoding")
    run_cmd([PYTHON_EXE, "-u", "-m", "src.evaluation.train_baselines",
             "--model", "yolov8s.pt",
             "--encoding", "cce",
             "--yolo_dir", "data/yolo_cce",
             "--datasets", DATASETS,
             "--seeds", str(seed),
             "--results", RESULTS_FILE])

def run_analysis_and_build():
    log("STEP: Running downstream analyses (risk calibration, drift, triage)")
    run_cmd([PYTHON_EXE, "-u", "scripts/run_all_experiments.py"])

    log("STEP: Computing Seed Summary & Paired Ablation")
    run_cmd([PYTHON_EXE, "-u", "-m", "src.evaluation.seed_analysis"])

    log("STEP: Building Certified Notebook")
    run_cmd([PYTHON_EXE, "-u", "scripts/build_certified_notebook.py"])

def main():
    total_t0 = time.time()
    log("STARTING FULL OVERNIGHT PROJECT RUN")
    log(f"Using Python: {PYTHON_EXE}")
    log(f"Datasets: {DATASETS}")

    # Phase 1: Seed 0 -> Immediate full deliverable
    log(">>> PHASE 1: Complete Seed 0 + Initial Deliverables")
    train_seed(0)
    run_analysis_and_build()

    # Phase 2: Seeds 1 and 2
    log(">>> PHASE 2: Seed 1 Training")
    train_seed(1)

    log(">>> PHASE 3: Seed 2 Training")
    train_seed(2)

    # Phase 4: Final Analysis & Notebook
    log(">>> PHASE 4: Final Analysis & Full Multi-Seed Aggregation")
    run_analysis_and_build()

    total_dt = time.time() - total_t0
    log(f"ALL OVERNIGHT PHASES COMPLETE! Total elapsed time: {total_dt/3600:.2f} hours")

if __name__ == "__main__":
    main()
