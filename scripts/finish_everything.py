"""
finish_everything.py
====================
Unattended completion: wait for the GPU, finish the remaining runs, execute the
notebook, and leave the repo in a committable state.

Waits on *free VRAM* rather than utilisation. Utilisation swings to 0% between
batches of a perfectly healthy job, so polling it would start a second training
on top of a running one - which is exactly the collision that happened earlier
and would have OOM'd at the high-resolution datasets.
"""
import os, sys, time, json, subprocess, glob, shutil
os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PY = os.path.join(ROOT, "vqaenv", "Scripts", "python.exe")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def free_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
        used, total = [int(x) for x in out.split(",")]
        return total - used
    except Exception:
        return 0


def wait_for_gpu(need_mb=6000, stable_s=90, poll=30, timeout_h=24):
    log(f"waiting for >={need_mb} MB free VRAM, stable for {stable_s}s")
    t0, ok_since = time.time(), None
    while time.time() - t0 < timeout_h * 3600:
        f = free_vram_mb()
        if f >= need_mb:
            ok_since = ok_since or time.time()
            if time.time() - ok_since >= stable_s:
                log(f"GPU free ({f} MB) - starting")
                return True
        else:
            if ok_since:
                log(f"GPU busy again ({f} MB free) - resetting timer")
            ok_since = None
        time.sleep(poll)
    log("timed out waiting for GPU")
    return False


def run(cmd, label):
    log(f"START {label}")
    r = subprocess.run(cmd, cwd=ROOT)
    log(f"{'OK   ' if r.returncode == 0 else 'FAIL '} {label} (rc={r.returncode})")
    return r.returncode == 0


def main():
    if not wait_for_gpu():
        return 1

    # 1. remaining detector runs
    run([PY, "-u", "-m", "src.evaluation.train_baselines", "--model", "yolov8s.pt",
         "--encoding", "cce", "--yolo_dir", "data/yolo_cce",
         "--datasets", "gc10,pcb", "--results", "runs/results_yolov8s.json"],
        "yolov8s + CCE (gc10, pcb)")
    run([PY, "-u", "-m", "src.evaluation.train_baselines", "--model", "rtdetr-l.pt",
         "--encoding", "cce", "--yolo_dir", "data/yolo_cce",
         "--datasets", "neu,magnetic_tile", "--results", "runs/results_rtdetr.json"],
        "rtdetr + CCE (neu, magnetic_tile)")

    # 2. downstream analysis
    run([PY, "-u", "scripts/run_all_experiments.py", "--model", "yolov8s.pt"],
        "risk + triage experiments")

    # 3. execute the notebook
    nb = "notebook/VisionQA_RiskControlled_Inspection.ipynb"
    run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=3600",
         "--ExecutePreprocessor.kernel_name=vqaenv", nb], "execute notebook")

    # 4. tidy for commit
    for p in ["yolo26n.pt", "yolov8s.pt", "rtdetr-l.pt"]:
        if os.path.exists(p):
            os.makedirs("weights_cache", exist_ok=True)
            shutil.move(p, os.path.join("weights_cache", p))
    for d in glob.glob("**/__pycache__", recursive=True):
        shutil.rmtree(d, ignore_errors=True)
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
