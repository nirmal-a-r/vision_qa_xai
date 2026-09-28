"""
run_pipeline.py
===============
One command from raw datasets to every result in the project document. Every
stage is resumable (finished work is detected and skipped), so an interrupted run
can simply be started again. Stage numbers in brackets are the document's
experimental stages (Section 12).

    python scripts/run_pipeline.py                      # everything, config defaults
    python scripts/run_pipeline.py --datasets kolektor,magnetic_tile   # primary first
    python scripts/run_pipeline.py --skip-train         # reuse trained detectors
    python scripts/run_pipeline.py --generate copy_paste,training_free   # + synthetic defects
    python scripts/run_pipeline.py --skip-train --p2    # + protocol P2 (full cold start)
    python scripts/run_pipeline.py --wait-for-gpu       # wait until the GPU is free
    python scripts/smoke_test.py                        # tiny end-to-end check first

Stages
    1  prepare   raw datasets -> COCO -> train / val / cal / test -> YOLO trees
    2  encode    CCE copies of the trees (only with --with-cce; ablation)
    3  train     RT-DETR-L and YOLOv8s per dataset and seed, primary datasets first  [Stage 3]
    4  cache     detections on cal and test for every trained run (runs/preds/)
    5  analyse   CRC on both escape events, drift and triage (src/evaluation/experiments.py)
    6  generate  optional (--generate): synthetic defects for the primary datasets
    6b synth     generator output -> labelled sets -> scored by EVERY detector and seed
    7  sperc     commissioning table (Section 8.4), Stage 1 go / no-go [Stage 1],
                 cold-start sweep for every model x seed [Stages 4, 5], summary (RQ1-RQ4)
    7b p2        optional (--p2): protocol P2 [Stage 6]
    7c mvtec     MVTec AD bridge when data/raw/MVTEC-AD exists [Stage 2]
    8  seeds     detector accuracy mean / spread across seeds
    9  notebook  rebuild and execute notebook/VisionQA_CertifiedInspection.ipynb

Why --wait-for-gpu exists: one 8 GB GPU is shared with other work. Two trainings
on one card crash each other, so the pipeline waits for free VRAM (Section 10.4).

Windows behaviour (Section 10.4)
* every (dataset, seed, model) training runs in its OWN process: a crash, an
  out-of-memory error or leaked VRAM in one run cannot take down the others, and
  each run starts from a clean CUDA context;
* the PC is kept awake while the pipeline runs (no power setting is changed);
* logs are UTF-8 (Ultralytics prints emoji, which crash a cp1252 log);
* a warning is printed when a laptop runs on battery.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils.winenv import keep_awake, on_ac_power, setup_console  # noqa: E402
_WIN_PY = os.path.join(ROOT, "vqaenv", "Scripts", "python.exe")      # Windows venv
PY = _WIN_PY if (os.name == "nt" and os.path.exists(_WIN_PY)) else sys.executable

MODEL_RESULTS = {"yolov8s.pt": "runs/results_yolov8s.json",
                 "rtdetr-l.pt": "runs/results_rtdetr.json"}
NOTEBOOK = os.path.join(ROOT, "notebook", "VisionQA_CertifiedInspection.ipynb")
NOTEBOOK_RUN = os.path.join(ROOT, "notebook", "VisionQA_CertifiedInspection_executed.ipynb")

with open(os.path.join(ROOT, "configs", "config.yaml")) as _f:
    CFG = yaml.safe_load(_f)


def _env():
    e = dict(os.environ)
    e["PYTHONPATH"] = ROOT + os.pathsep + e.get("PYTHONPATH", "")
    e.setdefault("TQDM_DISABLE", "1")
    e.setdefault("PYTHONIOENCODING", "utf-8")       # logs redirected to a file on Windows
    e.setdefault("PYTHONUTF8", "1")
    return e


def run(cmd, label):
    print(f"\n{'=' * 72}\n>>> {label}\n{'=' * 72}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, env=_env())
    print(f"<<< {label}: rc={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rc


def script(name):
    return os.path.join(ROOT, "scripts", name)


def stem(model):
    return model.replace(".pt", "")


# ------------------------------------------------------------------ GPU gating
def _nvsmi_free_mb(index=0):
    """Free VRAM from nvidia-smi (does not create a CUDA context in this process)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=20)
        t, f = (int(x) for x in out.decode().strip().splitlines()[0].split(","))
        return t, f
    except Exception:
        return None, None


def wait_for_gpu(need_free_mb=5500, poll_s=120, max_wait_h=24, device="0"):
    """Block until the GPU has ``need_free_mb`` MB free.

    Measured as FREE memory rather than used memory: on Windows the desktop, a
    browser or VS Code can hold 0.5-1.5 GB of the card indefinitely, so a
    "used < X" rule may never be met, while what a run actually needs is free space.
    """
    if str(device) == "cpu":
        return True
    idx = int(device) if str(device).isdigit() else 0
    deadline = time.time() + max_wait_h * 3600
    while time.time() < deadline:
        total, free = _nvsmi_free_mb(idx)
        if free is None:
            print("  nvidia-smi unavailable; proceeding without GPU gating", flush=True)
            return True
        need = min(need_free_mb, int(0.8 * total))
        if free >= need:
            print(f"  GPU ready: {free} MB of {total} MB free", flush=True)
            return True
        print(f"  GPU busy: {free} MB free, need {need} MB - another job is using it; "
              f"waiting {poll_s}s ...", flush=True)
        time.sleep(poll_s)
    print("  gave up waiting for the GPU; trying anyway", flush=True)
    return False


# ------------------------------------------------------------------ helpers
def trained_runs(model, dataset=None, encoding="baseline"):
    p = MODEL_RESULTS[model]
    if not os.path.exists(p):
        return []
    with open(p) as f:
        rows = json.load(f)
    return sorted([r for r in rows if (dataset is None or r.get("dataset") == dataset)
                   and (r.get("encoding") or "baseline") == encoding and r.get("epochs", 0) > 0
                   and os.path.exists(r.get("weights", ""))],
                  key=lambda r: r.get("seed", 0))


def tag_of(model, encoding="baseline", seed=0):
    from src.evaluation.dump_ultralytics_preds import cache_tag
    return cache_tag(model, encoding, seed)


def order_primary_first(datasets):
    prim = [d for d in datasets if d in CFG.get("primary_datasets", [])]
    return prim, [d for d in datasets if d not in prim]


# ------------------------------------------------------------------ stages
def stage_prepare(datasets):
    need = [d for d in datasets if not os.path.exists(f"data/yolo/{d}/data.yaml")]
    if not need:
        print("  datasets already prepared, skipping")
        return 0
    only = ",".join(need)
    rc = run([PY, "-m", "src.data.prepare_datasets", "--only", only], f"1a  raw -> COCO ({only})")
    return rc or run([PY, "-m", "src.data.splits_and_yolo", "--only", only],
                     f"1b  splits -> YOLO trees ({only})")


def stage_encode():
    if os.path.exists("data/yolo_cce/kolektor/data.yaml"):
        print("  CCE trees already present, skipping")
        return 0
    return run([PY, "-m", "src.data.build_cce_dataset"], "2  Complementary Channel Encoding (ablation)")


def _done(model, dataset, seed, encoding="baseline"):
    p = MODEL_RESULTS[model]
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            rows = json.load(f)
    except Exception:
        return False
    return any(r.get("dataset") == dataset and int(r.get("seed", 0)) == int(seed)
               and r.get("model") == model and (r.get("encoding") or "baseline") == encoding
               for r in rows)


def stage_train(datasets, seeds, models, with_cce, device, overrides, wait, gpu_mb):
    """One process per (dataset, seed, model, encoding); primary datasets first
    (Section 12, Stage 3), and within a dataset all seeds before the next dataset."""
    rc = 0
    extra = ["--device", str(device)] + overrides
    arms = [("baseline", "data/yolo")] + ([("cce", "data/yolo_cce")] if with_cce else [])
    todo = [(ds, sd, m, enc, ydir) for group in order_primary_first(datasets) for ds in group
            for sd in [s.strip() for s in str(seeds).split(",") if s.strip()]
            for m in models for enc, ydir in arms]
    left = [t for t in todo if not _done(t[2], t[0], t[1], t[3])]
    print(f"  {len(todo) - len(left)} of {len(todo)} training runs already done; {len(left)} to go")
    for i, (ds, sd, model, enc, ydir) in enumerate(left, 1):
        if wait:
            wait_for_gpu(gpu_mb, device=device)
        rc |= run([PY, "-m", "src.evaluation.train_baselines", "--model", model, "--encoding", enc,
                   "--yolo_dir", ydir, "--datasets", ds, "--seeds", sd,
                   "--results", MODEL_RESULTS[model]] + extra,
                  f"3  [{i}/{len(left)}] train {model} on {ds}, seed {sd}" + ("" if enc == "baseline" else f", {enc}"))
    return rc


def stage_cache(models, device):
    rc = 0
    for model in models:
        if os.path.exists(MODEL_RESULTS[model]):
            rc |= run([PY, "-m", "src.evaluation.dump_ultralytics_preds",
                       "--results", MODEL_RESULTS[model], "--device", str(device)],
                      f"4  cache detections ({model})")
    return rc


def stage_analyse(datasets, models):
    rc = 0
    for model in models:
        rc |= run([PY, "-m", "src.evaluation.experiments", "--model", model,
                   "--datasets", ",".join(datasets), "--skip_predict"],
                  f"5  CRC / drift / triage ({model})")
    return rc


def stage_generate(datasets, generators, n, device):
    rc = 0
    for ds in [d for d in datasets if d in CFG.get("primary_datasets", [])]:
        for g in generators:
            cmd = [PY, script("generate_synthetic.py"), "--dataset", ds, "--generator", g]
            if n:
                cmd += ["--n", str(n)]
            if device is not None and str(device) != "cpu":
                cmd += ["--device", f"cuda:{device}" if str(device).isdigit() else str(device)]
            rc |= run(cmd, f"6  generate {g} defects for {ds}")
    return rc


def _score(weights, imgsz, syn_root, out, device, label):
    if os.path.exists(out):
        return 0
    return run([PY, "-m", "src.evaluation.score_synthetic", "--weights", weights,
                "--imgsz", str(imgsz), "--syn_root", syn_root, "--out", out,
                "--device", str(device)], label)


def stage_synth(datasets, models, mode, device):
    from src.synth.composite import build_synthetic_set
    rc = 0
    gen_root = CFG["paths"].get("generated_dir", "data/generated")
    syn_dir = CFG["paths"].get("synthetic_dir", "data/synthetic")
    for ds in datasets:
        for gen in sorted(glob.glob(os.path.join(gen_root, ds, "*"))):
            if not os.path.isdir(gen):
                continue
            source = os.path.basename(gen)
            syn_root = os.path.join(syn_dir, ds, source)
            if not os.path.exists(os.path.join(syn_root, "manifest.json")):
                clean = os.path.join("data", "yolo", ds, "images", "train")
                r = build_synthetic_set(gen, syn_root, mode=mode,
                                        clean_dir=clean if mode == "paste" else None)
                print(f"  [{ds}/{source}] {r.written} labelled synthetic images -> {syn_root}")
            for model in models:
                for run_ in trained_runs(model, ds):        # every seed's frozen detector
                    tag = tag_of(model, "baseline", run_.get("seed", 0))
                    out = os.path.join("runs", "preds", f"{ds}_{tag}_syn_{source}.json")
                    rc |= _score(run_["weights"], run_["imgsz"], syn_root, out, device,
                                 f"6b score synthetic {ds}/{source} with {tag}")
    return rc


def stage_sperc(datasets, models, seeds, n_draws=None, m_values=None):
    rc = run([PY, script("commissioning_table.py"), "--out", "runs/commissioning_table.csv"],
             "7a  commissioning table (Section 8.4)")
    head = stem(CFG.get("headline_model", "rtdetr-l")) + ".pt"
    head_model = head if head in models else models[0]
    s1 = [d for d in CFG.get("stage1", {}).get("datasets", ["gc10"]) if d in datasets]
    if s1:
        cmd = [PY, script("run_stage1.py"), "--datasets", ",".join(s1), "--model", stem(head_model),
               "--preds_dir", "runs/preds", "--out_dir", "runs/stage1"]
        if n_draws:
            cmd += ["--n_draws", str(n_draws)]
        rc |= run(cmd, "7b  Stage 1: best case on GC10, go / no-go")
    cmd = [PY, script("run_coldstart_sweep.py"), "--datasets", ",".join(datasets),
           "--models", ",".join(stem(m) for m in models), "--seeds", seeds,
           "--preds_dir", "runs/preds", "--out_dir", "runs/coldstart"]
    if n_draws:
        cmd += ["--n_draws", str(n_draws)]
    if m_values:
        cmd += ["--m_values", m_values]
    rc |= run(cmd, "7c  cold-start sweep, every model x seed (Stages 4-5)")
    return rc | run([PY, script("summarize_sperc.py"), "--out_dir", "runs/coldstart"],
                    "7d  pool seeds: m-star, validity, planned vs realised, alignment")


def stage_p2(datasets, seeds, models, device, overrides, k, n_draws=None, m_values=None,
             wait=False, gpu_mb=5500):
    """Protocol P2 on the primary datasets, one detector per synthetic source."""
    from src.data.build_coldstart_tree import build_tree
    rc = 0
    syn_dir = CFG["paths"].get("synthetic_dir", "data/synthetic")
    for ds in [d for d in datasets if d in CFG.get("primary_datasets", [])]:
        for syn_root in sorted(glob.glob(os.path.join(syn_dir, ds, "*"))):
            source = os.path.basename(syn_root)
            if source.endswith("_p2holdout") or not os.path.exists(os.path.join(syn_root, "manifest.json")):
                continue
            enc = f"p2_{source}"
            tree = os.path.join(f"data/yolo_{enc}", ds)
            hold = syn_root.rstrip("/\\") + "_p2holdout"
            if not os.path.exists(os.path.join(tree, "data.yaml")):
                root, c = build_tree(ds, source, k=k, syn_root=syn_root, holdout_root=hold)
                print(f"  [P2 {ds}/{source}] {root}: {c}")
            for model in models:
                for sd in [x.strip() for x in str(seeds).split(",") if x.strip()]:
                    if _done(model, ds, sd, enc):
                        continue
                    if wait:
                        wait_for_gpu(gpu_mb, device=device)
                    rc |= run([PY, "-m", "src.evaluation.train_baselines", "--model", model,
                               "--yolo_dir", f"data/yolo_{enc}", "--encoding", enc, "--datasets", ds,
                               "--seeds", sd, "--results", MODEL_RESULTS[model], "--device", str(device)]
                              + overrides, f"P2  train {model} on {ds}, seed {sd} ({k} real + {source})")
                rc |= run([PY, "-m", "src.evaluation.dump_ultralytics_preds",
                           "--results", MODEL_RESULTS[model], "--device", str(device)],
                          f"P2  cache detections ({model})")
                tags = []
                for run_ in trained_runs(model, ds, enc):
                    tag = tag_of(model, enc, run_.get("seed", 0))
                    tags.append(tag)
                    out = os.path.join("runs", "preds", f"{ds}_{tag}_syn_{source}.json")
                    rc |= _score(run_["weights"], run_["imgsz"], hold, out, device,
                                 f"P2  score held-out synthetic ({tag})")
                for tag in tags:
                    cmd = [PY, script("run_coldstart_sweep.py"), "--datasets", ds, "--model", tag,
                           "--preds_dir", "runs/preds", "--out_dir", "runs/coldstart_p2"]
                    if n_draws:
                        cmd += ["--n_draws", str(n_draws)]
                    if m_values:
                        cmd += ["--m_values", m_values]
                    rc |= run(cmd, f"P2  cold-start sweep ({tag})")
    if glob.glob("runs/coldstart_p2/coldstart_*.json"):
        rc |= run([PY, script("summarize_sperc.py"), "--out_dir", "runs/coldstart_p2"], "P2  summary")
    return rc


def stage_mvtec(n_draws=None, device=None):
    if not os.path.isdir(os.path.join("data", "raw", "MVTEC-AD")):
        print("  data/raw/MVTEC-AD not present - Stage 2 (MVTec AD bridge) skipped")
        return 0
    cmd = [PY, script("run_mvtec_bridge.py"), "--root", os.path.join("data", "raw", "MVTEC-AD"),
           "--out_dir", os.path.join("runs", "mvtec")]
    if n_draws:
        cmd += ["--n_draws", str(n_draws)]
    if device is not None:
        cmd += ["--device", "cpu" if str(device) == "cpu" else f"cuda:{device}" if str(device).isdigit() else str(device)]
    return run(cmd, "7e  Stage 2: MVTec AD bridge")


def stage_seeds():
    return run([PY, "-m", "src.evaluation.seed_analysis"], "8  detector accuracy across seeds")


def stage_notebook(timeout_s=7200):
    rc = run([PY, script("build_certified_notebook.py")], "9a  rebuild notebook")
    if rc:
        return rc
    same = os.path.normcase(os.path.abspath(os.getcwd())) == os.path.normcase(ROOT)
    out = NOTEBOOK_RUN if same else os.path.join(os.getcwd(), os.path.basename(NOTEBOOK_RUN))
    return run([PY, script("execute_notebook.py"), "--work", os.getcwd(), "--out", out,
                "--timeout", str(timeout_s)], "9b  execute notebook")


def main():
    ap = argparse.ArgumentParser(description="Run the full project pipeline.")
    ap.add_argument("--work", default=ROOT, help="folder holding data/ and runs/ (default: repo)")
    ap.add_argument("--datasets", default=",".join(CFG["datasets"]))
    ap.add_argument("--seeds", default=",".join(str(s) for s in CFG.get("seeds", [0])), help="e.g. 0,1,2")
    ap.add_argument("--models", default=",".join(stem(m) + ".pt" for m in CFG.get("models", ["rtdetr-l", "yolov8s"])))
    ap.add_argument("--with-cce", action="store_true", help="also train the CCE ablation arm")
    ap.add_argument("--generate", default="", help="comma list of in-repo generators: copy_paste,training_free")
    ap.add_argument("--synthetic-n", type=int, default=None, help="override generators.n_synthetic")
    ap.add_argument("--synth-mode", default="full", choices=["full", "paste"])
    ap.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--imgsz", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--batch", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--n_draws", type=int, default=None, help="override coldstart.n_draws (smoke tests)")
    ap.add_argument("--m_values", default=None, help="override coldstart.m_values, e.g. 5,10,full")
    ap.add_argument("--p2", action="store_true", help="also run protocol P2 (needs synthetic sets)")
    ap.add_argument("--p2-k", type=int, default=5, help="real defects kept for P2 training")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-notebook", action="store_true")
    ap.add_argument("--wait-for-gpu", action="store_true",
                    help="before every GPU run, wait until enough VRAM is free")
    ap.add_argument("--gpu-free-mb", type=int, default=5500, help="free VRAM needed to start a run")
    ap.add_argument("--vram-profile", default=None, help="auto | 8gb | full (default: config training)")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: config, 0)")
    a = ap.parse_args()
    setup_console()

    os.makedirs(a.work, exist_ok=True)
    os.chdir(a.work)
    print(f"code: {ROOT}\nwork: {os.getcwd()}\npython: {PY}")
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    models = [m.strip() if m.strip().endswith(".pt") else m.strip() + ".pt"
              for m in a.models.split(",") if m.strip()]
    bad = [m for m in models if m not in MODEL_RESULTS]
    if bad:
        print(f"unknown model(s) {bad}; choose from {list(MODEL_RESULTS)}")
        return 2
    overrides = sum(([f"--{k}", str(v)] for k, v in (("epochs", a.epochs), ("imgsz", a.imgsz),
                                                        ("batch", a.batch), ("vram_profile", a.vram_profile),
                                                        ("workers", a.workers)) if v is not None), [])
    if on_ac_power() is False:
        print("  WARNING: this laptop is on battery. A multi-hour GPU run should be plugged in "
              "(Windows also throttles the GPU on battery).", flush=True)
    with keep_awake("run_pipeline.py"):
        return _run_stages(a, datasets, models, overrides)


def _run_stages(a, datasets, models, overrides):
    t0 = time.time()
    if stage_prepare(datasets):
        return 1
    if a.with_cce and stage_encode():
        return 1
    if not a.skip_train:
        stage_train(datasets, a.seeds, models, a.with_cce, a.device, overrides,
                    a.wait_for_gpu, a.gpu_free_mb)
    stage_cache(models, a.device)
    stage_analyse(datasets, models)
    if a.generate:
        stage_generate(datasets, [g for g in a.generate.split(",") if g], a.synthetic_n, a.device)
    stage_synth(datasets, models, a.synth_mode, a.device)
    stage_sperc(datasets, models, a.seeds, a.n_draws, a.m_values)
    if a.p2:
        stage_p2(datasets, a.seeds, models, a.device, overrides, a.p2_k, a.n_draws, a.m_values,
                 a.wait_for_gpu, a.gpu_free_mb)
    stage_mvtec(a.n_draws, a.device)
    stage_seeds()
    if not a.skip_notebook:
        stage_notebook()
    print(f"\nPIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
