"""
smoke_test.py - run EVERY pipeline stage end to end on a tiny copy of the data,
in a separate workspace, before committing GPU-days to the real run.

It copies a few dozen images per dataset from data/raw into <work>/data/raw, then
runs scripts/run_pipeline.py there with 1-epoch training, 2 seeds, both detectors,
copy-paste synthetic defects, protocol P2, a short sweep and the notebook. Nothing
in the repository's own data/ or runs/ is touched. Numbers from it mean nothing;
it only proves that every stage runs and hands the next one what it needs.

    python scripts/smoke_test.py                  # GPU 0
    python scripts/smoke_test.py --device cpu     # no GPU (slow but works)
    python scripts/smoke_test.py --keep           # keep the workspace afterwards

Workspace: runs_smoke/ (git-ignored). Exit code 0 = every stage produced its outputs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils.winenv import keep_awake, rmtree, setup_console   # noqa: E402


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)


def subset(raw, work_raw, n_def=30, n_clean=30, n_voc=60, seed=0):
    rng = np.random.default_rng(seed)
    made = {}
    # KolektorSDD2: <id>.png + <id>_GT.png
    k = os.path.join(raw, "KOLEKTORSDD2", "train")
    if os.path.isdir(k):
        imgs = sorted(p for p in glob.glob(os.path.join(k, "*.png")) if not p.endswith("_GT.png"))
        rng.shuffle(imgs)
        d = c = 0
        for p in imgs:
            gt = cv2.imread(p[:-4] + "_GT.png", cv2.IMREAD_GRAYSCALE)
            if gt is None:
                continue
            bad = bool(gt.max() > 0)
            if (bad and d < n_def) or (not bad and c < n_clean):
                for q in (p, p[:-4] + "_GT.png"):
                    _copy(q, os.path.join(work_raw, "KOLEKTORSDD2", "train", os.path.basename(q)))
                d, c = d + bad, c + (not bad)
            if d >= n_def and c >= n_clean:
                break
        made["kolektor"] = (d, c)
    # Magnetic-Tile: MT_<cls>/Imgs/<stem>.jpg (+ .png mask)
    mt = os.path.join(raw, "MAGNETIC-TILE")
    if os.path.isdir(mt):
        d = c = 0
        for cls in ("MT_Free", "MT_Blowhole", "MT_Crack", "MT_Break"):
            jpgs = sorted(glob.glob(os.path.join(mt, cls, "Imgs", "*.jpg")))
            take = n_clean if cls == "MT_Free" else n_def // 3
            for p in jpgs[:take]:
                for q in (p, p[:-4] + ".png"):
                    if os.path.exists(q):
                        _copy(q, os.path.join(work_raw, "MAGNETIC-TILE", cls, "Imgs", os.path.basename(q)))
                if cls == "MT_Free":
                    c += 1
                else:
                    d += 1
        made["magnetic_tile"] = (d, c)
    # GC10-DET: lable/*.xml + <class dir>/<stem>.jpg
    g = os.path.join(raw, "GC10-DET")
    if os.path.isdir(g):
        index = {}
        for cls_dir in glob.glob(os.path.join(g, "*")):
            if os.path.isdir(cls_dir) and os.path.basename(cls_dir).isdigit():
                for p in glob.glob(os.path.join(cls_dir, "*.jpg")):
                    index[os.path.splitext(os.path.basename(p))[0]] = p
        xmls = sorted(glob.glob(os.path.join(g, "lable", "*.xml")))
        rng.shuffle(xmls)
        n = 0
        for x in xmls:
            s = os.path.splitext(os.path.basename(x))[0]
            if s in index:
                _copy(x, os.path.join(work_raw, "GC10-DET", "lable", os.path.basename(x)))
                _copy(index[s], os.path.join(work_raw, "GC10-DET", os.path.basename(os.path.dirname(index[s])),
                                             os.path.basename(index[s])))
                n += 1
            if n >= n_voc:
                break
        made["gc10"] = (n, 0)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "runs_smoke"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--datasets", default="kolektor,magnetic_tile,gc10")
    ap.add_argument("--models", default="rtdetr-l.pt,yolov8s.pt")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-p2", action="store_true")
    a = ap.parse_args(argv)
    setup_console()

    if os.path.exists(a.work) and not a.keep and not rmtree(a.work):
        print(f"could not clear {a.work} (a file is open?) - close it or pass --keep")
        return 1
    made = subset(os.path.join(ROOT, "data", "raw"), os.path.join(a.work, "data", "raw"))
    print(f"smoke workspace {a.work}: (defective, clean) per dataset {made}")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_pipeline.py"), "--work", a.work,
           "--datasets", a.datasets, "--models", a.models, "--seeds", a.seeds,
           "--epochs", "1", "--imgsz", str(a.imgsz), "--batch", "4", "--device", a.device,
           "--generate", "copy_paste", "--synthetic-n", "40", "--n_draws", "30",
           "--m_values", "5,10,full"] + ([] if a.no_p2 else ["--p2"])
    print(" ".join(cmd), flush=True)
    with keep_awake("smoke test"):
        rc = subprocess.call(cmd)

    # every stage must have left its artefact behind
    w = lambda *p: os.path.join(a.work, *p)                       # noqa: E731
    ds = a.datasets.split(",")
    prim = [d for d in ds if d in ("kolektor", "magnetic_tile")]
    tags = [m.replace(".pt", "") + ("" if s == "0" else f"_s{s}")
            for m in a.models.split(",") for s in a.seeds.split(",")]
    checks = {
        "1 prepare": all(os.path.exists(w("data", "yolo", d, "data.yaml")) for d in ds),
        "3 train": all(os.path.exists(w("runs", {"yolov8s.pt": "results_yolov8s.json",
                                                   "rtdetr-l.pt": "results_rtdetr.json"}[m]))
                       for m in a.models.split(",")),
        "4 cache": all(os.path.exists(w("runs", "preds", f"{d}_{t}_cal.json")) for d in ds for t in tags),
        "5 analyse": os.path.exists(w("runs", "risk_results.json")),
        "6 generate": all(len(glob.glob(w("data", "generated", d, "copy_paste", "*_mask.png"))) > 0 for d in prim),
        "6b synth scored": all(os.path.exists(w("runs", "preds", f"{d}_{t}_syn_copy_paste.json"))
                               for d in prim for t in tags),
        "7a commissioning": os.path.exists(w("runs", "commissioning_table.csv")),
        "7b stage 1": (("gc10" not in ds) or os.path.exists(w("runs", "stage1", "stage1_verdict.json"))),
        "7c sweep": all(os.path.exists(w("runs", "coldstart", f"coldstart_{t}.json")) for t in tags),
        "7d summary": os.path.exists(w("runs", "coldstart", "summary", "summary.json")),
        "7e P2": a.no_p2 or bool(glob.glob(w("runs", "coldstart_p2", "coldstart_*_p2_copy_paste*.json"))),
        "8 seeds": os.path.exists(w("runs", "seed_summary.json")),
        "9 notebook": os.path.exists(w("VisionQA_CertifiedInspection_executed.ipynb")),
    }
    print("\nSMOKE TEST")
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    ok = all(checks.values()) and rc == 0
    with open(w("smoke_result.json"), "w") as f:
        json.dump({"pipeline_rc": rc, "checks": checks, "ok": ok}, f, indent=1)
    print("SMOKE TEST PASSED" if ok else f"SMOKE TEST FAILED (pipeline rc={rc})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
