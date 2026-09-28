"""
score_synthetic.py
==================
Score a synthetic defect set with the FROZEN detector and cache the detections in
the same format as the real calibration/test caches, so SPERC can read real and
synthetic escape scores the same way.

    python -m src.evaluation.score_synthetic \
        --weights runs/detect/kolektor_yolov8s_s0/weights/best.pt --imgsz 512 \
        --syn_root data/synthetic/kolektor_training_free \
        --out runs/preds/kolektor_yolov8s_syn_training_free.json

The synthetic root is what ``src/synth/composite.py`` writes
(``images/syn`` + ``labels/syn``). The detector must be the same weights used for
the real calibration scores, and it must never have been trained on the
calibration split.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.dump_ultralytics_preds import dump_split   # noqa: E402
from src.synth.composite import SPLIT                           # noqa: E402


def main():
    try:
        from src.utils.winenv import setup_console
        setup_console()
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--syn_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    dump_split(a.weights, a.syn_root, SPLIT, a.out, imgsz=a.imgsz,
               device=int(a.device) if a.device.isdigit() else a.device)


if __name__ == "__main__":
    main()
