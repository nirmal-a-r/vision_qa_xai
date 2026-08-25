"""
dump_ultralytics_preds.py
=========================
Cache per-image detections from a trained ultralytics detector (YOLO or RT-DETR)
for the calibration and test blocks.

Everything downstream - conformal calibration, the triage policy, the
faithfulness audit, every figure - reads this cache instead of the model, so the
forward passes happen once and every later analysis is reproducible from a
single artefact.

Two details that are easy to get wrong and would quietly invalidate the
guarantees:

* `conf` is set near zero. Conformal calibration must see the whole score
  distribution in order to *choose* an operating point; pre-filtering here at
  some default would make every threshold below that filter uncertifiable, and
  the resulting "guarantee" would be conditioned on an arbitrary constant baked
  in at dump time.

* Defect-free images are kept. They carry no boxes and contribute nothing to
  escape risk, but they are the entire basis of the review-load and
  false-scrap statistics - dropping them (the natural thing to do when you
  iterate over annotations rather than images) would make the triage numbers
  meaningless.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse

os.environ.setdefault("TQDM_DISABLE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DUMP_CONF = 0.001          # see module docstring
DUMP_MAX_DET = 300


def load_model(weights):
    from ultralytics import YOLO, RTDETR
    return (RTDETR if "rtdetr" in str(weights).lower() else YOLO)(weights)


def read_yolo_labels(label_path, W, H):
    """YOLO txt -> (xyxy boxes, class ids). Missing/empty file = genuine negative."""
    boxes, cls = [], []
    if not os.path.exists(label_path):
        return boxes, cls
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *[float(v) for v in parts[1:5]]
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
            cls.append(c)
    return boxes, cls


def dump_split(weights, yolo_root, split, out_path, imgsz, device=0, batch=16):
    import cv2

    model = load_model(weights)
    img_dir = os.path.join(yolo_root, "images", split)
    lbl_dir = os.path.join(yolo_root, "labels", split)
    paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    if not paths:
        raise FileNotFoundError(f"no images in {img_dir}")

    records = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        results = model.predict(chunk, imgsz=imgsz, conf=DUMP_CONF,
                                max_det=DUMP_MAX_DET, device=device,
                                verbose=False, stream=False)
        for p, r in zip(chunk, results):
            H, W = r.orig_shape
            stem = os.path.splitext(os.path.basename(p))[0]
            gt_boxes, gt_labels = read_yolo_labels(os.path.join(lbl_dir, stem + ".txt"), W, H)
            b = r.boxes
            records.append({
                "image_id": stem,
                "file_name": p,
                "orig_hw": [int(H), int(W)],
                "gt_boxes": [[round(v, 2) for v in bb] for bb in gt_boxes],
                "gt_labels": gt_labels,
                "pred_boxes": [[round(float(v), 2) for v in bb]
                               for bb in b.xyxy.cpu().numpy()] if len(b) else [],
                "pred_scores": [round(float(v), 5) for v in b.conf.cpu().numpy()] if len(b) else [],
                "pred_labels": [int(v) for v in b.cls.cpu().numpy()] if len(b) else [],
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"split": split, "weights": str(weights), "imgsz": imgsz,
                   "records": records}, f)

    n_gt = sum(len(r["gt_boxes"]) for r in records)
    n_neg = sum(1 for r in records if not r["gt_boxes"])
    n_pred = sum(len(r["pred_boxes"]) for r in records)
    print(f"  [{split:5s}] {len(records):5d} imgs ({n_neg:4d} clean) "
          f"{n_gt:5d} GT  {n_pred:6d} preds >= {DUMP_CONF} -> {out_path}", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="runs/results_*.json from training")
    ap.add_argument("--yolo_dir", default="data/yolo")
    ap.add_argument("--out_dir", default="runs/preds")
    ap.add_argument("--splits", default="cal,test")
    a = ap.parse_args()

    with open(a.results) as f:
        runs = json.load(f)

    for r in runs:
        w = r["weights"]
        if not os.path.exists(w):
            print(f"[{r['dataset']}] missing weights {w}, skipping")
            continue
        tag = f"{r['dataset']}_{r['model'].replace('.pt', '')}"
        print(f"[{tag}] imgsz={r['imgsz']}", flush=True)
        for split in a.splits.split(","):
            out = os.path.join(a.out_dir, f"{tag}_{split.strip()}.json")
            if os.path.exists(out):
                print(f"  [{split.strip():5s}] cached, skipping")
                continue
            try:
                dump_split(w, os.path.join(a.yolo_dir, r["dataset"]), split.strip(),
                           out, imgsz=r["imgsz"])
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [{split}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
