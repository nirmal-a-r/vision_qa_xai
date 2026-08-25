"""Compute audited faithfulness on a real sample and save it for the notebook."""
import os, sys, json, glob
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, r"C:/Users/nirma/Desktop/vision_qa_xai")
os.chdir(r"C:/Users/nirma/Desktop/vision_qa_xai")

import numpy as np, cv2
from ultralytics import YOLO
from src.xai.detector_saliency import occlusion_saliency, audited_faithfulness

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
DATASET = sys.argv[2] if len(sys.argv) > 2 else "neu"

res = [r for r in json.load(open("runs/results_yolov8s.json")) if r["dataset"] == DATASET][0]
model, imgsz = YOLO(res["weights"]), res["imgsz"]
recs = [r for r in json.load(open(f"runs/preds/{DATASET}_yolov8s_test.json"))["records"]
        if r["gt_boxes"]]
rng = np.random.default_rng(0)
pick = rng.choice(len(recs), size=min(N, len(recs)), replace=False)

rows = []
for k, i in enumerate(pick):
    r = recs[i]
    img = cv2.cvtColor(cv2.imread(r["file_name"]), cv2.COLOR_BGR2RGB)
    try:
        a = audited_faithfulness(model, img, r["gt_boxes"], imgsz, grid=8, steps=8)
    except Exception as e:
        print(f"  {r['image_id']}: FAILED {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    a.pop("heatmap")
    a["image_id"] = r["image_id"]
    a["n_gt"] = len(r["gt_boxes"])
    rows.append(a)
    if (k + 1) % 5 == 0:
        print(f"  {k+1}/{len(pick)}", flush=True)

if rows:
    arr = lambda key: np.array([x[key] for x in rows], dtype=float)
    print(f"\naudited faithfulness on {len(rows)} {DATASET} test images "
          f"(occlusion saliency, 8x8 grid):")
    print(f"  insertion AUC     : {arr('insertion_auc').mean():.3f} +/- {arr('insertion_auc').std():.3f}")
    print(f"  energy pointing   : {arr('energy_pointing').mean():.3f} "
          f"(chance {arr('chance_level').mean():.3f})")
    print(f"  pointing game     : {np.mean([x['pointing_game'] for x in rows]):.3f}")
    print(f"  composite         : {arr('faithfulness').mean():.3f} +/- {arr('faithfulness').std():.3f}")
    out = f"runs/faithfulness_{DATASET}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"wrote {out}")
