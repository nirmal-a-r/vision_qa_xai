"""
evaluate.py
===========
COCO mAP evaluation against the held-out test split, against the >95% mAP
target from PROJECT.md's M2 milestone. Writes a results table to
outputs/eval/.
"""

import os
import json
import yaml
import argparse
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data.dataset import DefectCocoDataset, collate_fn
from src.data.transforms import get_val_transforms
from src.models.detector import build_model
from torch.utils.data import DataLoader


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference_for_eval(model, loader, device, score_thresh=0.05):
    model.eval()
    results = []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"]
        outputs = model(pixel_values=pixel_values)

        orig_sizes = torch.stack([t["orig_size"] for t in labels]).to(device)
        probs = outputs.logits.sigmoid()
        scores, class_ids = probs.max(-1)
        boxes = outputs.pred_boxes  # cxcywh normalized

        for b in range(pixel_values.shape[0]):
            h, w = orig_sizes[b].tolist()
            img_id = int(labels[b]["image_id"].item())
            for q in range(boxes.shape[1]):
                s = scores[b, q].item()
                if s < score_thresh:
                    continue
                cx, cy, bw, bh = boxes[b, q].tolist()
                x = (cx - bw / 2) * w
                y = (cy - bh / 2) * h
                results.append({
                    "image_id": img_id,
                    "category_id": int(class_ids[b, q].item()),
                    "bbox": [x, y, bw * w, bh * h],
                    "score": s,
                })
    return results


def evaluate(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_json = cfg["paths"]["coco_merged_json"].replace(".json", "_test.json")
    test_ds = DefectCocoDataset(
        coco_json=test_json,
        images_dir=cfg["paths"]["images_merged_dir"],
        transforms=get_val_transforms(cfg["model"]["image_size"]),
    )
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model, _ = build_model(
        num_classes=ckpt["num_classes"],
        backbone=cfg["model"]["backbone"],
        use_timm_backbone=cfg["model"]["use_timm_backbone"],
        base_checkpoint=cfg["model"]["detector_checkpoint"],
        num_queries=cfg["model"]["num_queries"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    predictions = run_inference_for_eval(model, test_loader, device)

    os.makedirs(cfg["paths"]["eval_dir"], exist_ok=True)
    pred_path = os.path.join(cfg["paths"]["eval_dir"], "test_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f)

    coco_gt = COCO(test_json)
    if len(predictions) == 0:
        print("[evaluate] no predictions above threshold — skipping COCOeval")
        return None
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP@[.5:.95]": coco_eval.stats[0],
        "mAP@.5": coco_eval.stats[1],
        "mAP@.75": coco_eval.stats[2],
    }
    with open(os.path.join(cfg["paths"]["eval_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] {metrics}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    evaluate(args.config, args.checkpoint)
