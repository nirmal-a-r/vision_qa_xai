"""
dump_predictions.py
===================
Run a trained detector over a COCO split and cache per-image records.

Everything downstream - conformal calibration, the triage policy, the
faithfulness audit, the figures - consumes this cache rather than the model, so
the expensive forward passes happen exactly once and every later analysis is
reproducible from a single artefact.

Records are stored at a *low* score threshold on purpose. Conformal calibration
has to see the whole score distribution to choose an operating point; if we
pre-filtered at some default threshold, the calibration could never certify any
threshold below it and the guarantee would be conditioned on an arbitrary
choice made here.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from src.data.dataset import DefectCocoDataset, collate_fn
from src.data.transforms import get_val_transforms
from src.models.detector import build_model


DUMP_SCORE_FLOOR = 0.01   # keep almost everything; see module docstring


@torch.no_grad()
def dump_split(cfg, checkpoint_path, split, out_path, batch_size=4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    coco_json = cfg["paths"]["coco_merged_json"].replace(".json", f"_{split}.json")

    ds = DefectCocoDataset(
        coco_json=coco_json,
        images_dir=cfg["paths"]["images_merged_dir"],
        transforms=get_val_transforms(cfg["model"]["image_size"]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model, _ = build_model(
        num_classes=ckpt["num_classes"],
        backbone=cfg["model"]["backbone"],
        base_checkpoint=cfg["model"]["detector_checkpoint"],
        num_queries=cfg["model"]["num_queries"],
        image_size=cfg["model"]["image_size"],
        pretrained_backbone=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    label_to_name = {int(k): v for k, v in ckpt["label_to_name"].items()}

    records = []
    for batch in loader:
        pv = batch["pixel_values"].to(device)
        targets = batch["labels"]
        out = model(pixel_values=pv)
        probs = out.logits.sigmoid()
        scores, class_ids = probs.max(-1)
        boxes = out.pred_boxes

        for b in range(pv.shape[0]):
            t = targets[b]
            oh, ow = t["orig_size"].tolist()
            img_id = int(t["image_id"].item())

            # ground truth: cxcywh normalized -> xyxy pixels
            gt = t["boxes"].cpu().numpy().reshape(-1, 4)
            gt_xyxy = np.stack([
                (gt[:, 0] - gt[:, 2] / 2) * ow, (gt[:, 1] - gt[:, 3] / 2) * oh,
                (gt[:, 0] + gt[:, 2] / 2) * ow, (gt[:, 1] + gt[:, 3] / 2) * oh,
            ], axis=1) if len(gt) else np.zeros((0, 4))

            keep = (scores[b] >= DUMP_SCORE_FLOOR).cpu().numpy()
            pb = boxes[b].cpu().numpy()[keep]
            pred_xyxy = np.stack([
                (pb[:, 0] - pb[:, 2] / 2) * ow, (pb[:, 1] - pb[:, 3] / 2) * oh,
                (pb[:, 0] + pb[:, 2] / 2) * ow, (pb[:, 1] + pb[:, 3] / 2) * oh,
            ], axis=1) if len(pb) else np.zeros((0, 4))

            records.append({
                "image_id": img_id,
                "file_name": ds.coco.imgs[img_id]["file_name"],
                "orig_hw": [int(oh), int(ow)],
                "gt_boxes": gt_xyxy.round(2).tolist(),
                "gt_labels": t["class_labels"].cpu().numpy().astype(int).tolist(),
                "pred_boxes": pred_xyxy.round(2).tolist(),
                "pred_scores": scores[b].cpu().numpy()[keep].round(5).tolist(),
                "pred_labels": class_ids[b].cpu().numpy()[keep].astype(int).tolist(),
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"label_to_name": label_to_name, "split": split, "records": records}, f)

    n_gt = sum(len(r["gt_boxes"]) for r in records)
    n_pred = sum(len(r["pred_boxes"]) for r in records)
    print(f"[dump] {split}: {len(records)} images, {n_gt} GT boxes, "
          f"{n_pred} preds >= {DUMP_SCORE_FLOOR} -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for split in args.splits.split(","):
        dump_split(cfg, args.checkpoint, split.strip(),
                   os.path.join(args.out_dir, f"preds_{split.strip()}.json"))


if __name__ == "__main__":
    main()
