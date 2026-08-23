"""
inference.py
============
End-to-end Phase 1 + Phase 2 inference for a single image, producing the
exact JSON handoff record PROJECT.md section 1.5 specifies for Shruhath's
Phase 3/4 to consume:

    {
      "image_id": "string",
      "boxes": [[x1, y1, x2, y2], ...],
      "labels": ["scratch", "mouse_bite", ...],
      "scores": [0.93, ...],
      "masks": "path/to/mask.png",
      "damage_pct": 12.4,
      "heatmap_overlay": "outputs/heatmaps/xxx.png"
    }

Field names/types are fixed per PROJECT.md's warning not to change them
without telling Shruhath — this module is the single place they're produced,
so any future change only has to happen here.
"""

import os
import json
import yaml
import numpy as np
import torch
from PIL import Image

from src.models.detector import build_model
from src.data.transforms import get_val_transforms
from src.segmentation.pseudo_mask import build_component_mask, damage_percentage
from src.xai.rollout import SwinAttentionRollout, gradient_relevance, combine_heatmaps
from src.xai.overlay import render_heatmap_overlay, save_overlay


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class VisionQAPipeline:
    def __init__(self, config_path: str, checkpoint_path: str):
        self.cfg = load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.label_to_name = ckpt["label_to_name"]
        self.model, _ = build_model(
            num_classes=ckpt["num_classes"],
            backbone=self.cfg["model"]["backbone"],
            use_timm_backbone=self.cfg["model"]["use_timm_backbone"],
            base_checkpoint=self.cfg["model"]["detector_checkpoint"],
            num_queries=self.cfg["model"]["num_queries"],
            image_size=self.cfg["model"]["image_size"],
            pretrained_backbone=False,   # overwritten by the checkpoint below
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.transform = get_val_transforms(self.cfg["model"]["image_size"])
        self.score_thresh = float(self.cfg.get("inference", {}).get("score_thresh", 0.25))

        try:
            self.rollout = SwinAttentionRollout(self.model, discard_ratio=self.cfg["xai"]["discard_ratio"])
        except Exception as e:
            print(f"[inference] Swin rollout hooks unavailable ({e}); will use gradient relevance only.")
            self.rollout = None

    @torch.no_grad()
    def _detect(self, pixel_values: torch.Tensor, orig_hw, score_thresh: float = None):
        score_thresh = self.score_thresh if score_thresh is None else score_thresh
        outputs = self.model(pixel_values=pixel_values.unsqueeze(0).to(self.device))
        probs = outputs.logits.sigmoid()[0]         # [num_queries, num_classes]
        scores, class_ids = probs.max(-1)
        boxes_cxcywh = outputs.pred_boxes[0]         # [num_queries, 4] normalized

        h, w = orig_hw
        keep = scores > score_thresh
        boxes_xyxy, labels, kept_scores, query_indices = [], [], [], []
        for q in torch.nonzero(keep).flatten().tolist():
            cx, cy, bw, bh = boxes_cxcywh[q].tolist()
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
            boxes_xyxy.append([x1, y1, x2, y2])
            cid = int(class_ids[q].item())
            labels.append(self.label_to_name.get(cid, str(cid)))
            kept_scores.append(round(float(scores[q].item()), 4))
            query_indices.append((q, cid))

        return boxes_xyxy, labels, kept_scores, query_indices, outputs

    def run(self, image_path: str, image_id: str = None, out_dir: str = None) -> dict:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_id = image_id or os.path.splitext(os.path.basename(image_path))[0]
        out_dir = out_dir or self.cfg["paths"]["outputs_dir"]

        transformed = self.transform(image=image_np, bboxes=[], labels=[])
        pixel_values = transformed["image"]

        boxes_xyxy, labels, scores, query_indices, outputs = self._detect(
            pixel_values, orig_hw=(image_np.shape[0], image_np.shape[1])
        )

        # ---- Phase 2a: pixel masks + damage % (see segmentation/pseudo_mask.py) ----
        component_mask = build_component_mask(
            boxes_xyxy, labels, scores, image_np,
            grabcut_iters=self.cfg["segmentation"]["grabcut_iters"],
            # _detect already applied the threshold; without passing it here
            # build_component_mask re-filters at its own 0.5 default and
            # silently drops every box back out again.
            score_thresh=self.score_thresh,
        )
        damage_pct = damage_percentage(component_mask)

        mask_path = os.path.join(out_dir, "masks", f"{image_id}_mask.png")
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        Image.fromarray((component_mask * 255).astype(np.uint8)).save(mask_path)

        # ---- Phase 2b: attribution heatmap -----------------------------------
        heatmaps = []
        if self.rollout is not None:
            try:
                heatmaps.append(self.rollout(pixel_values.unsqueeze(0).to(self.device)))
            except Exception as e:
                print(f"[inference] rollout failed for {image_id}: {e}")

        if query_indices:
            q, cid = query_indices[0]  # attribute w.r.t. the top detection
            try:
                grad_map = gradient_relevance(self.model, pixel_values.unsqueeze(0).to(self.device),
                                               query_index=q, class_index=cid)
                heatmaps.append(grad_map)
            except Exception as e:
                print(f"[inference] gradient relevance failed for {image_id}: {e}")

        heatmap_path = os.path.join(self.cfg["paths"]["heatmaps_dir"], f"{image_id}_heatmap.png")
        if heatmaps:
            fused = combine_heatmaps(heatmaps)
            resized_img = pixel_values.permute(1, 2, 0).cpu().numpy()
            resized_img = np.clip((resized_img * np.array([0.229, 0.224, 0.225])
                                    + np.array([0.485, 0.456, 0.406])) * 255, 0, 255).astype(np.uint8)
            # boxes were computed in original-image coords; rescale to the
            # (possibly padded/resized) tensor used for the heatmap overlay
            scale_x = resized_img.shape[1] / image_np.shape[1]
            scale_y = resized_img.shape[0] / image_np.shape[0]
            scaled_boxes = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                             for x1, y1, x2, y2 in boxes_xyxy]
            overlay = render_heatmap_overlay(
                resized_img, fused, alpha=self.cfg["xai"]["overlay_alpha"],
                boxes_xyxy=scaled_boxes, labels_text=labels,
            )
            save_overlay(overlay, heatmap_path)
        else:
            heatmap_path = None

        # ---- decision rule (PROJECT.md 1.4) -----------------------------------
        decision = "Replace" if damage_pct > self.cfg["decision_rule"]["replace_damage_pct_threshold"] else "Rework"

        record = {
            "image_id": image_id,
            "boxes": [[round(v, 2) for v in b] for b in boxes_xyxy],
            "labels": labels,
            "scores": scores,
            "masks": mask_path,
            "damage_pct": damage_pct,
            "heatmap_overlay": heatmap_path,
            "decision": decision,  # convenience field, not required by the I/O contract
        }
        return record


def run_batch(config_path: str, checkpoint_path: str, image_paths, out_json: str):
    pipeline = VisionQAPipeline(config_path, checkpoint_path)
    records = [pipeline.run(p) for p in image_paths]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[inference] wrote {len(records)} records to {out_json}")
    return records


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    pipeline = VisionQAPipeline(args.config, args.checkpoint)
    record = pipeline.run(args.image)
    print(json.dumps(record, indent=2))
