"""
overlay.py
==========
Renders an XAI heatmap (HxW float [0,1], from rollout.py) as a colored
overlay on the original image, plus optional box/mask drawing — the "PNG
overlays rendered in Streamlit UI" output from PROJECT.md's Phase 2 table.
(Any image viewer works for a Kaggle notebook; Streamlit is just the eventual
consumer downstream.)
"""

import os
import numpy as np
import cv2


def render_heatmap_overlay(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    boxes_xyxy=None,
    labels_text=None,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    image_rgb: HxWx3 uint8
    heatmap:   HxW float in [0,1] (already resized to image_rgb's shape)
    mask:      optional HxW binary defect mask (from pseudo_mask.py), drawn
               as a translucent green fill so the damage-% region is visible
               alongside the attribution heatmap.
    Returns HxWx3 uint8 RGB overlay image.
    """
    h, w = image_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))

    heat_u8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heat_color, alpha, 0)

    if mask is not None:
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(overlay, 0.6, green, 0.4, 0)[mask_bool]

    if boxes_xyxy is not None:
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
            if labels_text is not None and i < len(labels_text):
                cv2.putText(overlay, labels_text[i], (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def save_overlay(overlay_rgb: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    return out_path
