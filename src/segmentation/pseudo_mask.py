"""
pseudo_mask.py
==============
Pixel-level segmentation + damage % calculation.

IMPORTANT DEVIATION FROM PROJECT.md, DOCUMENTED HERE:
------------------------------------------------------
PROJECT.md's Phase 1 calls for a Mask2Former segmentation head trained
jointly/pipelined after the detector, using polygon ground-truth masks. The
two real datasets used here (PKU-Market-PCB, NEU-DET) only ship axis-aligned
bounding boxes — neither has polygon/pixel annotations, on Kaggle or
upstream. Training Mask2Former without real mask ground truth would mean
training it to reproduce rectangles, which adds a lot of complexity for no
actual pixel-accuracy gain over just using the box.

Instead, this module gets pixel-accurate masks *without* needing mask
labels: for every detected box, OpenCV GrabCut is initialized with that box
as the foreground prior and refines it into a real pixel mask of the defect
region. This is classical, well-understood CV (no training required), keeps
the codebase simple, and still gives real per-pixel `damage_pct` matching
PROJECT.md's formula:

    damage_pct = (defective_pixel_area / total_component_area) * 100

If you later obtain real polygon labels (e.g. via SAM-assisted labeling),
swap this module for a trained Mask2Former head — `overlay.py` and the
downstream JSON schema don't need to change, they just consume whatever
binary mask this function returns.
"""

import numpy as np
import cv2


def grabcut_mask_from_box(image_rgb: np.ndarray, box_xyxy, iters: int = 5, pad: int = 4) -> np.ndarray:
    """
    image_rgb: HxWx3 uint8 RGB image
    box_xyxy:  [x1, y1, x2, y2] in pixel coords
    Returns:   HxW uint8 binary mask (1 = defect pixel), zero outside a
               small padded region around the box.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=np.uint8)

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)

    try:
        cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model, iters, cv2.GC_INIT_WITH_RECT)
        binary = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        # GrabCut can fail on degenerate/tiny regions — fall back to the
        # raw box as the mask rather than dropping the detection.
        binary = np.zeros((h, w), dtype=np.uint8)
        binary[y1:y2, x1:x2] = 1

    # If GrabCut collapsed to nothing (common on very small/low-contrast
    # defects), fall back to the box too, so damage_pct is never zeroed out
    # by a segmentation failure on a real detection.
    if binary.sum() == 0:
        binary[y1:y2, x1:x2] = 1

    return binary


def build_component_mask(boxes_xyxy, labels, scores, image_rgb, grabcut_iters: int = 5, score_thresh: float = 0.5):
    """Combine per-box GrabCut masks into one full-image binary defect mask."""
    h, w = image_rgb.shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for box, score in zip(boxes_xyxy, scores):
        if score < score_thresh:
            continue
        m = grabcut_mask_from_box(image_rgb, box, iters=grabcut_iters)
        full_mask = np.maximum(full_mask, m)
    return full_mask


def damage_percentage(defect_mask: np.ndarray, component_mask: np.ndarray = None) -> float:
    """
    damage_pct = (defective pixel area / total component area) * 100

    If `component_mask` (the PCB/steel surface region, e.g. from a simple
    background threshold) isn't supplied, the whole image is treated as the
    component area — a reasonable default for these datasets, where the
    board/strip fills most of the frame.
    """
    defective_area = float(defect_mask.sum())
    if component_mask is not None:
        total_area = float(component_mask.sum())
    else:
        total_area = float(defect_mask.shape[0] * defect_mask.shape[1])
    if total_area <= 0:
        return 0.0
    return round(100.0 * defective_area / total_area, 2)
