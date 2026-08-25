"""
detector_saliency.py
====================
Saliency maps for a black-box detector, and the audited faithfulness score
built on them.

Occlusion sensitivity is used rather than a gradient or CAM method, for a
reason that matters to this paper specifically. The faithfulness score gates the
triage policy, so it has to be computable for *any* detector the framework is
applied to - including one shipped as a binary by a vendor, which is the normal
situation on a production line. Gradient and CAM methods need internals: named
layers, a backward pass, an architecture the hook code understands. Occlusion
needs only `predict`, so the same audit runs unchanged over YOLO, RT-DETR, or a
closed-source model.

The cost is forward passes: a G x G grid costs G^2 of them per image. That is
why the notebook's full policy sweep uses a cheap proxy and the audited score is
reported on a sample - stated plainly rather than hidden, since the distinction
matters when reading the numbers.
"""

from __future__ import annotations

import numpy as np
import cv2


def _score_of(model, img, imgsz, conf=0.001):
    """Top detection confidence for one image - the scalar being explained."""
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False, device=0)[0]
    b = r.boxes
    return float(b.conf.max().item()) if len(b) else 0.0


def occlusion_saliency(model, image_rgb, imgsz, grid=8, patch_fill="blur"):
    """Score drop when each cell of a G x G grid is occluded.

    Returns an HxW map in [0,1]: high where hiding the region hurts the
    detector most, i.e. where the evidence actually is.
    """
    H, W = image_rgb.shape[:2]
    base = _score_of(model, image_rgb, imgsz)
    if patch_fill == "blur":
        k = max(3, (min(H, W) // 8) | 1)
        filler = cv2.GaussianBlur(image_rgb, (k, k), 0)
    else:
        filler = np.zeros_like(image_rgb)

    drops = np.zeros((grid, grid), dtype=float)
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    for i in range(grid):
        for j in range(grid):
            work = image_rgb.copy()
            work[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = filler[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            drops[i, j] = base - _score_of(model, work, imgsz)

    drops = np.clip(drops, 0, None)
    if drops.max() > 0:
        drops /= drops.max()
    return cv2.resize(drops.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)


def audited_faithfulness(model, image_rgb, gt_boxes, imgsz, grid=8, steps=10):
    """Full audited faithfulness for one image, plus its components.

    Returns insertion AUC, energy pointing game (and its chance level), the
    pointing game, and the composite score the triage policy gates on.
    """
    from .faithfulness import (insertion_curve, normalized_auc,
                               energy_pointing_game, box_area_fraction,
                               pointing_game, faithfulness_score)

    hm = occlusion_saliency(model, image_rgb, imgsz, grid=grid)
    score_fn = lambda im: _score_of(model, im, imgsz)

    ins = normalized_auc(insertion_curve(image_rgb, hm, score_fn, steps=steps))
    epg = energy_pointing_game(hm, gt_boxes)
    chance = box_area_fraction(gt_boxes, hm.shape)
    return {
        "insertion_auc": ins,
        "energy_pointing": epg,
        "chance_level": chance,
        "pointing_game": bool(pointing_game(hm, gt_boxes)),
        "faithfulness": float(faithfulness_score(hm, gt_boxes, image_rgb, score_fn,
                                                 steps=steps)),
        "heatmap": hm,
    }


def randomized_model_sanity(model_factory, weights, image_rgb, imgsz, grid=8):
    """Adebayo et al. model-randomisation check.

    Explain with the trained model, then with randomly re-initialised weights.
    A high rank correlation between the two maps means the "explanation" is
    tracking image structure rather than anything the model learned, and the
    method should be discarded no matter how convincing its heatmaps look.
    """
    import torch
    from .faithfulness import map_rank_correlation

    trained = model_factory(weights)
    hm_trained = occlusion_saliency(trained, image_rgb, imgsz, grid=grid)

    rand = model_factory(weights)
    with torch.no_grad():
        for p in rand.model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                p.zero_()
    hm_random = occlusion_saliency(rand, image_rgb, imgsz, grid=grid)

    return {"rank_correlation": map_rank_correlation(hm_trained, hm_random),
            "heatmap_trained": hm_trained, "heatmap_random": hm_random}
