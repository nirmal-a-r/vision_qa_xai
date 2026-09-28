"""
composite.py
============
Turn the output of ANY few-shot defect generator into a labelled synthetic
defect set that the frozen detector can score.

The generators recommended in the plan (AnomalyDiffusion, AAAI 2024; the
training-free generator of ICCV 2025; MaCoDiff / JGTDiff, J. Intell. Manuf. 2026)
all emit (defect image, defect mask) pairs. This module is deliberately
generator-agnostic: it takes a folder of such pairs and writes

    <out>/images/syn/<name>.png     the synthetic defective image
    <out>/labels/syn/<name>.txt     YOLO labels, one box per mask component
    <out>/manifest.json             provenance for every synthetic image

The ``images/syn`` layout is the Ultralytics split layout, so the frozen detector
can score the set with ``src/evaluation/score_synthetic.py`` unchanged.

Two modes:

* ``full``   the generator already produced a whole defective image the size of
             a real part; boxes come straight from its mask.
* ``paste``  the generator produced a defect patch + mask; it is alpha-blended
             onto a clean training image at a random location, with a feathered
             mask edge so the seam is not an easier-to-detect artefact.

Rules that protect the guarantee (see docs/RESEARCH_PLAN.md, Section 4):
  * clean backgrounds and generator references come from the TRAIN split only;
  * nothing from the calibration split may be written here.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLIT = "syn"


def boxes_from_mask(mask: np.ndarray, min_area: int = 4) -> List[Tuple[int, int, int, int]]:
    """xyxy boxes of the connected components of a binary mask."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append((int(x), int(y), int(x + w), int(y + h)))
    return out


def yolo_lines(boxes, W: int, H: int, cls: int = 0) -> List[str]:
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def paste_defect(background: np.ndarray, patch: np.ndarray, patch_mask: np.ndarray,
                 xy: Tuple[int, int], feather: int = 3):
    """Alpha-blend ``patch`` onto ``background`` at top-left ``xy``.

    Returns (image, full-size binary mask). The patch is clipped at the border.
    """
    bg = background.copy()
    H, W = bg.shape[:2]
    ph, pw = patch.shape[:2]
    x, y = xy
    x2, y2 = min(W, x + pw), min(H, y + ph)
    if x2 <= x or y2 <= y:
        raise ValueError("patch placed entirely outside the background")
    pm = (np.asarray(patch_mask) > 0).astype(np.float32)[: y2 - y, : x2 - x]
    if feather > 0:
        k = 2 * feather + 1
        alpha = cv2.GaussianBlur(pm, (k, k), 0)
        alpha = np.maximum(alpha, pm * 0.999) if pm.any() else alpha
    else:
        alpha = pm
    if bg.ndim == 3:
        alpha = alpha[..., None]
    region = bg[y:y2, x:x2].astype(np.float32)
    src = patch[: y2 - y, : x2 - x].astype(np.float32)
    if src.ndim == 2 and bg.ndim == 3:
        src = np.repeat(src[..., None], 3, axis=2)
    if src.ndim == 3 and bg.ndim == 2:
        src = src.mean(axis=2)
    bg[y:y2, x:x2] = np.clip(alpha * src + (1 - alpha) * region, 0, 255).astype(bg.dtype)
    full = np.zeros((H, W), np.uint8)
    full[y:y2, x:x2] = (pm > 0).astype(np.uint8) * 255
    return bg, full


def _pairs(gen_dir: str):
    """(image, mask) pairs: <stem>.<ext> with <stem>_mask.png beside it."""
    pairs = []
    for p in sorted(glob.glob(os.path.join(gen_dir, "*"))):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() not in IMG_EXT or stem.endswith("_mask"):
            continue
        mp = os.path.join(gen_dir, stem + "_mask.png")
        if os.path.exists(mp):
            pairs.append((p, mp))
    return pairs


@dataclass
class BuildReport:
    written: int
    skipped: int
    out_dir: str


def build_synthetic_set(gen_dir: str, out_dir: str, mode: str = "full",
                        clean_dir: Optional[str] = None, cls: int = 0,
                        n_per_pair: int = 1, seed: int = 0,
                        min_area: int = 4) -> BuildReport:
    """Write a YOLO-format synthetic defect set from generator output.

    ``gen_dir`` holds ``<stem>.png`` + ``<stem>_mask.png`` pairs. In ``paste`` mode
    ``clean_dir`` must hold clean TRAIN images; each pair is pasted ``n_per_pair``
    times at random positions on random clean images.
    """
    if mode not in ("full", "paste"):
        raise ValueError("mode must be 'full' or 'paste'")
    pairs = _pairs(gen_dir)
    if not pairs:
        raise FileNotFoundError(f"no <stem>.png + <stem>_mask.png pairs in {gen_dir}")
    rng = np.random.default_rng(seed)
    clean = []
    if mode == "paste":
        if not clean_dir:
            raise ValueError("paste mode needs clean_dir (clean TRAIN images)")
        clean = [p for p in sorted(glob.glob(os.path.join(clean_dir, "*")))
                 if p.lower().endswith(IMG_EXT)]
        if not clean:
            raise FileNotFoundError(f"no clean images in {clean_dir}")
    img_out = os.path.join(out_dir, "images", SPLIT)
    lbl_out = os.path.join(out_dir, "labels", SPLIT)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    manifest, written, skipped = [], 0, 0
    for pi, (ip, mp) in enumerate(pairs):
        img = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            skipped += 1
            continue
        reps = 1 if mode == "full" else n_per_pair
        for k in range(reps):
            if mode == "full":
                out_img, out_msk, bg = img, msk, None
            else:
                bg = clean[int(rng.integers(len(clean)))]
                bgi = cv2.imread(bg, cv2.IMREAD_UNCHANGED)
                H, W = bgi.shape[:2]
                ph, pw = img.shape[:2]
                if ph > H or pw > W:
                    skipped += 1
                    continue
                xy = (int(rng.integers(0, W - pw + 1)), int(rng.integers(0, H - ph + 1)))
                out_img, out_msk = paste_defect(bgi, img, msk, xy)
            boxes = boxes_from_mask(out_msk, min_area)
            if not boxes:
                skipped += 1
                continue
            name = f"syn_{pi:05d}_{k:02d}"
            H, W = out_img.shape[:2]
            cv2.imwrite(os.path.join(img_out, name + ".png"), out_img)
            with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines(boxes, W, H, cls)) + "\n")
            manifest.append({"name": name, "source_image": os.path.basename(ip),
                             "source_mask": os.path.basename(mp), "background": bg,
                             "mode": mode, "n_boxes": len(boxes)})
            written += 1
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return BuildReport(written, skipped, out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen_dir", required=True, help="folder of <stem>.png + <stem>_mask.png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "paste"])
    ap.add_argument("--clean_dir", default=None, help="clean TRAIN images (paste mode)")
    ap.add_argument("--cls", type=int, default=0)
    ap.add_argument("--n_per_pair", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = build_synthetic_set(a.gen_dir, a.out_dir, a.mode, a.clean_dir, a.cls, a.n_per_pair, a.seed)
    print(f"wrote {r.written} synthetic images to {r.out_dir} (skipped {r.skipped})")
