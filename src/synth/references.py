"""
references.py
=============
Pick the 3-5 real reference defects a few-shot generator is allowed to see, and
the clean backgrounds it may paint on - from the TRAIN split only.

This is the rule that keeps SPERC's Tier N meaningful (project document, Section
8.5, assumption 3): generator references must come from the detector's training
split, never from calibration or test. ``select_references`` enforces it by
reading only ``data/processed/splits/<dataset>_train.json`` and then checking
that none of the chosen files appears in the cal / test / val splits.

Masks are read from each dataset's own pixel annotations when it has them
(KolektorSDD2 ``*_GT.png``, Magnetic-Tile ``*.png`` beside the ``.jpg``, DAGM
``Label/*_label.PNG``, MVTec AD ``ground_truth``); box-only datasets (NEU, GC10,
PCB) get filled boxes.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class Reference:
    file_name: str
    image_id: int
    boxes: list                       # xyxy
    class_name: str
    mask: Optional[np.ndarray] = field(default=None, repr=False)


def _split_json(dataset, split, processed_dir="data/processed"):
    p = os.path.join(processed_dir, "splits", f"{dataset}_{split}.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing - run the pipeline's prepare stage first")
    with open(p) as f:
        return json.load(f)


def _boxes_by_image(d):
    by = defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        by[a["image_id"]].append(([x, y, x + w, y + h], a["category_id"]))
    return by


def mask_for_image(dataset: str, file_name: str, shape, boxes) -> np.ndarray:
    """Binary uint8 mask (0/255) of the defect pixels of one image."""
    H, W = shape[:2]
    stem, _ = os.path.splitext(file_name)
    cands = []
    if dataset == "kolektor":
        cands = [stem + "_GT.png"]
    elif dataset == "magnetic_tile":
        cands = [stem + ".png"]
    elif dataset == "dagm":
        d, b = os.path.split(stem)
        cands = [os.path.join(d, "Label", b + "_label.PNG"), os.path.join(d, "Label", b + "_label.png")]
    elif dataset == "mvtec":
        parts = stem.replace("\\", "/").split("/")
        if len(parts) >= 3:
            root = "/".join(parts[:-3])
            cands = [os.path.join(root, "ground_truth", parts[-2], parts[-1] + "_mask.png")]
    for c in cands:
        if os.path.exists(c):
            m = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                return ((m > 0) * 255).astype(np.uint8)
    m = np.zeros((H, W), np.uint8)
    for x1, y1, x2, y2 in boxes:
        m[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 255
    return m


def held_out_files(dataset, processed_dir="data/processed"):
    """Every file in the val / cal / test splits (what references must avoid)."""
    out = set()
    for s in ("val", "cal", "test"):
        try:
            out |= {os.path.normcase(os.path.abspath(im["file_name"]))
                    for im in _split_json(dataset, s, processed_dir)["images"]}
        except FileNotFoundError:
            pass
    return out


def select_references(dataset: str, k: int = 5, seed: int = 0,
                      processed_dir: str = "data/processed",
                      load_masks: bool = True) -> List[Reference]:
    """k defective TRAIN images, spread over defect classes (round-robin)."""
    d = _split_json(dataset, "train", processed_dir)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    by = _boxes_by_image(d)
    per_class = defaultdict(list)
    for im in d["images"]:
        if by.get(im["id"]):
            cls = by[im["id"]][0][1]
            per_class[cls].append(im)
    rng = np.random.default_rng(seed)
    for c in per_class:
        rng.shuffle(per_class[c])
    chosen, order = [], sorted(per_class)
    while len(chosen) < k and any(per_class[c] for c in order):
        for c in order:
            if per_class[c] and len(chosen) < k:
                chosen.append((per_class[c].pop(), c))
    forbidden = held_out_files(dataset, processed_dir)
    refs = []
    for im, c in chosen:
        fn = im["file_name"]
        if os.path.normcase(os.path.abspath(fn)) in forbidden:
            raise RuntimeError(f"reference {fn} is in a held-out split - refusing (guarantee rule)")
        boxes = [b for b, _ in by[im["id"]]]
        mask = None
        if load_masks:
            img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            mask = mask_for_image(dataset, fn, img.shape, boxes)
        refs.append(Reference(fn, im["id"], boxes, cats.get(c, "defect"), mask))
    return refs


def clean_train_images(dataset: str, processed_dir: str = "data/processed") -> List[str]:
    """Defect-free TRAIN images (backgrounds for generators). Empty for NEU / GC10 / PCB."""
    d = _split_json(dataset, "train", processed_dir)
    return sorted(im["file_name"] for im in d["images"] if im.get("n_boxes", 1) == 0)


def export_references(dataset: str, out_dir: str, k: int = 5, seed: int = 0,
                      processed_dir: str = "data/processed") -> List[dict]:
    """Write ref_XX.png + ref_XX_mask.png + references.json (provenance)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, r in enumerate(select_references(dataset, k, seed, processed_dir)):
        img = cv2.imread(r.file_name, cv2.IMREAD_COLOR)
        name = f"ref_{i:02d}"
        cv2.imwrite(os.path.join(out_dir, name + ".png"), img)
        cv2.imwrite(os.path.join(out_dir, name + "_mask.png"), r.mask)
        rows.append({"name": name, "source_file": r.file_name, "image_id": r.image_id,
                     "class": r.class_name, "boxes": r.boxes, "split": "train"})
    with open(os.path.join(out_dir, "references.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows
