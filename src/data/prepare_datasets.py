"""
prepare_datasets.py
===================
One converter for every source dataset -> per-dataset COCO JSON.

Six datasets (plus optional MVTec AD), two annotation styles:

  VOC XML boxes      NEU-DET (steel surface, 6 cls), PCB-DEFECTS (PCB, 6 cls),
                     GC10-DET (galvanized steel sheet, 10 cls)
  binary masks       MAGNETIC-TILE (magnetic tile, 5 defect cls + defect-free),
                     KOLEKTORSDD2 (electrical commutator, 1 cls),
                     DAGM 2007 (10 synthetic textures, weak ellipse masks),
                     MVTec AD (15 categories; optional bridge dataset)

The two mask datasets earn their place for a specific reason rather than to pad
the dataset count: NEU, PCB and GC10 contain a defect in *every* image, so a
policy that auto-accepts a part can never be evaluated honestly on them. Only
MAGNETIC-TILE (952 defect-free tiles), KOLEKTORSDD2 (majority clean) and DAGM
(about 87% clean, artificial textures) contain genuine negatives, and without negatives the escape-rate / review-load
trade-off that this whole method is about is not measurable.

Everything is emitted with absolute image paths so a later merge step can pull
from several roots without copying.
"""

from __future__ import annotations

import os
import re
import glob
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")


# ---------------------------------------------------------------------------
# Label vocabularies
# ---------------------------------------------------------------------------

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"]

PCB_CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur",
               "spurious_copper"]

# GC10-DET ships Pinyin class names. Translated per the dataset paper
# (Lv et al., "Deep Metallic Surface Defect Detection", Sensors 2020) so the
# merged label space is readable and comparable across datasets.
GC10_RENAME = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "weld_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",
    # Real label noise in the published files: 131 boxes carry a trailing "d"
    # and 12 carry the clean spelling. Folding them together recovers the class;
    # leaving them split would silently create an 11th class with 12 examples.
    "10_yaozhed": "waist_folding",
}

MAGNETIC_CLASSES = ["blowhole", "break", "crack", "fray", "uneven"]


def _index_images(root: str) -> dict:
    """stem -> path for every image under root."""
    index = {}
    for ext in IMAGE_EXTS:
        for p in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
            index.setdefault(Path(p).stem, p)
    return index


def _empty_coco():
    return {"images": [], "annotations": [], "categories": []}


def _finish(images, annotations, cat_to_id, out_json, name, verbose=True):
    categories = [{"id": cid, "name": n, "supercategory": "defect"}
                  for n, cid in sorted(cat_to_id.items(), key=lambda kv: kv[1])]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)
    if verbose:
        n_neg = sum(1 for im in images if im.get("n_boxes", 1) == 0)
        print(f"[{name}] {len(images)} images ({n_neg} defect-free), "
              f"{len(annotations)} boxes, {len(categories)} classes -> {out_json}")
    return out_json


# ---------------------------------------------------------------------------
# VOC XML sources
# ---------------------------------------------------------------------------

def _parse_voc(xml_path: str):
    root = ET.parse(xml_path).getroot()
    fn = root.find("filename")
    filename = fn.text.strip() if fn is not None and fn.text else Path(xml_path).stem

    size = root.find("size")
    w = int(float(size.find("width").text)) if size is not None else 0
    h = int(float(size.find("height").text)) if size is not None else 0

    boxes = []
    for obj in root.findall("object"):
        nm = obj.find("name")
        bb = obj.find("bndbox")
        if nm is None or not nm.text or bb is None:
            continue
        try:
            coords = [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (AttributeError, TypeError, ValueError):
            continue
        boxes.append((nm.text.strip().lower().replace(" ", "_"), *coords))
    return filename, w, h, boxes


def convert_voc(dataset_root: str, out_json: str, name: str,
                rename: dict = None, min_box: float = 2.0):
    """VOC-XML tree -> COCO. Works for NEU, PCB and GC10 unchanged."""
    rename = rename or {}
    xmls = sorted(glob.glob(os.path.join(dataset_root, "**", "*.xml"), recursive=True))
    if not xmls:
        raise FileNotFoundError(f"no .xml under {dataset_root}")
    index = _index_images(dataset_root)

    images, annotations, cat_to_id = [], [], {}
    img_id = ann_id = 1
    unmatched, dropped_boxes, unknown_labels = 0, 0, {}

    for xp in xmls:
        filename, w, h, boxes = _parse_voc(xp)
        stem = Path(filename).stem
        ipath = index.get(stem) or index.get(Path(xp).stem)
        if ipath is None:
            unmatched += 1
            continue
        if w <= 0 or h <= 0:
            im = cv2.imread(ipath)
            if im is None:
                unmatched += 1
                continue
            h, w = im.shape[:2]

        kept = 0
        for label, x1, y1, x2, y2 in boxes:
            label = rename.get(label, label)
            # A single GC10 box is labelled "d" -- a truncated annotation, not a
            # class. Count and drop rather than minting a phantom category.
            if len(label) <= 1:
                unknown_labels[label] = unknown_labels.get(label, 0) + 1
                continue
            x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
            y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
            bw, bh = x2 - x1, y2 - y1
            if bw < min_box or bh < min_box:
                dropped_boxes += 1
                continue
            cat_to_id.setdefault(label, len(cat_to_id) + 1)
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_to_id[label],
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0, "segmentation": [],
            })
            ann_id += 1
            kept += 1

        images.append({"id": img_id, "file_name": ipath, "width": int(w),
                       "height": int(h), "n_boxes": kept, "source": name})
        img_id += 1

    if unmatched:
        print(f"[{name}] warning: {unmatched} XML had no matching image")
    if dropped_boxes:
        print(f"[{name}] dropped {dropped_boxes} degenerate boxes (<{min_box}px)")
    if unknown_labels:
        print(f"[{name}] dropped malformed labels: {unknown_labels}")
    return _finish(images, annotations, cat_to_id, out_json, name)


# ---------------------------------------------------------------------------
# Mask sources
# ---------------------------------------------------------------------------

def boxes_from_mask(mask: np.ndarray, min_area: int = 20, thresh: int = 0):
    """Connected components of a binary mask -> xyxy boxes.

    Masks are anti-aliased in these datasets (values like 1, 9, 13 at defect
    edges), so anything above `thresh` counts as foreground rather than testing
    for a specific label value.
    """
    binary = (mask > thresh).astype(np.uint8)
    if binary.sum() == 0:
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    for i in range(1, n):                      # 0 is background
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append((float(x), float(y), float(x + w), float(y + h)))
    return out


def convert_magnetic_tile(dataset_root: str, out_json: str, name: str = "magnetic_tile"):
    """MT_<Class>/Imgs/<stem>.jpg + <stem>.png mask. MT_Free is defect-free."""
    images, annotations, cat_to_id = [], [], {}
    img_id = ann_id = 1

    for cls_dir in sorted(glob.glob(os.path.join(dataset_root, "MT_*"))):
        raw = os.path.basename(cls_dir).replace("MT_", "").lower()
        is_free = raw == "free"
        for jpg in sorted(glob.glob(os.path.join(cls_dir, "Imgs", "*.jpg"))):
            im = cv2.imread(jpg)
            if im is None:
                continue
            h, w = im.shape[:2]
            kept = 0
            if not is_free:
                png = os.path.splitext(jpg)[0] + ".png"
                if os.path.exists(png):
                    mask = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        cat_to_id.setdefault(raw, len(cat_to_id) + 1)
                        for x1, y1, x2, y2 in boxes_from_mask(mask):
                            annotations.append({
                                "id": ann_id, "image_id": img_id,
                                "category_id": cat_to_id[raw],
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1),
                                "iscrowd": 0, "segmentation": [],
                            })
                            ann_id += 1
                            kept += 1
            images.append({"id": img_id, "file_name": jpg, "width": w, "height": h,
                           "n_boxes": kept, "source": name})
            img_id += 1

    return _finish(images, annotations, cat_to_id, out_json, name)


def convert_kolektor(dataset_root: str, out_json: str, name: str = "kolektor_sdd2"):
    """<split>/<id>.png image + <id>_GT.png mask. Single defect class."""
    images, annotations, cat_to_id = [], [], {"defect": 1}
    img_id = ann_id = 1

    for split in ("train", "test"):
        sd = os.path.join(dataset_root, split)
        if not os.path.isdir(sd):
            continue
        for png in sorted(glob.glob(os.path.join(sd, "*.png"))):
            if png.endswith("_GT.png"):
                continue
            im = cv2.imread(png)
            if im is None:
                continue
            h, w = im.shape[:2]
            kept = 0
            gt = os.path.splitext(png)[0] + "_GT.png"
            if os.path.exists(gt):
                mask = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    for x1, y1, x2, y2 in boxes_from_mask(mask):
                        annotations.append({
                            "id": ann_id, "image_id": img_id, "category_id": 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": (x2 - x1) * (y2 - y1),
                            "iscrowd": 0, "segmentation": [],
                        })
                        ann_id += 1
                        kept += 1
            images.append({"id": img_id, "file_name": png, "width": w, "height": h,
                           "n_boxes": kept, "source": name, "split_hint": split})
            img_id += 1

    return _finish(images, annotations, cat_to_id, out_json, name)


def convert_dagm(dataset_root: str, out_json: str, name: str = "dagm"):
    """DAGM 2007 (Kaggle layout): Class<k>/{Train,Test}/<stem>.PNG and
    Class<k>/{Train,Test}/Label/<stem>_label.PNG (weak ellipse masks).

    Each of the 10 classes is a different texture with its own defect type, so
    the class number is the category. Most images are defect-free, which makes
    DAGM a clean-part dataset like KolektorSDD2 - but its textures and defects
    are artificially generated, and the paper must say so. Only the top-level
    Class* folders are read (a nested duplicate copy is ignored). Train and Test
    are pooled and re-split by splits_and_yolo.py like every other source.
    """
    images, annotations, cat_to_id = [], [], {}
    img_id = ann_id = 1

    def _k(p):
        digits = "".join(ch for ch in os.path.basename(p) if ch.isdigit())
        return int(digits) if digits else 0

    class_dirs = [p for p in glob.glob(os.path.join(dataset_root, "*"))
                  if os.path.isdir(p) and os.path.basename(p).lower().startswith("class")]
    for cls_dir in sorted(class_dirs, key=_k):
        cls = f"class{_k(cls_dir)}"
        for split in ("Train", "Test"):
            sd = os.path.join(cls_dir, split)
            if not os.path.isdir(sd):
                continue
            label_dir = os.path.join(sd, "Label")
            for p in sorted(os.listdir(sd)):
                if not p.lower().endswith(".png"):
                    continue
                ipath = os.path.join(sd, p)
                im = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                h, w = im.shape[:2]
                kept = 0
                stem = os.path.splitext(p)[0]
                lp = None
                for cand in (stem + "_label.PNG", stem + "_label.png"):
                    if os.path.exists(os.path.join(label_dir, cand)):
                        lp = os.path.join(label_dir, cand)
                        break
                if lp is not None:
                    mask = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        if mask.shape[:2] != (h, w):
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        cat_to_id.setdefault(cls, len(cat_to_id) + 1)
                        for x1, y1, x2, y2 in boxes_from_mask(mask):
                            annotations.append({
                                "id": ann_id, "image_id": img_id,
                                "category_id": cat_to_id[cls],
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1),
                                "iscrowd": 0, "segmentation": [],
                            })
                            ann_id += 1
                            kept += 1
                images.append({"id": img_id, "file_name": ipath, "width": w, "height": h,
                               "n_boxes": kept, "source": name, "texture": cls,
                               "split_hint": split.lower()})
                img_id += 1

    return _finish(images, annotations, cat_to_id, out_json, name)


def convert_mvtec(dataset_root: str, out_json: str, name: str = "mvtec"):
    """MVTec AD: <category>/{train/good, test/<defect|good>}/<stem>.png and
    <category>/ground_truth/<defect>/<stem>_mask.png.

    The bridge dataset of the plan (Section 9): SPI's native setting is one score
    per image, and MVTec AD is the benchmark reviewers know. Defective images
    exist only in MVTec's test folders; here all images are pooled and re-split,
    so the detector does see some defects in training. That departs from MVTec's
    unsupervised protocol on purpose - this project needs a detector - and the
    paper must state it. The object category is the class.
    """
    images, annotations, cat_to_id = [], [], {}
    img_id = ann_id = 1
    for cat_dir in sorted(p for p in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(p)):
        cat = os.path.basename(cat_dir)
        for split in ("train", "test"):
            for sub_dir in sorted(glob.glob(os.path.join(cat_dir, split, "*"))):
                defect = os.path.basename(sub_dir)
                for png in sorted(glob.glob(os.path.join(sub_dir, "*.png"))):
                    im = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        continue
                    h, w = im.shape[:2]
                    kept = 0
                    if defect != "good":
                        stem = os.path.splitext(os.path.basename(png))[0]
                        mp = os.path.join(cat_dir, "ground_truth", defect, stem + "_mask.png")
                        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) if os.path.exists(mp) else None
                        if mask is not None:
                            if mask.shape[:2] != (h, w):
                                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            cat_to_id.setdefault(cat, len(cat_to_id) + 1)
                            for x1, y1, x2, y2 in boxes_from_mask(mask):
                                annotations.append({
                                    "id": ann_id, "image_id": img_id,
                                    "category_id": cat_to_id[cat],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "area": (x2 - x1) * (y2 - y1),
                                    "iscrowd": 0, "segmentation": [],
                                })
                                ann_id += 1
                                kept += 1
                    images.append({"id": img_id, "file_name": png, "width": w, "height": h,
                                   "n_boxes": kept, "source": name, "category": cat,
                                   "defect_type": defect, "split_hint": split})
                    img_id += 1
    return _finish(images, annotations, cat_to_id, out_json, name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

REGISTRY = {
    "neu":           dict(kind="voc", sub="NEU-DET",                  rename=None),
    "pcb":           dict(kind="voc", sub="PCB-DEFECTS/PCB_DATASET",  rename=None),
    "gc10":          dict(kind="voc", sub="GC10-DET",                 rename=GC10_RENAME),
    "magnetic_tile": dict(kind="magnetic", sub="MAGNETIC-TILE",       rename=None),
    "kolektor":      dict(kind="kolektor", sub="KOLEKTORSDD2",        rename=None),
    "dagm":          dict(kind="dagm", sub="DAGM2007/DAGM_KaggleUpload", rename=None),
    "mvtec":         dict(kind="mvtec", sub="MVTEC-AD",                  rename=None),   # optional
}


def prepare_all(raw_dir: str, out_dir: str, only=None):
    results = {}
    for key, spec in REGISTRY.items():
        if only and key not in only:
            continue
        root = os.path.join(raw_dir, spec["sub"])
        if not os.path.isdir(root):
            print(f"[{key}] skipped: {root} not present")
            continue
        out_json = os.path.join(out_dir, f"{key}_coco.json")
        try:
            if spec["kind"] == "voc":
                results[key] = convert_voc(root, out_json, key, rename=spec["rename"])
            elif spec["kind"] == "magnetic":
                results[key] = convert_magnetic_tile(root, out_json, key)
            elif spec["kind"] == "kolektor":
                results[key] = convert_kolektor(root, out_json, key)
            elif spec["kind"] == "dagm":
                results[key] = convert_dagm(root, out_json, key)
            elif spec["kind"] == "mvtec":
                results[key] = convert_mvtec(root, out_json, key)
        except Exception as e:
            print(f"[{key}] FAILED: {type(e).__name__}: {e}")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--only", default=None, help="comma-separated subset of "
                                                 + ",".join(REGISTRY))
    a = ap.parse_args()
    prepare_all(a.raw_dir, a.out_dir,
                only=set(a.only.split(",")) if a.only else None)
