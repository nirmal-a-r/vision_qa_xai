"""
neu_xml_to_coco.py
===================
Converts the NEU-DET surface defect dataset (kaustubhdikshit/neu-surface-defect-
database on Kaggle) from its native PASCAL-VOC-style XML annotations into a
single COCO-format JSON.

The Kaggle mirror of NEU-DET is typically laid out as:

    neu-surface-defect-database/
        NEU-DET/
            train/
                images/<class>/*.jpg   (or images/*.jpg with class in filename)
                annotations/*.xml
            validation/
                images/...
                annotations/...

Layouts vary slightly between re-uploads of this dataset, so this script
walks the tree and matches *.xml files to same-stem image files anywhere
under the root, rather than assuming one fixed structure. Run this file
directly (or import `convert_neu_to_coco`) and point it at the dataset root;
it will find what it needs.

Classes (fixed, matches the standard NEU-DET label set):
    crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
"""

import os
import json
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _index_images(root: str) -> Dict[str, str]:
    """Map image filename stem -> full path, searched recursively."""
    index = {}
    for ext in IMAGE_EXTS:
        for p in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
            stem = Path(p).stem
            # Don't overwrite if a duplicate stem already found (first wins,
            # but warn so the user can check for real collisions).
            if stem not in index:
                index[stem] = p
    return index


def _parse_voc_xml(xml_path: str) -> Tuple[str, int, int, List[Tuple[str, float, float, float, float]]]:
    """Returns (image_filename_in_xml, width, height, [(label, xmin, ymin, xmax, ymax), ...])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    filename = filename_node.text.strip() if filename_node is not None and filename_node.text else Path(xml_path).stem

    size = root.find("size")
    width = int(size.find("width").text) if size is not None else 0
    height = int(size.find("height").text) if size is not None else 0

    boxes = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        if name_node is None or not name_node.text:
            continue
        label = name_node.text.strip().lower().replace(" ", "_")
        # normalize a couple of common spelling variants seen in re-uploads
        label = {
            "rolled-in_scale": "rolled-in_scale",
            "rolled_in_scale": "rolled-in_scale",
            "pitted-surface": "pitted_surface",
        }.get(label, label)

        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append((label, xmin, ymin, xmax, ymax))

    return filename, width, height, boxes


def convert_neu_to_coco(
    dataset_root: str,
    out_json: str,
    classes: List[str] = None,
    verbose: bool = True,
) -> str:
    """
    Walk `dataset_root` for every *.xml (VOC) annotation, match it to its
    image, and write a single merged COCO json to `out_json`.

    Returns the path to the written json.
    """
    classes = classes or NEU_CLASSES
    cat_name_to_id = {name: i + 1 for i, name in enumerate(classes)}  # COCO ids start at 1

    xml_files = glob.glob(os.path.join(dataset_root, "**", "*.xml"), recursive=True)
    if not xml_files:
        raise FileNotFoundError(
            f"No .xml annotation files found under {dataset_root}. "
            "Check the Kaggle dataset mount path / 'Add Data' slug."
        )
    image_index = _index_images(dataset_root)

    images, annotations = [], []
    img_id, ann_id = 1, 1
    matched, missing = 0, 0

    for xml_path in sorted(xml_files):
        filename, width, height, boxes = _parse_voc_xml(xml_path)
        stem = Path(filename).stem
        img_path = image_index.get(stem) or image_index.get(Path(xml_path).stem)

        if img_path is None:
            missing += 1
            continue
        matched += 1

        # Fill width/height from the actual file if XML didn't have them.
        if width == 0 or height == 0:
            from PIL import Image
            with Image.open(img_path) as im:
                width, height = im.size

        images.append({
            "id": img_id,
            "file_name": img_path,   # absolute path kept for a direct-copy step later
            "width": width,
            "height": height,
        })

        for label, xmin, ymin, xmax, ymax in boxes:
            if label not in cat_name_to_id:
                # unseen/unexpected label -> register it dynamically instead of dropping data
                cat_name_to_id[label] = len(cat_name_to_id) + 1
            w, h = max(0.0, xmax - xmin), max(0.0, ymax - ymin)
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_name_to_id[label],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],  # no polygon ground truth available — see README
            })
            ann_id += 1
        img_id += 1

    categories = [{"id": cid, "name": name, "supercategory": "defect"}
                  for name, cid in sorted(cat_name_to_id.items(), key=lambda kv: kv[1])]

    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)

    if verbose:
        print(f"[neu_xml_to_coco] matched {matched} images ({missing} xml with no image found)")
        print(f"[neu_xml_to_coco] {len(annotations)} boxes across {len(categories)} classes")
        print(f"[neu_xml_to_coco] wrote {out_json}")

    return out_json


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="NEU dataset root (Kaggle input dir)")
    ap.add_argument("--out", required=True, help="output COCO json path")
    args = ap.parse_args()
    convert_neu_to_coco(args.root, args.out)
