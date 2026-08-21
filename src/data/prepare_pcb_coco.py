"""
prepare_pcb_coco.py
====================
Normalizes the PCB defects dataset (akhatova/pcb-defects on Kaggle) into a
single COCO json.

Two supported source formats (config.data.pcb_format):

  "coco" — you re-exported the dataset through Roboflow, which gives you
           train/valid/test folders each with an `_annotations.coco.json`.
           We just merge those three into one COCO file with paths made
           absolute.

  "voc"  — the raw Kaggle download, which ships as PASCAL-VOC XML
           (Annotations/<class>/*.xml + images/<class>/*.jpg), i.e. the same
           shape as NEU-DET. In that case we simply reuse the NEU converter
           logic (it's format-agnostic) with the PCB class list.
"""

import os
import json
import glob
from pathlib import Path
from typing import List

from .neu_xml_to_coco import convert_neu_to_coco

PCB_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


def _merge_roboflow_coco_splits(dataset_root: str, out_json: str) -> str:
    """Merge Roboflow-exported train/valid/test COCO jsons into one file,
    rewriting file_name to an absolute path so downstream code doesn't need
    to know which split an image came from."""
    split_jsons = glob.glob(os.path.join(dataset_root, "**", "_annotations.coco.json"), recursive=True)
    if not split_jsons:
        raise FileNotFoundError(
            f"No _annotations.coco.json found under {dataset_root}. "
            "If this isn't a Roboflow COCO export, set data.pcb_format: 'voc' in config.yaml."
        )

    merged_images, merged_annotations, categories = [], [], None
    img_id, ann_id = 1, 1

    for sj in sorted(split_jsons):
        split_dir = os.path.dirname(sj)
        with open(sj) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]

        id_remap = {}
        for im in data["images"]:
            old_id = im["id"]
            id_remap[old_id] = img_id
            merged_images.append({
                "id": img_id,
                "file_name": os.path.join(split_dir, im["file_name"]),
                "width": im["width"],
                "height": im["height"],
            })
            img_id += 1

        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = id_remap[ann["image_id"]]
            merged_annotations.append(new_ann)
            ann_id += 1

    coco = {"images": merged_images, "annotations": merged_annotations, "categories": categories}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"[prepare_pcb_coco] merged {len(split_jsons)} Roboflow splits -> "
          f"{len(merged_images)} images, {len(merged_annotations)} boxes")
    print(f"[prepare_pcb_coco] wrote {out_json}")
    return out_json


def prepare_pcb_coco(dataset_root: str, out_json: str, pcb_format: str = "coco",
                      classes: List[str] = None) -> str:
    if pcb_format == "coco":
        return _merge_roboflow_coco_splits(dataset_root, out_json)
    elif pcb_format == "voc":
        return convert_neu_to_coco(dataset_root, out_json, classes=classes or PCB_CLASSES)
    else:
        raise ValueError(f"Unknown pcb_format: {pcb_format!r} (expected 'coco' or 'voc')")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="coco", choices=["coco", "voc"])
    args = ap.parse_args()
    prepare_pcb_coco(args.root, args.out, pcb_format=args.format)
