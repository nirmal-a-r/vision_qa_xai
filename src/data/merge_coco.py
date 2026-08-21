"""
merge_coco.py
=============
Merges the PCB-defects COCO json and the NEU-DET COCO json into a single
COCO json with one unified category id space (config.data.unify_label_space).

Also copies every source image into a flat `images_merged_dir` so the final
dataset has stable, portable relative paths (source images can live in two
different Kaggle input mounts, which the model dataloader shouldn't need to
know about).

Also performs the train/val/test split (config.data.val_fraction /
test_fraction), grouped at the image level, and writes three sibling jsons:
    merged_coco.json          (everything — kept for reference)
    merged_coco_train.json
    merged_coco_val.json
    merged_coco_test.json
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def merge_and_split(
    pcb_json: str,
    neu_json: str,
    images_out_dir: str,
    merged_out_json: str,
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 42,
    copy_images: bool = True,
) -> dict:
    os.makedirs(images_out_dir, exist_ok=True)

    pcb = _load(pcb_json)
    neu = _load(neu_json)

    # ---- unify category space: name -> new id ----------------------------
    all_names = []
    for src in (pcb, neu):
        for c in src["categories"]:
            if c["name"] not in all_names:
                all_names.append(c["name"])
    name_to_new_id = {name: i + 1 for i, name in enumerate(sorted(all_names))}
    categories = [{"id": nid, "name": name, "supercategory": "defect"}
                  for name, nid in name_to_new_id.items()]

    images, annotations = [], []
    img_id, ann_id = 1, 1

    for src, src_tag in ((pcb, "pcb"), (neu, "neu")):
        old_cat_id_to_name = {c["id"]: c["name"] for c in src["categories"]}
        old_img_id_to_new = {}

        for im in src["images"]:
            src_path = im["file_name"]
            ext = Path(src_path).suffix or ".jpg"
            new_filename = f"{src_tag}_{im['id']:06d}{ext}"
            dst_path = os.path.join(images_out_dir, new_filename)

            if copy_images and os.path.exists(src_path) and not os.path.exists(dst_path):
                try:
                    shutil.copyfile(src_path, dst_path)
                except OSError:
                    pass  # tolerate missing/broken source paths, keep going

            old_img_id_to_new[im["id"]] = img_id
            images.append({
                "id": img_id,
                "file_name": new_filename,   # relative to images_out_dir
                "width": im["width"],
                "height": im["height"],
                "source": src_tag,
            })
            img_id += 1

        for ann in src["annotations"]:
            cat_name = old_cat_id_to_name.get(ann["category_id"])
            if cat_name is None:
                continue
            new_img_id = old_img_id_to_new.get(ann["image_id"])
            if new_img_id is None:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": new_img_id,
                "category_id": name_to_new_id[cat_name],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    os.makedirs(os.path.dirname(merged_out_json), exist_ok=True)
    with open(merged_out_json, "w") as f:
        json.dump(coco, f)

    # ---- split at image level --------------------------------------------
    rng = random.Random(seed)
    img_ids = [im["id"] for im in images]
    rng.shuffle(img_ids)
    n = len(img_ids)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_ids = set(img_ids[:n_test])
    val_ids = set(img_ids[n_test:n_test + n_val])
    train_ids = set(img_ids[n_test + n_val:])

    def _subset(id_set):
        imgs = [im for im in images if im["id"] in id_set]
        anns = [a for a in annotations if a["image_id"] in id_set]
        return {"images": imgs, "annotations": anns, "categories": categories}

    base, ext = os.path.splitext(merged_out_json)
    split_paths = {}
    for split_name, id_set in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        p = f"{base}_{split_name}{ext}"
        with open(p, "w") as f:
            json.dump(_subset(id_set), f)
        split_paths[split_name] = p

    print(f"[merge_coco] {len(images)} images, {len(annotations)} boxes, "
          f"{len(categories)} unified classes: {list(name_to_new_id)}")
    print(f"[merge_coco] split -> train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
    for k, v in split_paths.items():
        print(f"[merge_coco] wrote {v}")

    return {"merged": merged_out_json, **split_paths, "categories": categories}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcb_json", required=True)
    ap.add_argument("--neu_json", required=True)
    ap.add_argument("--images_out", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merge_and_split(args.pcb_json, args.neu_json, args.images_out, args.out)
