"""
splits_and_yolo.py
==================
Four-way splits + YOLO export.

Split policy is train / cal / test, where **cal** is a dedicated conformal
calibration set. This is not a stylistic choice: the CRC and LTT guarantees
require the calibration data to be exchangeable with the test data *and*
untouched by model fitting. Reusing the validation set that early stopping or
checkpoint selection looked at breaks exchangeability and silently voids the
guarantee. So the detector gets train (+ an inner val slice carved from it for
its own model selection), and calibration gets its own disjoint block that no
training decision ever sees.

Grouping: splits are made at the image level, stratified by the image's defect
class composition so that rare classes (GC10 crease, 74 boxes) do not land
entirely in one split.
"""

from __future__ import annotations

import os
import json
import shutil
import random
from collections import defaultdict, Counter

import yaml


def stratified_split(images, annotations, fractions=(0.60, 0.15, 0.25),
                     defect_fractions=(0.45, 0.30, 0.25), seed=42):
    """Split image ids into (train, cal, test), stratified by class signature.

    An image's stratum is its rarest present class (or "none" for defect-free),
    which keeps both rare defect classes and defect-free parts proportionally
    represented in every split - the latter matters because the triage policy is
    evaluated on exactly those.

    Defective and clean images are allocated on SEPARATE budgets, and that is the
    point of this function rather than an implementation detail.

    The conformal penalty is B / (n + 1) where n counts calibration points on
    which the loss is *defined*. Escape loss is only defined on images that
    contain a defect - a clean part cannot produce an escape and contributes an
    identically-zero row. So the guarantee's tightness is governed by the number
    of DEFECTIVE calibration images, not by calibration size.

    Splitting everything at one uniform 60/15/25 made that painfully concrete on
    the two rare-defect datasets. KolektorSDD2 has 356 defective images in 3337;
    a uniform split gave 500 calibration images of which only 53 were defective,
    so the finite-sample term alone spent 1.85% of the risk budget and alpha=0.01
    was unreachable however good the detector was. Magnetic-Tile had the same
    problem at 59.

    Giving defective images a calibration-heavier allocation (45/30/25 vs the
    clean 60/15/25) roughly doubles the usable calibration count on exactly those
    datasets and halves the penalty, at the cost of ~15% of the defective
    training pool. Clean images are moved at no cost at all, since they were
    never contributing to the risk estimate.
    """
    by_img = defaultdict(set)
    for a in annotations:
        by_img[a["image_id"]].add(a["category_id"])

    class_freq = Counter(a["category_id"] for a in annotations)
    strata = defaultdict(list)
    for im in images:
        cls = by_img.get(im["id"], set())
        key = "none" if not cls else min(cls, key=lambda c: class_freq[c])
        strata[key].append(im["id"])

    rng = random.Random(seed)
    train, cal, test = [], [], []
    for key, ids in sorted(strata.items(), key=lambda kv: str(kv[0])):
        # "none" is the defect-free stratum; everything else can realise a loss.
        f_tr, f_cal, _ = fractions if key == "none" else defect_fractions
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * f_tr))
        n_cal = int(round(n * f_cal))
        train += ids[:n_tr]
        cal += ids[n_tr:n_tr + n_cal]
        test += ids[n_tr + n_cal:]
    return set(train), set(cal), set(test)


def write_splits(coco_path, out_dir, fractions=(0.60, 0.15, 0.25),
                 defect_fractions=(0.45, 0.30, 0.25), seed=42):
    with open(coco_path) as f:
        d = json.load(f)
    name = os.path.basename(coco_path).replace("_coco.json", "")
    tr, cal, te = stratified_split(d["images"], d["annotations"], fractions,
                                   defect_fractions, seed)

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for split, ids in (("train", tr), ("cal", cal), ("test", te)):
        imgs = [im for im in d["images"] if im["id"] in ids]
        anns = [a for a in d["annotations"] if a["image_id"] in ids]
        p = os.path.join(out_dir, f"{name}_{split}.json")
        with open(p, "w") as f:
            json.dump({"images": imgs, "annotations": anns,
                       "categories": d["categories"]}, f)
        paths[split] = p
        neg = sum(1 for im in imgs if im.get("n_boxes", 1) == 0)
        pos = len(imgs) - neg
        extra = f"  alpha floor {1.0/(pos+1):.4f}" if split == "cal" else ""
        print(f"  {name:15s} {split:5s}: {len(imgs):5d} imgs ({pos:4d} defective, "
              f"{neg:4d} clean), {len(anns):5d} boxes{extra}")
    return paths, d["categories"]


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------

def coco_to_yolo(split_jsons, categories, out_root, dataset_name, link=True):
    """Write an ultralytics-style dataset tree and its data.yaml.

    Images are hard-linked rather than copied where the filesystem allows it -
    the five datasets total ~9 GB and three splits each would otherwise triple
    that for no benefit.
    """
    cat_ids = sorted(c["id"] for c in categories)
    cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    names = [next(c["name"] for c in categories if c["id"] == cid) for cid in cat_ids]

    root = os.path.join(out_root, dataset_name)
    for split in split_jsons:
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    for split, jp in split_jsons.items():
        with open(jp) as f:
            d = json.load(f)
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)

        for im in d["images"]:
            src = im["file_name"]
            stem = f"{im['id']:07d}"
            dst_img = os.path.join(root, "images", split, stem + os.path.splitext(src)[1])
            if not os.path.exists(dst_img):
                try:
                    if link:
                        os.link(src, dst_img)
                    else:
                        shutil.copyfile(src, dst_img)
                except OSError:
                    shutil.copyfile(src, dst_img)

            W, H = im["width"], im["height"]
            lines = []
            for a in anns.get(im["id"], []):
                x, y, w, h = a["bbox"]
                # YOLO wants normalised centre-x, centre-y, w, h
                lines.append(f"{cat_to_idx[a['category_id']]} "
                             f"{(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                             f"{w / W:.6f} {h / H:.6f}")
            # An empty label file is meaningful: it declares a genuine negative,
            # which is what the triage policy needs to learn to auto-accept.
            with open(os.path.join(root, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))

    data_yaml = os.path.join(root, "data.yaml")
    with open(data_yaml, "w") as f:
        yaml.safe_dump({
            "path": os.path.abspath(root),
            "train": "images/train",
            "val": "images/cal",     # ultralytics 'val' = our calibration block
            "test": "images/test",
            "names": {i: n for i, n in enumerate(names)},
        }, f, sort_keys=False)
    print(f"  -> {data_yaml}  ({len(names)} classes)")
    return data_yaml


if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--yolo_dir", default="data/yolo")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = {}
    for cp in sorted(glob.glob(os.path.join(a.processed_dir, "*_coco.json"))):
        name = os.path.basename(cp).replace("_coco.json", "")
        print(f"[{name}]")
        paths, cats = write_splits(cp, os.path.join(a.processed_dir, "splits"), seed=a.seed)
        out[name] = coco_to_yolo(paths, cats, a.yolo_dir, name)
    print("\ndata.yaml files:")
    for k, v in out.items():
        print(f"  {k}: {v}")
