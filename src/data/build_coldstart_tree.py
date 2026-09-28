"""
build_coldstart_tree.py
=======================
Protocol P2 ("full cold start") training tree: the detector itself is trained as
a newly commissioned line would train it.

    train  = ALL clean train images + only k real defective train images
             + the synthetic defect images of one generator source
    val    = clean val images + a held-out 20% of the synthetic images
             (no further real defects: a new line would not have them)
    cal    = identical to the standard tree (same files)
    test   = identical to the standard tree (same files)

A disjoint share of the synthetic images (``syn_holdout_fraction``, default one
half) is NEVER given to the P2 detector: it is written to
data/synthetic/<dataset>/<source>_p2holdout/ and is what SPERC calibrates with.
Scoring synthetic images the detector was trained on would make them look easier
than fresh ones; Tier H would still hold (SPI Corollary 3.6) but Tier N's eps
would lose its plain meaning (project document, Section 8.5, assumption 3).

Written to data/yolo_p2_<source>/<dataset>/ so train_baselines.py can train it
with --yolo_dir data/yolo_p2_<source> --encoding p2_<source>, and every later
stage finds it by the same convention (data/yolo_<encoding>).

    python -m src.data.build_coldstart_tree --dataset kolektor --source training_free --k 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil

import yaml

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _images(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*")) if p.lower().endswith(IMG_EXT))


def _is_defective(label_path):
    if not (os.path.exists(label_path) and os.path.getsize(label_path) > 0):
        return False
    with open(label_path) as f:
        return f.read().strip() != ""


def build_tree(dataset, source, k=5, yolo_dir="data/yolo", syn_root=None, out_root=None,
               syn_val_fraction=0.2, seed=0, syn_holdout_fraction=0.5, holdout_root=None):
    src_root = os.path.join(yolo_dir, dataset)
    syn_root = syn_root or os.path.join("data", "synthetic", dataset, source)
    out_root = out_root or os.path.join(f"data/yolo_p2_{source}", dataset)
    if not os.path.exists(os.path.join(src_root, "data.yaml")):
        raise FileNotFoundError(f"standard tree missing: {src_root}/data.yaml")
    syn_imgs = _images(os.path.join(syn_root, "images", "syn"))
    if not syn_imgs:
        raise FileNotFoundError(f"no synthetic images in {syn_root}/images/syn")
    rng = random.Random(seed)
    counts = {}

    def put(split, img, lbl_src, prefix=""):
        di, dl = (os.path.join(out_root, t, split) for t in ("images", "labels"))
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        name = prefix + os.path.basename(img)
        _link(img, os.path.join(di, name))
        dst_lbl = os.path.join(dl, os.path.splitext(name)[0] + ".txt")
        if lbl_src and os.path.exists(lbl_src):
            shutil.copyfile(lbl_src, dst_lbl)
        else:
            open(dst_lbl, "w").close()          # empty label = genuine negative
        counts[split] = counts.get(split, 0) + 1

    def lbl(split, img):
        return os.path.join(src_root, "labels", split, os.path.splitext(os.path.basename(img))[0] + ".txt")

    # train: clean + k real defects
    train = _images(os.path.join(src_root, "images", "train"))
    defective = [p for p in train if _is_defective(lbl("train", p))]
    clean = [p for p in train if p not in set(defective)]
    keep = rng.sample(defective, min(k, len(defective)))
    for p in clean + keep:
        put("train", p, lbl("train", p))
    # synthetic: a disjoint holdout for SPERC calibration, the rest split train / val
    rng.shuffle(syn_imgs)
    n_hold = int(round(syn_holdout_fraction * len(syn_imgs)))
    hold, used = syn_imgs[:n_hold], syn_imgs[n_hold:]
    syn_lbl = lambda p: os.path.join(syn_root, "labels", "syn",                      # noqa: E731
                                     os.path.splitext(os.path.basename(p))[0] + ".txt")
    n_val = int(round(syn_val_fraction * len(used)))
    for i, p in enumerate(used):
        split = "val" if i < n_val else "train"
        put(split, p, syn_lbl(p), prefix="syn_")
    holdout_root = holdout_root or os.path.join(os.path.dirname(syn_root.rstrip("/\\")),
                                                os.path.basename(syn_root.rstrip("/\\")) + "_p2holdout")
    if hold:
        for t in ("images", "labels"):
            os.makedirs(os.path.join(holdout_root, t, "syn"), exist_ok=True)
        for p in hold:
            _link(p, os.path.join(holdout_root, "images", "syn", os.path.basename(p)))
            if os.path.exists(syn_lbl(p)):
                shutil.copyfile(syn_lbl(p), os.path.join(holdout_root, "labels", "syn",
                                                         os.path.basename(syn_lbl(p))))
        with open(os.path.join(holdout_root, "manifest.json"), "w") as f:
            json.dump({"from": os.path.abspath(syn_root), "n": len(hold),
                       "role": "P2 synthetic calibration set, never used to train the P2 detector"}, f, indent=1)
    # val: clean val images only (plus the synthetic ones above)
    for p in _images(os.path.join(src_root, "images", "val")):
        if not _is_defective(lbl("val", p)):
            put("val", p, None)
    # cal / test: identical files
    for split in ("cal", "test"):
        for p in _images(os.path.join(src_root, "images", split)):
            put(split, p, lbl(split, p))

    cfg = yaml.safe_load(open(os.path.join(src_root, "data.yaml")))
    cfg["path"] = os.path.abspath(out_root)
    cfg.update(train="images/train", val="images/val", test="images/test")
    with open(os.path.join(out_root, "data.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    counts.update(real_defects_in_train=len(keep), synthetic_train_val=len(used),
                  synthetic_holdout=len(hold), holdout_root=holdout_root if hold else None)
    return out_root, counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", required=True, help="synthetic source name under data/synthetic/<dataset>/")
    ap.add_argument("--k", type=int, default=5, help="real defective train images to keep")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, c = build_tree(a.dataset, a.source, a.k, seed=a.seed)
    print(f"wrote {root}: {c}")
