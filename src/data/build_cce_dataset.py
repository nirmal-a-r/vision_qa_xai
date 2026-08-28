"""
build_cce_dataset.py
====================
Materialise CCE-encoded copies of the YOLO dataset trees.

Encoding is done offline rather than in the dataloader for three reasons:

  1. **Determinism.** CCE is a fixed function of the input, not an augmentation.
     Computing it once means every epoch, every seed, and every model sees
     byte-identical inputs, so an ablation isolates the encoding rather than
     the interaction between the encoding and the augmentation RNG.

  2. **Fairness of the ablation.** Doing it online would add CPU work to the
     CCE arm only, which on a dataloader-bound run makes the CCE arm look
     slower for reasons that have nothing to do with the method. Offline, both
     arms have identical input pipelines.

  3. **Deployment honesty.** In a real line the encoding runs once per frame on
     the acquisition side, which is exactly what materialising it models. The
     reported inference latency then reflects what the customer would see.

Labels are hard-linked, never rewritten: CCE is photometric only and does not
move a single pixel, so the boxes are bit-identical by construction. Copying
them would create an opportunity for them to drift out of sync; linking removes
that class of bug entirely.
"""

from __future__ import annotations

import os
import sys
import glob
import shutil
import argparse

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.photometric import complementary_channels


def encode_tree(src_root: str, dst_root: str, clip_limit=2.0, tile_grid=8,
                sigma_frac=0.02, force=False):
    splits = [d for d in ("train", "cal", "test")
              if os.path.isdir(os.path.join(src_root, "images", d))]
    n_done = n_skip = n_fail = 0

    for split in splits:
        src_img = os.path.join(src_root, "images", split)
        dst_img = os.path.join(dst_root, "images", split)
        src_lbl = os.path.join(src_root, "labels", split)
        dst_lbl = os.path.join(dst_root, "labels", split)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)

        for p in sorted(glob.glob(os.path.join(src_img, "*"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            # Always write PNG: CCE packs three decorrelated channels, and JPEG's
            # chroma subsampling would quantise ch1/ch2 as if they were colour
            # difference signals, destroying part of what the encoding just added.
            out_p = os.path.join(dst_img, stem + ".png")
            if os.path.exists(out_p) and not force:
                n_skip += 1
            else:
                im = cv2.imread(p, cv2.IMREAD_COLOR)
                if im is None:
                    n_fail += 1
                    continue
                enc = complementary_channels(cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                                             clip_limit, tile_grid, sigma_frac)
                cv2.imwrite(out_p, cv2.cvtColor(enc, cv2.COLOR_RGB2BGR))
                n_done += 1

            src_txt = os.path.join(src_lbl, stem + ".txt")
            dst_txt = os.path.join(dst_lbl, stem + ".txt")
            if os.path.exists(src_txt) and not os.path.exists(dst_txt):
                try:
                    os.link(src_txt, dst_txt)
                except OSError:
                    shutil.copyfile(src_txt, dst_txt)

    with open(os.path.join(src_root, "data.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = os.path.abspath(dst_root)
    with open(os.path.join(dst_root, "data.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"  encoded {n_done}, reused {n_skip}, failed {n_fail} -> {dst_root}")
    return os.path.join(dst_root, "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_dir", default="data/yolo")
    ap.add_argument("--out_dir", default="data/yolo_cce")
    ap.add_argument("--datasets", default="neu,magnetic_tile,kolektor,gc10,pcb")
    ap.add_argument("--clip_limit", type=float, default=2.0)
    ap.add_argument("--tile_grid", type=int, default=8)
    ap.add_argument("--sigma_frac", type=float, default=0.02)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for name in a.datasets.split(","):
        name = name.strip()
        src = os.path.join(a.yolo_dir, name)
        if not os.path.isdir(src):
            print(f"[{name}] skipped: {src} missing")
            continue
        print(f"[{name}]")
        encode_tree(src, os.path.join(a.out_dir, name),
                    a.clip_limit, a.tile_grid, a.sigma_frac, a.force)


if __name__ == "__main__":
    main()
