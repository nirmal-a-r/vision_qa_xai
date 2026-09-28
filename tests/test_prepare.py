"""
Tests for the data stage: the DAGM and KolektorSDD2 converters on tiny fake
datasets, and the four-way split (the calibration block must never overlap the
detector's train or val slices).
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.prepare_datasets import convert_dagm, convert_kolektor, convert_mvtec  # noqa: E402
from src.data.build_coldstart_tree import build_tree                                  # noqa: E402
from src.data.splits_and_yolo import write_splits                      # noqa: E402


def _img(path, h, w, box=None):
    im = np.full((h, w), 120, np.uint8)
    m = np.zeros((h, w), np.uint8)
    if box:
        x1, y1, x2, y2 = box
        im[y1:y2, x1:x2] = 20
        m[y1:y2, x1:x2] = 255
    cv2.imwrite(path, im)
    return m


def test_dagm_converter():
    with tempfile.TemporaryDirectory() as d:
        for c in ("Class1", "Class10"):
            for s in ("Train", "Test"):
                os.makedirs(os.path.join(d, c, s, "Label"))
                for i in range(4):
                    box = (10, 20, 40, 60) if i == 0 else None
                    m = _img(os.path.join(d, c, s, f"{i:04d}.PNG"), 64, 64, box)
                    if box:
                        cv2.imwrite(os.path.join(d, c, s, "Label", f"{i:04d}_label.PNG"), m)
        os.makedirs(os.path.join(d, "DAGM_KaggleUpload", "Class1", "Train"))   # nested duplicate: ignored
        out = os.path.join(d, "dagm.json")
        convert_dagm(d, out)
        j = json.load(open(out))
    assert len(j["images"]) == 16 and len(j["annotations"]) == 4
    assert sorted(c["name"] for c in j["categories"]) == ["class1", "class10"]
    assert j["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert sum(im["n_boxes"] == 0 for im in j["images"]) == 12
    print("  16 images, 4 boxes from masks, 12 clean, 2 texture classes")


def test_kolektor_converter():
    with tempfile.TemporaryDirectory() as d:
        for s in ("train", "test"):
            os.makedirs(os.path.join(d, s))
            for i in range(3):
                box = (5, 5, 25, 15) if i == 1 else None
                m = _img(os.path.join(d, s, f"{s}{i}.png"), 40, 100, box)
                cv2.imwrite(os.path.join(d, s, f"{s}{i}_GT.png"), m)
        out = os.path.join(d, "k.json")
        convert_kolektor(d, out)
        j = json.load(open(out))
    assert len(j["images"]) == 6 and len(j["annotations"]) == 2
    print("  masks -> boxes, _GT files not treated as images")


def test_splits_are_disjoint_and_have_val():
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(400):
        n = 1 if rng.random() < 0.2 else 0
        images.append({"id": i + 1, "file_name": f"x{i}.png", "width": 10, "height": 10, "n_boxes": n})
        if n:
            anns.append({"id": len(anns) + 1, "image_id": i + 1, "category_id": 1,
                         "bbox": [1, 1, 3, 3], "area": 9, "iscrowd": 0})
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "kolektor_coco.json")
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "defect"}]}, open(cp, "w"))
        paths, _ = write_splits(cp, os.path.join(d, "splits"))
        ids = {s: {im["id"] for im in json.load(open(p))["images"]} for s, p in paths.items()}
    assert set(ids) == {"train", "val", "cal", "test"}
    names = list(ids)
    for a in range(4):
        for b in range(a + 1, 4):
            assert not ids[names[a]] & ids[names[b]], (names[a], names[b])
    assert sum(len(v) for v in ids.values()) == 400
    assert len(ids["val"]) > 0
    print(f"  train/val/cal/test = {[len(ids[s]) for s in ('train', 'val', 'cal', 'test')]}, all disjoint")


def test_mvtec_converter():
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "bottle")
        os.makedirs(os.path.join(c, "train", "good"))
        os.makedirs(os.path.join(c, "test", "good"))
        os.makedirs(os.path.join(c, "test", "crack"))
        os.makedirs(os.path.join(c, "ground_truth", "crack"))
        for i in range(3):
            _img(os.path.join(c, "train", "good", f"{i:03d}.png"), 32, 32)
        _img(os.path.join(c, "test", "good", "000.png"), 32, 32)
        m = _img(os.path.join(c, "test", "crack", "000.png"), 32, 32, (4, 4, 12, 20))
        cv2.imwrite(os.path.join(c, "ground_truth", "crack", "000_mask.png"), m)
        out = os.path.join(d, "mvtec.json")
        convert_mvtec(d, out)
        j = json.load(open(out))
    assert len(j["images"]) == 5 and len(j["annotations"]) == 1
    assert j["categories"][0]["name"] == "bottle"
    print("  good images clean, defect boxes from ground_truth masks")


def test_p2_tree_keeps_only_k_real_defects():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "yolo", "toy")
        for split in ("train", "val", "cal", "test"):
            os.makedirs(os.path.join(root, "images", split))
            os.makedirs(os.path.join(root, "labels", split))
            for i in range(10):
                _img(os.path.join(root, "images", split, f"{split}{i}.png"), 16, 16)
                with open(os.path.join(root, "labels", split, f"{split}{i}.txt"), "w") as f:
                    f.write("0 0.5 0.5 0.2 0.2\n" if i < 4 else "")
        with open(os.path.join(root, "data.yaml"), "w") as f:
            f.write("path: x\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n")
        syn = os.path.join(d, "syn")
        os.makedirs(os.path.join(syn, "images", "syn"))
        os.makedirs(os.path.join(syn, "labels", "syn"))
        for i in range(10):
            _img(os.path.join(syn, "images", "syn", f"s{i}.png"), 16, 16)
            with open(os.path.join(syn, "labels", "syn", f"s{i}.txt"), "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")
        out, c = build_tree("toy", "gen", k=2, yolo_dir=os.path.join(d, "yolo"), syn_root=syn,
                            out_root=os.path.join(d, "p2", "toy"))
        def n_def(split):
            ld = os.path.join(out, "labels", split)
            return sum(1 for f in os.listdir(ld) if open(os.path.join(ld, f)).read().strip()
                       and not f.startswith("syn_"))
        assert c["real_defects_in_train"] == 2 and n_def("train") == 2
        assert n_def("val") == 0, "no real defects may enter P2 validation"
        assert c["cal"] == 10 and c["test"] == 10 and n_def("cal") == 4
        # 10 synthetic: 5 held out for SPERC calibration, 5 used (4 train + 1 val)
        assert c["synthetic_holdout"] == 5 and c["synthetic_train_val"] == 5
        assert c["train"] == 6 + 2 + 4 and c["val"] == 6 + 1
        hold = set(os.listdir(os.path.join(c["holdout_root"], "images", "syn")))
        used = {f[len("syn_"):] for sp in ("train", "val")
                for f in os.listdir(os.path.join(out, "images", sp)) if f.startswith("syn_")}
        assert len(hold) == 5 and not (hold & used), "P2 calibration synthetics must be unseen"
    print("  train = 6 clean + 2 real + 4 synthetic; val has no real defects; cal/test unchanged;"
          " 5 synthetic held out for calibration, disjoint from training")


if __name__ == "__main__":
    from _runner import run
    run(globals())
