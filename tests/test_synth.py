"""
Tests for src/synth/composite.py - labels produced from generator masks.
"""

import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synth.composite import SPLIT, boxes_from_mask, build_synthetic_set, paste_defect  # noqa: E402


def test_boxes_from_mask():
    m = np.zeros((50, 80), np.uint8)
    m[10:20, 5:15] = 255
    m[30:40, 60:70] = 255
    assert sorted(boxes_from_mask(m)) == [(5, 10, 15, 20), (60, 30, 70, 40)]
    print("  one box per connected component")


def test_paste_defect_places_mask_correctly():
    bg = np.full((100, 100, 3), 128, np.uint8)
    patch = np.zeros((10, 10, 3), np.uint8)
    pm = np.full((10, 10), 255, np.uint8)
    out, full = paste_defect(bg, patch, pm, (30, 40), feather=0)
    assert boxes_from_mask(full) == [(30, 40, 40, 50)]
    assert out[45, 35].sum() == 0 and out[0, 0].sum() == 3 * 128
    print("  patch and mask land at the requested position")


def test_build_synthetic_set_full_and_paste():
    with tempfile.TemporaryDirectory() as d:
        gen, clean, out = (os.path.join(d, x) for x in ("gen", "clean", "out"))
        for p in (gen, clean):
            os.makedirs(p)
        img = np.full((64, 64, 3), 100, np.uint8)
        msk = np.zeros((64, 64), np.uint8)
        msk[20:30, 20:40] = 255
        cv2.imwrite(os.path.join(gen, "a.png"), img)
        cv2.imwrite(os.path.join(gen, "a_mask.png"), msk)
        cv2.imwrite(os.path.join(clean, "c.png"), np.full((128, 128, 3), 50, np.uint8))

        r = build_synthetic_set(gen, out, mode="full")
        assert r.written == 1
        lbl = open(os.path.join(out, "labels", SPLIT, "syn_00000_00.txt")).read().split()
        assert lbl[0] == "0" and abs(float(lbl[1]) - 30 / 64) < 1e-6 and abs(float(lbl[3]) - 20 / 64) < 1e-6

        r2 = build_synthetic_set(gen, out + "_p", mode="paste", clean_dir=clean, n_per_pair=3, seed=1)
        assert r2.written == 3
        man = json.load(open(os.path.join(out + "_p", "manifest.json")))
        assert all(e["n_boxes"] == 1 for e in man)
    print("  full and paste modes write images, YOLO labels and a manifest")


if __name__ == "__main__":
    from _runner import run
    run(globals())


# ------------------------------------------------------------------ generators
def _toy_processed(d, n_def=6, n_clean=6):
    """A tiny processed dataset: data/processed/splits/toy_{train,val,cal,test}.json."""
    img_dir = os.path.join(d, "imgs")
    os.makedirs(img_dir)
    splits = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "scratch"}]}
              for s in ("train", "val", "cal", "test")}
    iid = aid = 1
    for s in splits:
        for i in range(n_def + n_clean):
            p = os.path.join(img_dir, f"{s}_{i}.png")
            im = np.full((96, 128, 3), 140, np.uint8)
            nb = 1 if i < n_def else 0
            if nb:
                im[30:50, 40:70] = 30
                splits[s]["annotations"].append({"id": aid, "image_id": iid, "category_id": 1,
                                                 "bbox": [40, 30, 30, 20]})
                aid += 1
            cv2.imwrite(p, im)
            splits[s]["images"].append({"id": iid, "file_name": p, "width": 128, "height": 96, "n_boxes": nb})
            iid += 1
    sd = os.path.join(d, "processed", "splits")
    os.makedirs(sd)
    for s, j in splits.items():
        json.dump(j, open(os.path.join(sd, f"toy_{s}.json"), "w"))
    return os.path.join(d, "processed")


def test_references_come_from_train_only():
    from src.synth.references import select_references, clean_train_images
    with tempfile.TemporaryDirectory() as d:
        proc = _toy_processed(d)
        refs = select_references("toy", k=4, seed=0, processed_dir=proc)
        assert len(refs) == 4 and all("/train_" in r.file_name.replace("\\", "/") for r in refs)
        assert all(r.mask is not None and r.mask.max() == 255 for r in refs)   # boxes -> filled mask
        assert all("/train_" in p.replace("\\", "/") for p in clean_train_images("toy", proc))
        # a reference that also sits in a held-out split must be refused
        j = json.load(open(os.path.join(proc, "splits", "toy_cal.json")))
        j["images"][0]["file_name"] = refs[0].file_name
        json.dump(j, open(os.path.join(proc, "splits", "toy_cal.json"), "w"))
        try:
            select_references("toy", k=4, seed=0, processed_dir=proc)
            raise AssertionError("a held-out file was accepted as a reference")
        except RuntimeError:
            pass
    print("  references and backgrounds from the train split only; held-out overlap refused")


def test_copy_paste_generator_and_collector():
    from src.synth.generators import CopyPasteGenerator, audit_sheet, list_pairs
    from src.synth.external import collect_pairs, export_mvtec_layout
    with tempfile.TemporaryDirectory() as d:
        proc = _toy_processed(d)
        out = os.path.join(d, "gen")
        n = CopyPasteGenerator("toy", k=3, seed=0, processed_dir=proc).generate(12, out, verbose=False)
        assert n == 12 and len(list_pairs(out)) == 12
        for s in list_pairs(out):
            m = cv2.imread(os.path.join(out, s + "_mask.png"), cv2.IMREAD_GRAYSCALE)
            assert m is not None and (m > 0).sum() > 20
        assert audit_sheet(out) and json.load(open(os.path.join(out, "generator.json")))["split"] == "train"
        r = build_synthetic_set(out, os.path.join(d, "syn"), mode="full")
        assert r.written == 12
        # AnomalyDiffusion-style output layout: image/ + mask/
        ext = os.path.join(d, "ad", "toy", "scratch")
        for sub in ("image", "mask"):
            os.makedirs(os.path.join(ext, sub))
        for i, s in enumerate(list_pairs(out)[:5]):
            cv2.imwrite(os.path.join(ext, "image", f"{i}.jpg"), cv2.imread(os.path.join(out, s + ".png")))
            cv2.imwrite(os.path.join(ext, "mask", f"{i}.jpg"), cv2.imread(os.path.join(out, s + "_mask.png")))
        assert collect_pairs(os.path.join(d, "ad"), os.path.join(d, "col"), "anomalydiffusion") == 5
        root = export_mvtec_layout("toy", os.path.join(d, "mv"), k=3, processed_dir=proc)
        prov = json.load(open(os.path.join(root, "provenance.json")))
        assert prov["split"] == "train" and all("/train_" in p.replace("\\", "/") for p in prov["references"])
    print("  copy-paste pairs with masks; collector reads image/ + mask/; MVTec export holds train refs only")
