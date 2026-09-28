"""
external.py
===========
Wrappers for the two generators that live in their own repositories, and one
collector that turns any generator's output into this project's format
(``data/generated/<dataset>/<source>/<stem>.png`` + ``<stem>_mask.png``).

AnomalyDiffusion (Hu et al., AAAI 2024; MIT; github.com/sjtuplayer/anomalydiffusion)
    The fine-tuned few-shot arm of RQ3. Its scripts read an MVTec-AD-style folder
    and target Ubuntu (run under WSL2 on Windows, same GPU - project document
    Section 10.4). ``export_mvtec_layout`` writes such a folder containing ONLY
    the k train-split reference defects and clean train parts, so whatever
    fraction of "test" defects the repo trains on, it can only ever see training
    data. ``anomalydiffusion_commands`` prints the commands, including an
    under-trained checkpoint for the Stage-5 stress source
    ``anomalydiffusion_undertrained``.

TF-IDG (Xu et al., ICCV 2025; github.com/rubymiaomiao/TF-IDG)
    The official training-free generator. Its README documents only
    ``python run_inference.py`` and the checkpoints, not the input layout, so
    ``export_tfidg_inputs`` writes the four inputs the paper defines (normal image,
    reference defect, reference mask, target mask) and the user points the
    script at them. The in-repo alternative is ``generators.SDInpaintGenerator``.

``collect_pairs`` understands the common output layouts:
    <root>/**/image/<name>.* + <root>/**/mask/<name>.*      (AnomalyDiffusion)
    <root>/**/<name>.* + <name>_mask.*                      (this project, many repos)
"""

from __future__ import annotations

import glob
import json
import os
import shutil

import cv2
import numpy as np

from src.synth.references import clean_train_images, select_references

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ------------------------------------------------------------------ collector
def _find_pairs(root):
    pairs = []
    for img_dir in glob.glob(os.path.join(root, "**", "image"), recursive=True):
        mdir = os.path.join(os.path.dirname(img_dir), "mask")
        if not os.path.isdir(mdir):
            continue
        masks = {os.path.splitext(f)[0]: os.path.join(mdir, f) for f in os.listdir(mdir)
                 if f.lower().endswith(IMG_EXT)}
        for f in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXT and stem in masks:
                pairs.append((os.path.join(img_dir, f), masks[stem]))
    if pairs:
        return pairs
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        stem, ext = os.path.splitext(p)
        if ext.lower() not in IMG_EXT or stem.endswith("_mask"):
            continue
        for mext in (".png", ".jpg", ".bmp"):
            if os.path.exists(stem + "_mask" + mext):
                pairs.append((p, stem + "_mask" + mext))
                break
    return pairs


def collect_pairs(src_root: str, out_dir: str, source: str, min_mask_px: int = 4,
                  limit: int = None, note: str = "") -> int:
    """Copy (image, mask) pairs into ``out_dir`` with binarised masks. Returns count."""
    pairs = _find_pairs(src_root)
    if not pairs:
        raise FileNotFoundError(f"no image/mask pairs found under {src_root}")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for ip, mp in pairs:
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = ((m > 127) * 255).astype(np.uint8)
        if int((m > 0).sum()) < min_mask_px:
            continue
        stem = f"{source}_{n:05d}"
        cv2.imwrite(os.path.join(out_dir, stem + ".png"), img)
        cv2.imwrite(os.path.join(out_dir, stem + "_mask.png"), m)
        n += 1
        if limit and n >= limit:
            break
    with open(os.path.join(out_dir, "generator.json"), "w") as f:
        json.dump({"generator": source, "collected_from": os.path.abspath(src_root), "n": n,
                   "note": note}, f, indent=1)
    return n


# ------------------------------------------------------------------ AnomalyDiffusion
def export_mvtec_layout(dataset: str, out_root: str, k: int = 5, seed: int = 0,
                        n_good: int = 200, processed_dir: str = "data/processed") -> str:
    """MVTec-AD-style folder built ONLY from the train split.

    <out_root>/<dataset>/train/good/*.png                 clean train parts
    <out_root>/<dataset>/test/<class>/*.png               the k reference defects
    <out_root>/<dataset>/ground_truth/<class>/*_mask.png  their masks
    """
    cat = os.path.join(out_root, dataset)
    if os.path.exists(cat):
        shutil.rmtree(cat)
    refs = select_references(dataset, k, seed, processed_dir)
    clean = clean_train_images(dataset, processed_dir)
    rng = np.random.default_rng(seed)
    good = [clean[i] for i in rng.permutation(len(clean))[:n_good]]
    os.makedirs(os.path.join(cat, "train", "good"), exist_ok=True)
    os.makedirs(os.path.join(cat, "test", "good"), exist_ok=True)
    for i, p in enumerate(good):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(cat, "train", "good", f"{i:03d}.png"), img)
        if i < 5:
            cv2.imwrite(os.path.join(cat, "test", "good", f"{i:03d}.png"), img)
    counts = {}
    for r in refs:
        cls = r.class_name.replace(" ", "_")
        j = counts.get(cls, 0)
        counts[cls] = j + 1
        for sub in ("test", "ground_truth"):
            os.makedirs(os.path.join(cat, sub, cls), exist_ok=True)
        cv2.imwrite(os.path.join(cat, "test", cls, f"{j:03d}.png"), cv2.imread(r.file_name, cv2.IMREAD_COLOR))
        cv2.imwrite(os.path.join(cat, "ground_truth", cls, f"{j:03d}_mask.png"), r.mask)
    with open(os.path.join(cat, "provenance.json"), "w") as f:
        json.dump({"dataset": dataset, "split": "train", "references": [r.file_name for r in refs],
                   "n_good": len(good), "classes": counts}, f, indent=1)
    return cat


def anomalydiffusion_commands(mvtec_root: str, repo_dir: str = "external/anomalydiffusion",
                              undertrained_steps: int = 500) -> str:
    """Commands to run in WSL2 / Ubuntu from the AnomalyDiffusion repository."""
    wsl_root = mvtec_root.replace("\\", "/")
    if len(wsl_root) > 1 and wsl_root[1] == ":":
        wsl_root = f"/mnt/{wsl_root[0].lower()}{wsl_root[2:]}"
    return f"""# --- AnomalyDiffusion (AAAI 2024), run inside WSL2 / Ubuntu -------------------
# one-time setup (see the repository README):
#   git clone https://github.com/sjtuplayer/anomalydiffusion {repo_dir}
#   cd {repo_dir} && conda env create -f environment.yaml && conda activate Anomalydiffusion
#   mkdir -p models/ldm/text2img-large && wget -O models/ldm/text2img-large/model.ckpt \\
#        https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
# 1) learn the anomaly embeddings from the train-split references only:
CUDA_VISIBLE_DEVICES=0 python main.py --spatial_encoder_embedding --data_enhance \\
    --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml \\
    -t --actual_resume models/ldm/text2img-large/model.ckpt -n sperc --gpus 0, \\
    --init_word anomaly --mvtec_path={wsl_root}
# 2) train the mask generator and generate image-mask pairs:
CUDA_VISIBLE_DEVICES=0 python run-mvtec.py --data_path={wsl_root}
# 3) Stage-5 stress source: repeat 1)-2) with training stopped after ~{undertrained_steps} steps
#    (e.g. lightning --max_steps {undertrained_steps}) into a separate output folder.
# 4) back in Windows, collect the pairs into this project:
#   vqaenv\\Scripts\\python.exe scripts\\generate_synthetic.py --collect <generated_dir> \\
#       --dataset <dataset> --source anomalydiffusion            (or anomalydiffusion_undertrained)
"""


# ------------------------------------------------------------------ TF-IDG
def export_tfidg_inputs(dataset: str, out_dir: str, n: int, k: int = 5, seed: int = 0,
                        processed_dir: str = "data/processed") -> str:
    """The four inputs TF-IDG takes, one folder per sample, train split only:
    normal.png, reference.png, reference_mask.png, target_mask.png."""
    from src.synth.generators import defect_patch, random_transform
    refs = select_references(dataset, k, seed, processed_dir)
    clean = clean_train_images(dataset, processed_dir)
    rng = np.random.default_rng(seed + 5)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n):
        r = refs[i % len(refs)]
        bg = cv2.imread(clean[int(rng.integers(len(clean)))], cv2.IMREAD_COLOR)
        _, pm = defect_patch(r)
        _, tm = random_transform(np.zeros(pm.shape + (3,), np.uint8), pm, rng)
        H, W = bg.shape[:2]
        th, tw = tm.shape
        if th >= H or tw >= W:
            continue
        y, x = int(rng.integers(0, H - th)), int(rng.integers(0, W - tw))
        target = np.zeros((H, W), np.uint8)
        target[y:y + th, x:x + tw] = tm
        d = os.path.join(out_dir, f"sample_{i:05d}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "normal.png"), bg)
        cv2.imwrite(os.path.join(d, "reference.png"), cv2.imread(r.file_name, cv2.IMREAD_COLOR))
        cv2.imwrite(os.path.join(d, "reference_mask.png"), r.mask)
        cv2.imwrite(os.path.join(d, "target_mask.png"), target)
    return out_dir
