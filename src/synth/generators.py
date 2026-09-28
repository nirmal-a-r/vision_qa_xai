"""
generators.py
=============
Few-shot synthetic-defect generators that run inside this repository, writing
``<out_dir>/<stem>.png`` + ``<stem>_mask.png`` pairs that ``composite.py`` turns
into a labelled set. References and backgrounds come from the TRAIN split only
(``references.py``).

* ``copy_paste``   - training-free, no model: a real reference defect, randomly
                     rotated / scaled / flipped, Poisson-blended onto a clean
                     train part. Cheap, needs nothing but OpenCV; a low-fidelity
                     reference point for RQ3 and a smoke-test generator.
* ``training_free`` (``SDInpaintGenerator``) - training-free, one-shot diffusion
                     generator in the spirit of TF-IDG (Xu et al., ICCV 2025): the
                     reference defect is placed on a clean train part (adaptive
                     mask), the defect region is cropped and enlarged, re-synthesised
                     by Stable Diffusion inpainting at partial strength (appearance
                     guidance from the placed reference), and blended back so the
                     background texture outside the mask is preserved exactly.
                     This is NOT the official TF-IDG code (github.com/rubymiaomiao/TF-IDG);
                     its outputs can be used instead via ``external.collect_pairs``,
                     and the paper must name whichever was used.

The fine-tuned few-shot arm (AnomalyDiffusion, AAAI 2024) runs in its own
repository; see ``external.py``.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import cv2
import numpy as np

from src.synth.references import Reference, clean_train_images, select_references


# ------------------------------------------------------------------ geometry
def defect_patch(ref: Reference, margin: int = 6):
    """Crop (image, mask) to the defect's bounding box plus a margin."""
    img = cv2.imread(ref.file_name, cv2.IMREAD_COLOR)
    ys, xs = np.nonzero(ref.mask)
    if ys.size == 0:
        raise ValueError(f"empty mask for {ref.file_name}")
    H, W = ref.mask.shape
    y1, y2 = max(0, ys.min() - margin), min(H, ys.max() + 1 + margin)
    x1, x2 = max(0, xs.min() - margin), min(W, xs.max() + 1 + margin)
    return img[y1:y2, x1:x2].copy(), ref.mask[y1:y2, x1:x2].copy()


def random_transform(patch, mask, rng, max_rot=180.0, scale=(0.7, 1.3)):
    """Random flip / rotation / scale of a defect patch and its mask (kept binary)."""
    if rng.random() < 0.5:
        patch, mask = patch[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        patch, mask = patch[::-1], mask[::-1]
    s = float(rng.uniform(*scale))
    ang = float(rng.uniform(-max_rot, max_rot))
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    p = cv2.warpAffine(np.ascontiguousarray(patch), M, (nw, nh), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(np.ascontiguousarray(mask), M, (nw, nh), flags=cv2.INTER_NEAREST)
    return p, ((m > 127) * 255).astype(np.uint8)


def place(background, patch, mask, rng, blend="poisson"):
    """Paste a transformed defect onto a clean part. Returns (image, full mask) or None."""
    H, W = background.shape[:2]
    ph, pw = mask.shape
    if ph >= H - 2 or pw >= W - 2:
        f = 0.8 * min((H - 2) / ph, (W - 2) / pw)
        if f <= 0:
            return None
        patch = cv2.resize(patch, (max(2, int(pw * f)), max(2, int(ph * f))))
        mask = cv2.resize(mask, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_NEAREST)
        ph, pw = mask.shape
    if mask.max() == 0:
        return None
    x = int(rng.integers(1, W - pw))
    y = int(rng.integers(1, H - ph))
    full = np.zeros((H, W), np.uint8)
    full[y:y + ph, x:x + pw] = mask
    out = background.copy()
    if blend == "poisson":
        try:
            src = np.zeros_like(out)
            src[y:y + ph, x:x + pw] = patch
            m = cv2.dilate(full, np.ones((5, 5), np.uint8))
            ys, xs = np.nonzero(m)
            center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
            out = cv2.seamlessClone(src, out, m, center, cv2.NORMAL_CLONE)
            return out, full
        except cv2.error:
            pass
    a = cv2.GaussianBlur((full > 0).astype(np.float32), (7, 7), 0)[..., None]
    a = np.maximum(a, (full > 0)[..., None] * 1.0)
    src = out.astype(np.float32).copy()
    src[y:y + ph, x:x + pw] = patch
    out = (a * src + (1 - a) * out.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return out, full


def _is_gray(img):
    return img.ndim == 2 or (np.abs(img[..., 0].astype(int) - img[..., 1]).max() < 2
                             and np.abs(img[..., 1].astype(int) - img[..., 2]).max() < 2)


# ------------------------------------------------------------------ generators
class CopyPasteGenerator:
    name = "copy_paste"

    def __init__(self, dataset, k=5, seed=0, processed_dir="data/processed", blend="poisson"):
        self.dataset, self.seed, self.blend = dataset, seed, blend
        self.refs = select_references(dataset, k, seed, processed_dir)
        self.clean = clean_train_images(dataset, processed_dir)
        if not self.refs:
            raise RuntimeError(f"no defective train images for {dataset}")
        if not self.clean:
            raise RuntimeError(f"{dataset} has no clean TRAIN images to paint on; generators are "
                               "meant for the primary datasets (KolektorSDD2, Magnetic-Tile)")
        self.patches = [defect_patch(r) for r in self.refs]

    def compose(self, i, rng):
        j = i % len(self.patches)
        p, m = random_transform(*self.patches[j], rng)
        bg = cv2.imread(self.clean[int(rng.integers(len(self.clean)))], cv2.IMREAD_COLOR)
        res = place(bg, p, m, rng, self.blend)
        return res, j

    def refine(self, image, mask, j, rng):        # identity for copy-paste
        return image

    def generate(self, n, out_dir, verbose=True):
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.default_rng(self.seed + 17)
        manifest, i, tries = [], 0, 0
        while i < n and tries < 5 * n:
            tries += 1
            res, j = self.compose(i, rng)
            if res is None:
                continue
            img, msk = res
            img = self.refine(img, msk, j, rng)
            stem = f"{self.name}_{i:05d}"
            cv2.imwrite(os.path.join(out_dir, stem + ".png"), img)
            cv2.imwrite(os.path.join(out_dir, stem + "_mask.png"), msk)
            manifest.append({"stem": stem, "reference": os.path.basename(self.refs[j].file_name),
                             "reference_class": self.refs[j].class_name})
            i += 1
            if verbose and i % 100 == 0:
                print(f"  [{self.dataset}/{self.name}] {i}/{n}", flush=True)
        with open(os.path.join(out_dir, "generator.json"), "w") as f:
            json.dump({"generator": self.name, "dataset": self.dataset, "n": i,
                       "references": [r.file_name for r in self.refs], "split": "train",
                       "samples": manifest, **self.describe()}, f, indent=1)
        return i

    def describe(self):
        return {"method": "copy-paste with Poisson blending", "blend": self.blend}


class SDInpaintGenerator(CopyPasteGenerator):
    """Training-free one-shot diffusion generator (crop-enlarge-inpaint-blend)."""
    name = "training_free"

    def __init__(self, dataset, k=5, seed=0, processed_dir="data/processed",
                 model_id="sd2-community/stable-diffusion-2-inpainting", strength=0.55,
                 steps=30, guidance=4.0, prompt="a close-up photo of an industrial surface "
                 "with a {defect} defect", device=None, window=512):
        super().__init__(dataset, k, seed, processed_dir)
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        # stabilityai/stable-diffusion-2-inpainting was deprecated on the Hub; the
        # community mirror holds the same weights. A local folder path also works.
        candidates = [model_id] + [m for m in FALLBACK_INPAINT_MODELS if m != model_id]
        errors = []
        for mid in candidates:
            try:
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(mid, torch_dtype=dtype)
                model_id = mid
                break
            except Exception as e:                       # noqa: BLE001 - report every attempt
                errors.append(f"{mid}: {type(e).__name__}: {str(e)[:120]}")
        else:
            raise RuntimeError("could not load an inpainting model:\n  " + "\n  ".join(errors))
        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        # PyTorch's built-in scaled-dot-product attention is used by default (project
        # document, Section 10.4: prefer it over third-party kernels on a new GPU
        # architecture). Attention slicing saves memory but falls back to a slower
        # path, so it is only switched on for GPUs under 6 GB.
        if self.device.startswith("cuda"):
            try:
                total = torch.cuda.get_device_properties(torch.device(self.device)).total_memory
                if total < 6 * 2**30:
                    self.pipe.enable_attention_slicing()
            except Exception:
                pass
        self.torch = torch
        self.model_id, self.strength, self.steps = model_id, strength, steps
        self.guidance, self.prompt, self.window = guidance, prompt, window

    def refine(self, image, mask, j, rng):
        from PIL import Image
        H, W = mask.shape
        ys, xs = np.nonzero(mask)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        ext = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        # Adaptive mask: the crop is sized so the defect fills a sizeable part of the
        # enlarged window, so small defects are not ignored by the diffusion model.
        side = int(np.clip(2.5 * ext, 64, min(H, W, 2 * self.window)))
        y1 = int(np.clip(cy - side // 2, 0, max(0, H - side)))
        x1 = int(np.clip(cx - side // 2, 0, max(0, W - side)))
        crop, cm = image[y1:y1 + side, x1:x1 + side], mask[y1:y1 + side, x1:x1 + side]
        h0, w0 = cm.shape
        grow = max(3, int(0.08 * max(h0, w0)))
        m_in = cv2.dilate(cm, np.ones((grow, grow), np.uint8))
        S = self.window
        g = self.torch.Generator(device=self.device).manual_seed(int(rng.integers(2**31)))
        out = self.pipe(prompt=self.prompt.format(defect=self.refs[j].class_name.replace("_", " ")),
                        image=Image.fromarray(cv2.cvtColor(cv2.resize(crop, (S, S)), cv2.COLOR_BGR2RGB)),
                        mask_image=Image.fromarray(cv2.resize(m_in, (S, S), interpolation=cv2.INTER_NEAREST)),
                        strength=self.strength, num_inference_steps=self.steps,
                        guidance_scale=self.guidance, generator=g, height=S, width=S).images[0]
        gen = cv2.resize(cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR), (w0, h0),
                         interpolation=cv2.INTER_AREA)
        if _is_gray(crop):
            gen = cv2.cvtColor(cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        # Texture preservation: only the (feathered) defect region is replaced.
        a = cv2.GaussianBlur((m_in > 0).astype(np.float32), (0, 0), sigmaX=max(1.0, grow / 3))[..., None]
        res = image.copy()
        res[y1:y1 + h0, x1:x1 + w0] = (a * gen + (1 - a) * crop).clip(0, 255).astype(np.uint8)
        return res

    def describe(self):
        return {"method": "training-free SD inpainting (crop-enlarge-inpaint-blend), TF-IDG-style",
                "model_id": self.model_id, "strength": self.strength, "steps": self.steps,
                "guidance": self.guidance, "prompt": self.prompt, "window": self.window}


FALLBACK_INPAINT_MODELS = ("sd2-community/stable-diffusion-2-inpainting",
                           "stabilityai/stable-diffusion-2-inpainting",
                           "stable-diffusion-v1-5/stable-diffusion-inpainting")

GENERATORS = {"copy_paste": CopyPasteGenerator, "training_free": SDInpaintGenerator}


def audit_sheet(gen_dir: str, out_path: Optional[str] = None, n: int = 36, tile: int = 160):
    """Contact sheet of generated samples with the mask outline, for visual audit."""
    stems = sorted(f[:-9] for f in os.listdir(gen_dir) if f.endswith("_mask.png"))[:n]
    if not stems:
        return None
    cols = 6
    rows = (len(stems) + cols - 1) // cols
    sheet = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
    for i, s in enumerate(stems):
        img = cv2.imread(os.path.join(gen_dir, s + ".png"), cv2.IMREAD_COLOR)
        m = cv2.imread(os.path.join(gen_dir, s + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        cnt, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (0, 0, 255), 1)
        f = tile / max(img.shape[:2])
        t = cv2.resize(img, (max(1, int(img.shape[1] * f)), max(1, int(img.shape[0] * f))))
        r, c = divmod(i, cols)
        sheet[r * tile:r * tile + t.shape[0], c * tile:c * tile + t.shape[1]] = t
    out_path = out_path or os.path.join(gen_dir, "_audit_sheet.png")
    cv2.imwrite(out_path, sheet)
    return out_path


def list_pairs(gen_dir: str) -> List[str]:
    return sorted(f[:-9] for f in os.listdir(gen_dir) if f.endswith("_mask.png"))
