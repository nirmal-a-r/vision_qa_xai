"""
transforms.py
=============
Albumentations pipelines for train/val, plus the synthetic-augmentation
pipeline used to address the data-scarcity objective from PROJECT.md
(lighting variation, sensor noise, rare defect geometries).

All pipelines emit `pascal_voc` (xmin, ymin, xmax, ymax) boxes so they can be
converted straight to COCO xywh right before being handed to the HF image
processor.
"""

import inspect
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Albumentations has renamed a couple of kwargs across recent versions
# (PadIfNeeded: value -> fill; GaussNoise: var_limit -> std_range). Passing
# the old name doesn't error, it just silently falls back to a default and
# ignores what you asked for — so we detect which signature is installed
# once, here, rather than hard-coding one and hoping.
_PAD_FILL_KW = "fill" if "fill" in inspect.signature(A.PadIfNeeded).parameters else "value"
_GAUSS_NOISE_NEW_API = "std_range" in inspect.signature(A.GaussNoise).parameters


def _pad_if_needed(image_size: int):
    kwargs = {"min_height": image_size, "min_width": image_size, "border_mode": 0}
    kwargs[_PAD_FILL_KW] = (0, 0, 0) if _PAD_FILL_KW == "value" else 0
    return A.PadIfNeeded(**kwargs)


def _gauss_noise(light: bool = False, p: float = None):
    p = p if p is not None else (0.3 if light else 0.6)
    if _GAUSS_NOISE_NEW_API:
        # std_range is fraction-of-max-value based, roughly analogous to the
        # old var_limit=(5,25)/(10,50) pixel-variance ranges.
        return A.GaussNoise(std_range=(0.02, 0.08) if light else (0.04, 0.15), p=p)
    return A.GaussNoise(var_limit=(5.0, 25.0) if light else (10.0, 50.0), p=p)


def get_train_transforms(image_size: int = 800):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            _pad_if_needed(image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            _gauss_noise(light=True),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.2),
    )


def get_val_transforms(image_size: int = 800):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            _pad_if_needed(image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.0),
    )


def get_synthetic_augmentation_transforms(image_size: int = 800):
    """Heavier augmentation used to synthesize extra training samples for
    rare defect classes / cold-start lines. Applied as an offline pre-pass
    (see notebook cell 'Synthetic augmentation'), not during normal training."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            _pad_if_needed(image_size),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.CLAHE(clip_limit=3.0, p=1.0),
            ], p=0.8),
            A.OneOf([
                _gauss_noise(light=False, p=1.0),
                A.ISONoise(p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.6),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3),
    )
