"""
anomaly_scores.py
=================
Image-level anomaly scores for the MVTec AD bridge (project document, Section 9 and
Stage 2 of Section 12): SPI's native setting is ONE score per image, so running
SPERC on an anomaly detector's image scores validates the transporter on its own
kind of score before it is used on detector escape scores.

The scorer is a compact PatchCore (Roth et al., CVPR 2022): frozen ImageNet
WideResNet-50-2 features from layers 2 and 3, locally averaged, a memory bank of
patch features from the defect-free TRAIN images, greedy coreset subsampling, and
the image score = the largest nearest-neighbour distance of any test patch to the
memory bank. Higher = more anomalous, so "caught" is score >= threshold, exactly
like a detector's escape score.
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np


class PatchCoreLite:
    def __init__(self, backbone: str = "wide_resnet50_2", image_size: int = 224,
                 coreset_fraction: float = 0.1, device: str = None, seed: int = 0,
                 batch_size: int = 16, max_bank: int = 60000):
        import torch
        import torchvision
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = "IMAGENET1K_V1"
        net = getattr(torchvision.models, backbone)(weights=weights)
        net.eval().to(self.device)
        self.net = net
        self.size = image_size
        self.frac = coreset_fraction
        self.seed = seed
        self.bs = batch_size
        self.max_bank = max_bank
        self.bank = None
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    # ------------------------------------------------------------ features
    def _load(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        s = int(round(self.size * 256 / 224))
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        o = (s - self.size) // 2
        img = img[o:o + self.size, o:o + self.size].astype(np.float32) / 255.0
        return ((img - self.mean) / self.std).transpose(2, 0, 1)

    def _features(self, paths: List[str]):
        torch = self.torch
        F = torch.nn.functional
        x = torch.from_numpy(np.stack([self._load(p) for p in paths])).to(self.device)
        n = self.net
        with torch.no_grad():
            h = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l1 = n.layer1(h)
            l2 = n.layer2(l1)
            l3 = n.layer3(l2)
            f2 = F.avg_pool2d(l2, 3, 1, 1)
            f3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear",
                               align_corners=False)
            f = torch.cat([f2, f3], 1)                     # (B, 1536, h, w)
        B, C, H, W = f.shape
        return f.permute(0, 2, 3, 1).reshape(B, H * W, C)

    # ------------------------------------------------------------ memory bank
    def _coreset(self, feats):
        torch = self.torch
        n = feats.shape[0]
        k = max(1, min(int(self.frac * n), self.max_bank))
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        if self.device == "cpu" and n * k > 4e9:          # greedy too slow on CPU: random subset
            idx = torch.randperm(n, generator=g)[:k]
            return feats[idx.to(feats.device)]
        proj = torch.randn(feats.shape[1], 128, generator=g).to(feats.device) / np.sqrt(128)
        z = feats @ proj
        sel = [int(torch.randint(n, (1,), generator=g))]
        d = torch.cdist(z, z[sel[-1]][None]).squeeze(1)
        for _ in range(k - 1):
            i = int(torch.argmax(d))
            sel.append(i)
            d = torch.minimum(d, torch.cdist(z, z[i][None]).squeeze(1))
        return feats[torch.tensor(sel, device=feats.device)]

    def fit(self, good_paths: List[str]):
        feats = [self._features(good_paths[i:i + self.bs]).reshape(-1, 1536)
                 for i in range(0, len(good_paths), self.bs)]
        self.bank = self._coreset(self.torch.cat(feats, 0))
        return self

    def score(self, paths: List[str]) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(paths), self.bs):
            f = self._features(paths[i:i + self.bs])       # (B, P, C)
            for b in range(f.shape[0]):
                d = torch.cdist(f[b], self.bank).min(dim=1).values
                out.append(float(d.max()))
        return np.asarray(out, float)


def mvtec_category_files(root: str, category: str):
    """(good train paths, [(test path, defect_type)]) for one MVTec AD category."""
    cat = os.path.join(root, category)
    good = sorted(os.path.join(cat, "train", "good", f)
                  for f in os.listdir(os.path.join(cat, "train", "good")) if f.lower().endswith(".png"))
    test = []
    for t in sorted(os.listdir(os.path.join(cat, "test"))):
        d = os.path.join(cat, "test", t)
        if os.path.isdir(d):
            test += [(os.path.join(d, f), t) for f in sorted(os.listdir(d)) if f.lower().endswith(".png")]
    return good, test
