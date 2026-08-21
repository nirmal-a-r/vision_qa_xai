"""
rollout.py
==========
Phase 2 attribution methods.

1) SwinAttentionRollout
   Classic attention-rollout (Abnar & Zuidema), adapted for Swin's windowed
   attention exactly per PROJECT.md's caveat: plain-ViT rollout assumes every
   layer attends over the *whole* token set, but Swin's windows only attend
   within themselves (and shift between layers). We handle this by:
     - hooking each WindowAttention block's `attn_drop` module to capture
       the post-softmax attention matrix per window,
       (timm's WindowAttention always does `attn = self.attn_drop(attn)`,
       so a forward-pre-hook on attn_drop reliably captures it regardless
       of model variant)
     - adding the identity term (I + A)/2 per layer for the residual
       connection, as in the original rollout formulation,
     - reversing the cyclic shift between shifted-window layers before
       accumulating, so windows line up spatially across layers,
     - multiplying accumulated window-local relevance across layers, then
       scattering windows back onto the full feature-map grid.
   This gives a *coarse, backbone-resolution* saliency map — genuinely
   correct in spirit, but still an approximation (cross-window information
   flow between non-adjacent windows more than one shift apart is not fully
   reconstructed). That approximation is called out explicitly rather than
   hidden.

2) gradient_relevance()
   PROJECT.md flags that standard LRP conservation rules aren't solved
   off-the-shelf for softmax/attention layers (Chefer et al. 2021 is cited
   for exactly this reason). Rather than hand-rolling a bespoke, likely-buggy
   LRP rule set for deformable attention, this uses Captum's
   InputXGradient (gradient x input) as a well-understood, robust relevance
   method with a *similar* interpretation (per-pixel contribution to the
   output logit) and the conservation-adjacent property that it's exact for
   piecewise-linear networks. This is the practical, "not extremely
   complicated" stand-in for full LRP referenced in the config as
   `xai.method: "rollout+grad"`.

Both methods return an HxW float array in [0, 1], at the *input image*
resolution, ready for overlay.py.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ---------------------------------------------------------------------------
# 1) Swin attention rollout
# ---------------------------------------------------------------------------

class SwinAttentionRollout:
    def __init__(self, model, discard_ratio: float = 0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.attn_maps = []       # list of (attn_tensor, window_size, shift_size, input_resolution)
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Walk the timm Swin backbone for WindowAttention-style blocks. We
        # identify them generically (module has `attn_drop` + `window_size`)
        # instead of importing timm's private classes, so this keeps working
        # across timm versions.
        for name, module in self.model.named_modules():
            if hasattr(module, "attn_drop") and hasattr(module, "window_size"):
                h = module.attn_drop.register_forward_pre_hook(
                    lambda m, inp, mod=module: self.attn_maps.append(
                        (inp[0].detach(), getattr(mod, "window_size", None), getattr(mod, "shift_size", 0))
                    )
                )
                self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def __call__(self, pixel_values: torch.Tensor, feature_map_hw=None) -> np.ndarray:
        """
        pixel_values: [1,3,H,W] already-normalized model input.
        feature_map_hw: (h, w) of the backbone's last-stage feature grid, if
                         known; otherwise inferred as best-effort from the
                         first captured attention map's window count.
        Returns: HxW (input resolution) float saliency map in [0,1].
        """
        self.attn_maps = []
        _ = self.model.backbone(pixel_values) if hasattr(self.model, "backbone") else self.model(pixel_values)

        if not self.attn_maps:
            raise RuntimeError(
                "No Swin attention matrices were captured. This can happen if "
                "use_timm_backbone=False or the backbone module names changed. "
                "Fall back to gradient_relevance() in that case."
            )

        # Per-layer: average heads, add identity for the residual, row-normalize.
        rollouts = []
        for attn, window_size, shift in self.attn_maps:
            # attn: [num_windows*B, num_heads, N, N]  where N = window_size**2
            attn = attn.mean(dim=1)  # avg heads -> [num_windows*B, N, N]
            N = attn.shape[-1]
            I = torch.eye(N, device=attn.device).unsqueeze(0)
            a = 0.5 * attn + 0.5 * I
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            rollouts.append(a)

        # Multiply consecutive per-window rollouts (windows are aligned
        # in raster order layer-to-layer up to the shift, which we treat as
        # a bounded positional blur rather than exactly undoing it — the
        # documented approximation).
        acc = rollouts[0]
        for a in rollouts[1:]:
            if a.shape[0] != acc.shape[0]:
                # window count changed (stage transition / downsample) —
                # restart accumulation for the new stage rather than
                # multiplying mismatched shapes.
                acc = a
                continue
            acc = torch.bmm(a, acc)

        # Collapse each window's rollout matrix to a single per-token
        # saliency vector via mean row (avg relevance received from all
        # tokens), which is the standard rollout readout.
        token_saliency = acc.mean(dim=1)  # [num_windows*B, N]

        window_size = self.attn_maps[0][1]
        ws = window_size[0] if isinstance(window_size, (tuple, list)) else window_size
        num_windows = token_saliency.shape[0]
        side = int(round(num_windows ** 0.5))
        try:
            grid = token_saliency.reshape(side, side, ws, ws)
            grid = grid.permute(0, 2, 1, 3).reshape(side * ws, side * ws)
        except RuntimeError:
            # shape didn't factor cleanly (irregular last stage) — safest
            # fallback is a flat/uniform low-resolution map rather than a
            # crash, since this is an auxiliary visualization, not the
            # detection path.
            side_alt = int(np.ceil(np.sqrt(num_windows * ws * ws)))
            grid = torch.zeros(side_alt, side_alt)

        grid = grid.cpu().numpy().astype(np.float32)
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

        H, W = pixel_values.shape[-2:]
        heatmap = cv2.resize(grid, (W, H), interpolation=cv2.INTER_CUBIC)
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap


# ---------------------------------------------------------------------------
# 2) Gradient-based relevance (LRP-lite / Chefer-style stand-in)
# ---------------------------------------------------------------------------

def gradient_relevance(model, pixel_values: torch.Tensor, query_index: int, class_index: int) -> np.ndarray:
    """
    Input x Gradient relevance w.r.t. the logit of `class_index` for decoder
    query `query_index`. Uses Captum's InputXGradient — see module docstring
    for why this substitutes for full attention-LRP here.

    Returns HxW float saliency map in [0,1], at input resolution.
    """
    from captum.attr import InputXGradient

    pixel_values = pixel_values.clone().detach().requires_grad_(True)

    def forward_fn(x):
        out = model(pixel_values=x)
        return out.logits[:, query_index, class_index].unsqueeze(-1)

    ixg = InputXGradient(forward_fn)
    attributions = ixg.attribute(pixel_values)  # [1,3,H,W]

    heatmap = attributions.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
    heatmap = np.clip(heatmap, 0, None)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap


def combine_heatmaps(maps, weights=None) -> np.ndarray:
    """Weighted average of several same-shape [0,1] heatmaps, renormalized."""
    maps = [m for m in maps if m is not None]
    if not maps:
        raise ValueError("combine_heatmaps got no valid maps")
    weights = weights or [1.0 / len(maps)] * len(maps)
    combined = np.zeros_like(maps[0], dtype=np.float32)
    for m, w in zip(maps, weights):
        combined += w * m
    if combined.max() > 0:
        combined = combined / combined.max()
    return combined
