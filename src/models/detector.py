"""
detector.py
===========
Builds Deformable DETR (HF `transformers`) with a Swin Transformer backbone
in place of the default ResNet — matching PROJECT.md's Phase 1 choice
(hierarchical multi-scale features via shifted-window attention, needed
because defects range from tiny mouse-bites/pinholes to large delamination).

HF's DeformableDetr supports swapping in any timm backbone by setting
`use_timm_backbone=True` and `backbone=<timm model name>` on the config —
no custom model surgery required, which keeps this on the "not extremely
complicated" side while still being architecturally faithful to the spec.

Loss: HF's built-in DeformableDetrLoss already implements
  - sigmoid focal loss for classification (alpha/gamma configurable)
  - L1 + GIoU for box regression
  - Hungarian matching
so no separate losses.py is needed here (PROJECT.md's losses.py role is
folded into this config).
"""

import dataclasses

import torch
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection


def _set(config, name, value):
    """setattr that respects the field's declared type.

    transformers>=5 turned the model configs into `@strict` dataclasses, so
    `config.class_cost = 1.0` now raises (the field is declared `int`) where
    on 4.x it was silently accepted. Coerce to the declared type instead of
    hard-coding either version's expectations.
    """
    declared = {f.name: f.type for f in dataclasses.fields(config)} if dataclasses.is_dataclass(config) else {}
    t = declared.get(name)
    if t is int and isinstance(value, float) and float(value).is_integer():
        value = int(value)
    elif t is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    setattr(config, name, value)


def _configure_backbone(config, backbone, out_indices, pretrained_backbone, use_timm_backbone):
    """Point the detector at a timm backbone, across both config generations.

    transformers 4.x: flat `use_timm_backbone` / `backbone` / `backbone_kwargs`
    attributes on the detector config.

    transformers 5.x: those attributes are gone; the backbone is a nested
    `backbone_config` sub-config. Note that 5.x still *accepts* the old kwargs
    on the constructor but silently ignores `backbone=` and falls back to
    ResNet-50 — so we build the TimmBackboneConfig explicitly rather than
    relying on that path.
    """
    if hasattr(config, "use_timm_backbone"):          # transformers 4.x
        config.use_timm_backbone = use_timm_backbone
        config.backbone = backbone
        # Don't set "features_only" here — DeformableDetrConvEncoder already
        # passes features_only=True itself when calling timm.create_model, so
        # including it raises "multiple values for keyword argument".
        # out_indices selects which Swin stages feed the multi-scale deformable
        # attention (stages 1,2,3 -> strides 8/16/32).
        config.backbone_kwargs = {"out_indices": tuple(out_indices)}
        config.use_pretrained_backbone = pretrained_backbone
        return config

    from transformers import TimmBackboneConfig      # transformers 5.x
    config.backbone_config = TimmBackboneConfig(
        backbone=backbone,
        features_only=True,
        use_pretrained_backbone=pretrained_backbone,
        out_indices=list(out_indices),
        num_channels=3,
    )
    return config


def build_model(
    num_classes: int,
    backbone: str = "swin_base_patch4_window7_224",
    use_timm_backbone: bool = True,
    base_checkpoint: str = "SenseTime/deformable-detr",
    num_queries: int = 100,
    focal_alpha: float = 0.25,
    class_cost: float = 1.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
    pretrained_backbone: bool = True,
    out_indices=(1, 2, 3),
    image_size: int = None,
    auxiliary_loss: bool = True,
    with_box_refine: bool = True,
):
    """Returns a DeformableDetrForObjectDetection with a Swin backbone,
    randomly-initialized detection/segmentation heads sized for
    `num_classes`, loaded on top of the pretrained Deformable DETR
    transformer weights where shapes match.
    """
    config = DeformableDetrConfig.from_pretrained(base_checkpoint)

    # --- swap backbone: ResNet -> Swin (timm) -----------------------------
    # (stages 1,2,3 -> strides 8/16/32, the multi-scale pyramid PROJECT.md
    # calls for; deformable attention adds a 4th level on top of stage 3.)
    config = _configure_backbone(config, backbone, out_indices, pretrained_backbone, use_timm_backbone)
    _set(config, "dilation", False)

    # --- task-specific heads ----------------------------------------------
    config.num_labels = num_classes
    _set(config, "num_queries", num_queries)

    # --- loss hyperparameters (Focal + GIoU per PROJECT.md) ----------------
    _set(config, "focal_alpha", focal_alpha)
    _set(config, "class_cost", class_cost)
    _set(config, "bbox_cost", bbox_cost)
    _set(config, "giou_cost", giou_cost)
    _set(config, "bbox_loss_coefficient", bbox_cost)
    _set(config, "giou_loss_coefficient", giou_cost)

    # --- convergence settings ---------------------------------------------
    # Both default to False in HF but are ON in the Deformable DETR paper, and
    # they're the two biggest levers on how fast the decoder queries
    # specialize. Without them the model converges to a single averaged box
    # repeated across all queries, with confidences stuck near the prior.
    _set(config, "auxiliary_loss", auxiliary_loss)
    _set(config, "with_box_refine", with_box_refine)

    model = DeformableDetrForObjectDetection.from_pretrained(
        base_checkpoint,
        config=config,
        ignore_mismatched_sizes=True,  # class head shape changes with num_classes
    )
    _restore_timm_backbone(model, backbone, out_indices, pretrained_backbone, image_size)
    return model, config


class _ToNCHW(torch.nn.Module):
    """Wraps a timm feature extractor that emits NHWC maps, transposing to NCHW."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.feature_info = net.feature_info

    def forward(self, x):
        return [f.permute(0, 3, 1, 2).contiguous() for f in self.net(x)]


def _ensure_nchw(net, image_size):
    """Probe the feature extractor once and wrap it if it came back NHWC.

    `output_fmt="NCHW"` covers most timm versions, but it's silently ignored by
    some feature wrappers, and a wrong layout here fails deep inside the
    detector with a confusing conv channel-mismatch — so verify rather than
    assume.
    """
    expected = list(net.feature_info.channels())
    size = image_size or 224
    was_training = net.training
    net.eval()
    with torch.no_grad():
        feats = net(torch.zeros(1, 3, size, size))
    net.train(was_training)

    if feats[0].shape[1] == expected[0]:
        return net                      # already NCHW
    if feats[0].shape[-1] == expected[0]:
        return _ToNCHW(net)             # NHWC -> wrap
    raise RuntimeError(
        f"unexpected backbone feature layout {tuple(feats[0].shape)} for channels {expected}"
    )


def _restore_timm_backbone(model, backbone, out_indices, pretrained, image_size=None):
    """Rebuild the timm feature extractor and swap it into the built detector.

    Necessary because `from_pretrained` re-initializes every backbone weight
    absent from the Deformable-DETR checkpoint — and that checkpoint is
    ResNet-50 based, so *all* the Swin weights count as missing. Two things go
    wrong as a result:

      1. the timm ImageNet-pretrained Swin weights we asked for are silently
         discarded and replaced with random init, and
      2. timm's non-persistent buffers (`relative_position_index`) are never
         written, so they hold uninitialized memory — the first forward pass
         dies with an out-of-bounds IndexError in the relative position bias.

    Building the timm model directly also lets us pass `img_size`, which is the
    only way to run a `*_patch4_window7_224` Swin at a different resolution
    (timm asserts on the input size otherwise).
    """
    import timm

    kwargs = {
        "pretrained": pretrained,
        "features_only": True,
        "out_indices": tuple(out_indices),
        # Swin/ViT feature extractors emit NHWC by default; Deformable DETR's
        # input_proj is Conv2d and needs NCHW.
        "output_fmt": "NCHW",
    }
    if image_size is not None:
        kwargs["img_size"] = image_size

    net = None
    for drop in ([], ["img_size"], ["output_fmt"], ["img_size", "output_fmt"]):
        try:
            net = timm.create_model(backbone, **{k: v for k, v in kwargs.items() if k not in drop})
            break
        except TypeError:
            continue
    if net is None:
        raise RuntimeError(f"could not create timm backbone {backbone!r}")

    net = _ensure_nchw(net, image_size)

    # transformers 5.x: conv_encoder.model is the timm net; 4.x wraps it in a
    # TimmBackbone as .model._backbone. Handle both.
    encoder = model.model.backbone
    holder, attr = encoder, "model"
    inner = getattr(encoder, "model", None)
    if inner is not None and hasattr(inner, "_backbone"):
        holder, attr = inner, "_backbone"
    setattr(holder, attr, net)
    return model


def build_param_groups(model, lr, lr_backbone, weight_decay):
    """Separate (lower) learning rate for the pretrained backbone, as is
    standard practice for DETR-family models — helps stabilize the noisy
    early-training Hungarian-matching gradients called out as a risk in
    PROJECT.md."""
    lr, lr_backbone, weight_decay = float(lr), float(lr_backbone), float(weight_decay)
    backbone_params, other_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        else:
            other_params.append(p)
    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": other_params, "lr": lr, "weight_decay": weight_decay},
    ]
