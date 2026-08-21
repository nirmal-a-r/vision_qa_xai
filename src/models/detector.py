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

from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection


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
):
    """Returns a DeformableDetrForObjectDetection with a Swin backbone,
    randomly-initialized detection/segmentation heads sized for
    `num_classes`, loaded on top of the pretrained Deformable DETR
    transformer weights where shapes match.
    """
    config = DeformableDetrConfig.from_pretrained(base_checkpoint)

    # --- swap backbone: ResNet -> Swin (timm) -----------------------------
    config.use_timm_backbone = use_timm_backbone
    config.backbone = backbone
    # NOTE: don't set "features_only" here — transformers' DeformableDetrConvEncoder
    # already passes features_only=True itself when calling timm.create_model,
    # so including it in backbone_kwargs raises "multiple values for keyword
    # argument 'features_only'". out_indices is enough to select which Swin
    # stages feed the multi-scale deformable attention (stages 1,2,3 ->
    # strides 8/16/32, matching the 4-stage pyramid PROJECT.md calls for).
    config.backbone_kwargs = {"out_indices": (1, 2, 3)}
    config.use_pretrained_backbone = pretrained_backbone
    config.dilation = False

    # --- task-specific heads ----------------------------------------------
    config.num_labels = num_classes
    config.num_queries = num_queries

    # --- loss hyperparameters (Focal + GIoU per PROJECT.md) ----------------
    config.focal_alpha = focal_alpha
    config.class_cost = class_cost
    config.bbox_cost = bbox_cost
    config.giou_cost = giou_cost
    config.bbox_loss_coefficient = bbox_cost
    config.giou_loss_coefficient = giou_cost

    model = DeformableDetrForObjectDetection.from_pretrained(
        base_checkpoint,
        config=config,
        ignore_mismatched_sizes=True,  # class head shape changes with num_classes
    )
    return model, config


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
