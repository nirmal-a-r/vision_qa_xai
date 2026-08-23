"""
train.py
========
Custom training loop (deliberately not HF Trainer, to keep full visibility
into the Hungarian-matching loss behaviour PROJECT.md flags as a training
risk) for Swin + Deformable DETR on the merged PCB+NEU COCO dataset.

Run from the notebook, or:
    python -m src.train --config configs/config.yaml
"""

import os
import time
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DefectCocoDataset, collate_fn
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.detector import build_model, build_param_groups


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            out = model(pixel_values=pixel_values, labels=labels)
            total += out.loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def train(config_path: str):
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    train_ds = DefectCocoDataset(
        coco_json=cfg["paths"]["coco_merged_json"].replace(".json", "_train.json"),
        images_dir=cfg["paths"]["images_merged_dir"],
        transforms=get_train_transforms(cfg["model"]["image_size"]),
    )
    val_ds = DefectCocoDataset(
        coco_json=cfg["paths"]["coco_merged_json"].replace(".json", "_val.json"),
        images_dir=cfg["paths"]["images_merged_dir"],
        transforms=get_val_transforms(cfg["model"]["image_size"]),
    )
    print(f"[train] train={len(train_ds)} val={len(val_ds)} num_classes={train_ds.num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=2, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    model, _ = build_model(
        num_classes=train_ds.num_classes,
        backbone=cfg["model"]["backbone"],
        use_timm_backbone=cfg["model"]["use_timm_backbone"],
        base_checkpoint=cfg["model"]["detector_checkpoint"],
        num_queries=cfg["model"]["num_queries"],
        focal_alpha=cfg["training"]["focal_alpha"],
        bbox_cost=cfg["training"]["bbox_weight"],
        giou_cost=cfg["training"]["giou_weight"],
        image_size=cfg["model"]["image_size"],
    )
    model.to(device)

    param_groups = build_param_groups(
        model, lr=float(cfg["training"]["lr"]), lr_backbone=float(cfg["training"]["lr_backbone"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"]["fp16"] and device == "cuda")

    total_steps = len(train_loader) * cfg["training"]["epochs"] // cfg["training"]["grad_accum_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in param_groups], total_steps=max(total_steps, 1),
        pct_start=min(0.1, cfg["training"]["warmup_steps"] / max(total_steps, 1)),
    )

    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    log_path = os.path.join(cfg["paths"]["logs_dir"], "train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,step,loss,lr,elapsed_s\n")

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(cfg["training"]["epochs"]):
        for i, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            with torch.cuda.amp.autocast(enabled=cfg["training"]["fp16"] and device == "cuda"):
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss / cfg["training"]["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (i + 1) % cfg["training"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step % 20 == 0:
                    elapsed = time.time() - t0
                    lr_now = optimizer.param_groups[-1]["lr"]
                    print(f"[epoch {epoch} step {step}] loss={out.loss.item():.4f} lr={lr_now:.2e} elapsed={elapsed:.0f}s")
                    with open(log_path, "a") as f:
                        f.write(f"{epoch},{step},{out.loss.item():.4f},{lr_now:.6f},{elapsed:.1f}\n")

        if (epoch + 1) % cfg["training"]["eval_every_epochs"] == 0:
            val_loss = evaluate_loss(model, val_loader, device)
            print(f"[epoch {epoch}] val_loss={val_loss:.4f}")

        if (epoch + 1) % cfg["training"]["save_every_epochs"] == 0 or (epoch + 1) == cfg["training"]["epochs"]:
            ckpt_path = os.path.join(cfg["paths"]["checkpoints_dir"], f"detector_epoch{epoch+1}.pt")
            torch.save({"model_state_dict": model.state_dict(),
                        "num_classes": train_ds.num_classes,
                        "label_to_name": train_ds.label_to_name,
                        "config": cfg}, ckpt_path)
            print(f"[train] saved {ckpt_path}")

    return model, train_ds.label_to_name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    train(args.config)
