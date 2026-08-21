"""
dataset.py
==========
Torch Dataset that reads a COCO json (as produced by merge_coco.py) and
returns tensors in the exact shape HF's DeformableDetrForObjectDetection
expects for training:

    pixel_values : FloatTensor [3, H, W]
    labels       : {
        "class_labels": LongTensor [num_boxes]
        "boxes":        FloatTensor [num_boxes, 4]   # normalized cx,cy,w,h in [0,1]
        "image_id":     LongTensor [1]
        "orig_size":    LongTensor [2]                # (h, w) before padding/resize, for eval
    }

Kept deliberately simple: one Albumentations pipeline handles resize + pad +
augment + normalize in pascal_voc box space, and this class does the final
xyxy(px) -> cxcywh(normalized) conversion HF wants.
"""

import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class DefectCocoDataset(Dataset):
    def __init__(self, coco_json: str, images_dir: str, transforms=None):
        self.coco = COCO(coco_json)
        self.images_dir = images_dir
        self.img_ids = sorted(self.coco.imgs.keys())
        self.transforms = transforms
        # COCO category ids are not guaranteed to be 0-indexed/contiguous;
        # HF wants contiguous 0..num_classes-1 class labels.
        cat_ids = sorted(self.coco.cats.keys())
        self.catid_to_label = {cid: i for i, cid in enumerate(cat_ids)}
        self.label_to_name = {i: self.coco.cats[cid]["name"] for cid, i in self.catid_to_label.items()}

    def __len__(self):
        return len(self.img_ids)

    @property
    def num_classes(self):
        return len(self.catid_to_label)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes_xyxy, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(self.catid_to_label[a["category_id"]])

        import numpy as np
        image_np = np.array(image)

        if self.transforms is not None:
            transformed = self.transforms(image=image_np, bboxes=boxes_xyxy, labels=labels)
            pixel_values = transformed["image"]  # tensor, C,H,W (ToTensorV2 already applied)
            boxes_xyxy = transformed["bboxes"]
            labels = transformed["labels"]
            _, H, W = pixel_values.shape
        else:
            pixel_values = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            H, W = image_np.shape[0], image_np.shape[1]

        # xyxy (pixel, post-resize) -> cxcywh normalized in [0,1]
        boxes_cxcywh = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            boxes_cxcywh.append([cx, cy, bw, bh])

        target = {
            "class_labels": torch.as_tensor(labels, dtype=torch.long),
            "boxes": torch.as_tensor(boxes_cxcywh, dtype=torch.float32).reshape(-1, 4),
            "image_id": torch.as_tensor([img_id]),
            "orig_size": torch.as_tensor([orig_h, orig_w]),
            "size": torch.as_tensor([H, W]),
        }
        return pixel_values, target


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return {"pixel_values": pixel_values, "labels": labels}
