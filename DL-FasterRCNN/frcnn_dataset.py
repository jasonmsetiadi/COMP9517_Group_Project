# frcnn_dataset.py

import os
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YoloToFasterRCNNDataset(Dataset):
    
    def __init__(self, images_dir: str, labels_dir: str, image_size: int = 512):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size

        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_files.sort()

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(
            self.labels_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        # Load image (BGR -> RGB)
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        # Resize to fixed size for training
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        sx = self.image_size / w0
        sy = self.image_size / h0

        boxes: List[List[float]] = []
        labels: List[int] = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c, xc, yc, w, h = map(float, line.split()[:5])
                    cls_id = int(c)

                    # YOLO normalized -> absolute in original image
                    xc_abs = xc * w0
                    yc_abs = yc * h0
                    w_abs = w * w0
                    h_abs = h * h0

                    x1 = (xc_abs - w_abs / 2) * sx
                    y1 = (yc_abs - h_abs / 2) * sy
                    x2 = (xc_abs + w_abs / 2) * sx
                    y2 = (yc_abs + h_abs / 2) * sy

                    # Clip to image bounds
                    x1 = max(0.0, min(self.image_size - 1, x1))
                    y1 = max(0.0, min(self.image_size - 1, y1))
                    x2 = max(0.0, min(self.image_size - 1, x2))
                    y2 = max(0.0, min(self.image_size - 1, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Faster R-CNN expects labels: 1..num_classes (0 is background)
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id + 1)

        if len(boxes) == 0:
            boxes_arr = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)
        else:
            boxes_arr = np.array(boxes, dtype=np.float32)
            labels_arr = np.array(labels, dtype=np.int64)

        # Convert image to tensor [C,H,W] in [0,1]
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.as_tensor(boxes_arr, dtype=torch.float32),
            "labels": torch.as_tensor(labels_arr, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            # Areas and iscrowd can be added if needed
        }

        return img_t, target


def detection_collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets
