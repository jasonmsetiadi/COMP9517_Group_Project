#train_frcnn.py

"""
Training Faster R-CNN on AgroPest-12 (YOLO format).
"""

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import cv2
import numpy as np
from tqdm import tqdm

# Dataset paths
DATA_ROOT = "../data"
TRAIN_IMG = os.path.join(DATA_ROOT, "train", "images")
TRAIN_LBL = os.path.join(DATA_ROOT, "train", "labels")
VAL_IMG = os.path.join(DATA_ROOT, "valid", "images")
VAL_LBL = os.path.join(DATA_ROOT, "valid", "labels")

# Training settings
NUM_CLASSES = 13
BATCH_SIZE = 4
EPOCHS = 3
LR = 0.0005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory
SAVE_DIR = "../results/fasterrcnn"
os.makedirs(SAVE_DIR, exist_ok=True)


class YOLODataset(Dataset):

    def __init__(self, img_dir, lbl_dir):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.img_dir, filename)
        lbl_path = os.path.join(self.lbl_dir, filename.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes, labels = [], []

        if os.path.exists(lbl_path):
            for line in open(lbl_path):
                parts = line.strip().split()
                if not parts:
                    continue
                c, xc, yc, bw, bh = map(float, parts)
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c) + 1)

        img_tensor = F.to_tensor(img)

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return img_tensor, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def load_data():
    train_ds = YOLODataset(TRAIN_IMG, TRAIN_LBL)
    val_ds = YOLODataset(VAL_IMG, VAL_LBL)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train():
    train_loader, _ = load_data()
    model = build_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    save_path = os.path.join(SAVE_DIR, "fasterrcnn_final.pth")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train()
