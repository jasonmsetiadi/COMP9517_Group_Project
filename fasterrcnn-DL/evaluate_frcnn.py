# evaluate_frcnn.py
"""
Evaluate Faster R-CNN on AgroPest-12 using mAP@0.5.
"""

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

# Dataset paths
DATA_ROOT = "../data"
VAL_IMG = os.path.join(DATA_ROOT, "valid", "images")
VAL_LBL = os.path.join(DATA_ROOT, "valid", "labels")

NUM_CLASSES = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "../results/fasterrcnn/fasterrcnn_final.pth"

CLASS_NAMES = [
    "Background",
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
]


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
                labels.append(int(c) + 1)  # shift by +1

        img_tensor = F.to_tensor(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return img_tensor, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model():
    model = build_model(NUM_CLASSES)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def evaluate(model, loader, iou_thresh=0.5):
    all_preds = {i: [] for i in range(NUM_CLASSES)}
    all_gts = {i: [] for i in range(NUM_CLASSES)}

    for imgs, targets in tqdm(loader, desc="Evaluating"):
        imgs = [img.to(DEVICE) for img in imgs]

        with torch.no_grad():
            outputs = model(imgs)

        for out, gt in zip(outputs, targets):
            pred_boxes = out["boxes"].cpu().numpy().tolist()
            pred_scores = out["scores"].cpu().numpy().tolist()
            pred_labels = out["labels"].cpu().numpy().tolist()

            gt_boxes = gt["boxes"].numpy().tolist()
            gt_labels = gt["labels"].numpy().tolist()

            for b, s, c in zip(pred_boxes, pred_scores, pred_labels):
                if c < len(all_preds):
                    all_preds[c].append({"bbox": b, "score": s})

            for b, c in zip(gt_boxes, gt_labels):
                if c < len(all_gts):
                    all_gts[c].append({"bbox": b})

    aps = {}
    for c in range(1, NUM_CLASSES):  # skip background
        preds = sorted(all_preds[c], key=lambda x: x["score"], reverse=True)
        gts = all_gts[c]

        if len(gts) == 0:
            aps[CLASS_NAMES[c]] = 0.0
            continue

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = set()

        for i, p in enumerate(preds):
            best_iou = 0
            best_j = -1

            for j, g in enumerate(gts):
                iou_val = iou(p["bbox"], g["bbox"])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j

            if best_iou >= iou_thresh and best_j not in matched:
                tp[i] = 1
                matched.add(best_j)
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / len(gts)
        precisions = tp_cum / (tp_cum + fp_cum)

        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0
            ap += p / 11

        aps[CLASS_NAMES[c]] = ap

    mean_ap = np.mean(list(aps.values()))
    return mean_ap, aps


def main():
    ds = YOLODataset(VAL_IMG, VAL_LBL)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = load_model()

    mean_ap, aps = evaluate(model, loader)

    print("\nmAP@0.5 =", round(mean_ap, 4))
    print("\nPer-class AP:")
    for cls, ap in aps.items():
        print(f"{cls:15s}: {ap:.4f}")


if __name__ == "__main__":
    main()
