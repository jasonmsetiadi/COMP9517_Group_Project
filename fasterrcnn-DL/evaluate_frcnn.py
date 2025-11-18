# evaluate_frcnn.py

"""
Evaluate Faster R-CNN on AgroPest-12 validation set using mAP@0.5.
Prints per-class Precision, Recall, AP, F1, Accuracy and overall scores.
"""


import os
import cv2
import time
import json
import numpy as np
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# Paths
DATA_ROOT = "../data"
VAL_IMG = os.path.join(DATA_ROOT, "valid", "images")
VAL_LBL = os.path.join(DATA_ROOT, "valid", "labels")

# Settings
NUM_CLASSES = 13   # 0 = background, 1–12 = actual classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "../results/fasterrcnn/fasterrcnn_final.pth"
IOU_THRESH = 0.5

CLASS_NAMES = [
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
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        lbl_path = os.path.join(self.lbl_dir, fname.replace(".jpg", ".txt"))

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
            b = torch.zeros((0, 4), dtype=torch.float32)
        else:
            b = torch.tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": b,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
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


def box_iou(box1, box2):
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter + 1e-8
    return inter / union


def evaluate():
    start_time = time.time()

    model = load_model()
    dataset = YOLODataset(VAL_IMG, VAL_LBL)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    preds_by_class = {c: [] for c in range(1, NUM_CLASSES)}
    gt_counts = {c: 0 for c in range(1, NUM_CLASSES)}
    gt_meta = []

    # for confusion matrix
    y_true = []
    y_pred = []

    for img_idx, (imgs, targets) in enumerate(tqdm(loader, desc="Evaluating")):
        img = imgs[0].to(DEVICE)
        target = targets[0]

        gtb = target["boxes"].numpy()
        gtl = target["labels"].numpy()
        per_image = {}

        # ground truth count
        for c in range(1, NUM_CLASSES):
            mask = (gtl == c)
            boxes_c = gtb[mask]
            gt_counts[c] += len(boxes_c)
            per_image[c] = {"boxes": boxes_c, "matched": np.zeros(len(boxes_c), bool)}
        gt_meta.append(per_image)

        # model prediction
        with torch.no_grad():
            out = model([img])[0]

        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()

        # confusion matrix predicted class
        if len(scores) > 0:
            top_idx = np.argmax(scores)
            top_label = int(labels[top_idx])
            if top_label > 0:
                y_pred.append(top_label - 1)
            else:
                y_pred.append(-1)
        else:
            y_pred.append(-1)

        # confusion matrix true class
        if len(gtl) > 0:
            y_true.append(int(gtl[0] - 1))
        else:
            y_true.append(-1)

        # store predictions for AP calculation
        for box, score, label in zip(boxes, scores, labels):
            label = int(label)
            if label == 0 or label >= NUM_CLASSES:
                continue
            preds_by_class[label].append(
                {"image_idx": img_idx, "box": box, "score": float(score)}
            )

    # per class metrics
    rows = []
    ap_values = []
    total_tp = total_fp = total_fn = 0

    for c in range(1, NUM_CLASSES):
        preds = preds_by_class[c]
        n_gt = gt_counts[c]

        if n_gt == 0:
            rows.append({"Class": CLASS_NAMES[c-1], "Precision": 0,
                         "Recall": 0, "AP": np.nan, "F1": 0, "Accuracy": 0})
            continue

        if len(preds) == 0:
            total_fn += n_gt
            rows.append({"Class": CLASS_NAMES[c-1], "Precision": 0,
                         "Recall": 0, "AP": 0, "F1": 0, "Accuracy": 0})
            ap_values.append(0)
            continue

        preds = sorted(preds, key=lambda x: x["score"], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, p in enumerate(preds):
            img_idx = p["image_idx"]
            box = p["box"]

            gts = gt_meta[img_idx][c]["boxes"]
            matched = gt_meta[img_idx][c]["matched"]

            if len(gts) == 0:
                fp[i] = 1
                continue

            ious = box_iou(box, gts)
            best = int(np.argmax(ious))

            if ious[best] >= IOU_THRESH and not matched[best]:
                tp[i] = 1
                matched[best] = True
            else:
                fp[i] = 1

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        recalls = tp_c / (n_gt + 1e-8)
        precisions = tp_c / (tp_c + fp_c + 1e-8)

        # AP
        ap = 0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if np.any(mask):
                ap += np.max(precisions[mask]) / 11
        ap_values.append(ap)

        tp_final = tp_c[-1]
        fp_final = fp_c[-1]
        fn_final = n_gt - tp_final

        total_tp += tp_final
        total_fp += fp_final
        total_fn += fn_final

        prec = tp_final / (tp_final + fp_final + 1e-8)
        rec = tp_final / (n_gt + 1e-8)
        f1 = (2 * prec * rec) / (prec + rec + 1e-8)
        acc = tp_final / (tp_final + fp_final + fn_final + 1e-8)

        rows.append({
            "Class": CLASS_NAMES[c-1],
            "Precision": prec,
            "Recall": rec,
            "AP": ap,
            "F1": f1,
            "Accuracy": acc,
        })

    # summarise
    mAP = float(np.nanmean(ap_values))
    overall_prec = total_tp / (total_tp + total_fp + 1e-8)
    overall_rec = total_tp / (total_tp + total_fn + 1e-8)
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec + 1e-8)
    overall_acc = total_tp / (total_tp + total_fp + total_fn + 1e-8)

    rows.append({"Class": "mAP", "Precision": np.nan, "Recall": np.nan,
                 "AP": mAP, "F1": np.nan, "Accuracy": np.nan})
    rows.append({"Class": "Overall", "Precision": overall_prec, "Recall": overall_rec,
                 "AP": mAP, "F1": overall_f1, "Accuracy": overall_acc})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # save csv
    df.to_csv("evaluation_results.csv", index=False)

    # confusion matrix
    yt = np.array(y_true)
    yp = np.array(y_pred)
    valid = yt >= 0
    if np.sum(valid) > 0:
        cm = confusion_matrix(yt[valid], yp[valid], labels=range(12))
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(12), CLASS_NAMES, rotation=45, ha="right")
        plt.yticks(range(12), CLASS_NAMES)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")

    # json
    total_time = time.time() - start_time
    summary = {
        "mAP": mAP,
        "precision": float(overall_prec),
        "recall": float(overall_rec),
        "f1": float(overall_f1),
        "accuracy": float(overall_acc),
        "timing": {
            "total_s": total_time,
            "avg_ms": (total_time / len(dataset)) * 1000,
            "fps": len(dataset) / total_time
        }
    }

    with open("evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    evaluate()
