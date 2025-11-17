# detect_frcnn.py

"""
Run inference with Faster R-CNN on a single image.
"""

import os
import argparse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import cv2

NUM_CLASSES = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "../results/fasterrcnn/fasterrcnn_final.pth"

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


def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(ckpt_path):
    model = build_model(NUM_CLASSES)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, img_path, score_thresh=0.5, top_k=10):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(DEVICE)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    labels = outputs["labels"].cpu()

    num_det = len(scores)
    print(f"Total raw detections: {num_det}")

    if num_det > 0:
        order = torch.argsort(scores, descending=True)
        top_k = min(top_k, num_det)

        print(f"Top {top_k} predictions:")
        for rank in range(top_k):
            idx = order[rank].item()
            score = scores[idx].item()
            cls_id = int(labels[idx])
            box = boxes[idx].tolist()

            if cls_id == 0:
                name = "Background"
            else:
                name = CLASS_NAMES[cls_id - 1]

            x1, y1, x2, y2 = [round(v, 1) for v in box]
            print(f"{rank+1}. {name:12s}  score={score:.3f}  box=({x1}, {y1}, {x2}, {y2})")

    kept = 0
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        cls_id = int(label)
        if cls_id == 0:
            continue

        kept += 1
        x1, y1, x2, y2 = box.int().tolist()
        cls_name = CLASS_NAMES[cls_id - 1]

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_bgr,
            f"{cls_name} {score:.2f}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    print(f"Detections above threshold {score_thresh}: {kept}")
    return img_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="fasterrcnn_detect.jpg")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    result = run_inference(model, args.image, score_thresh=args.score_thresh)

    cv2.imwrite(args.output, result)


if __name__ == "__main__":
    main() 