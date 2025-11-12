import argparse
import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.eval import evaluate_map
from src.utils import CLASS_TO_ID, load_label_data, yolo_to_iou_format

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", "-id", type=str, required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_id = args.run_id
    RUN_DIR = os.path.join(PROJECT_ROOT, "runs", f"run_{run_id}")
    TEST_PREDICTIONS_DIR = os.path.join(RUN_DIR, "predictions")

    class_names = [name for name, _ in sorted(CLASS_TO_ID.items(), key=lambda item: item[1])]
    ground_truths = {}
    detections = []

    test_data = load_label_data(os.path.join(PROJECT_ROOT, "dataset", "test"))

    for image_path, gt_boxes in test_data:
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: could not read image at {image_path}. Skipping.")
            continue

        img_height, img_width = img.shape[:2]

        gt_entries = []
        for box in gt_boxes:
            x_min, y_min, x_max, y_max = yolo_to_iou_format(box["box"], img_width, img_height)
            gt_entries.append([box["label"], x_min, y_min, x_max, y_max])

        ground_truths[image_id] = gt_entries

        pred_path = os.path.join(TEST_PREDICTIONS_DIR, f"{image_id}.npy")
        if not os.path.exists(pred_path):
            print(f"Warning: no predictions found for {image_id}.")
            continue

        pred_boxes = np.load(pred_path)
        if pred_boxes.size == 0:
            print(f"Warning: empty predictions for {image_id}.")
            continue

        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.reshape(1, -1)

        for pred in pred_boxes:
            if len(pred) < 6:
                print(f"Warning: prediction format invalid for {image_id}. Skipping entry.")
                continue

            x_min, y_min, x_max, y_max, class_id, confidence = pred[:6]
            detections.append(
                [
                    image_id,
                    int(class_id),
                    float(x_min),
                    float(y_min),
                    float(x_max),
                    float(y_max),
                    float(confidence),
                ]
            )

    metrics_path = os.path.join(RUN_DIR, "metrics.csv")
    evaluate_map(
        detections,
        ground_truths,
        iou_threshold=args.iou_threshold,
        class_names=class_names,
        save_path=metrics_path,
    )