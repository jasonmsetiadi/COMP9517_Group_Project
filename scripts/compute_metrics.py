import os
import sys
import argparse
import numpy as np
import cv2
import csv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from mean_average_precision import MetricBuilder
from src.utils import load_label_data, yolo_to_iou_format, compute_iou
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def match_detections_to_ground_truth(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Match predicted boxes to ground truth boxes using IoU threshold.
    
    gt_boxes: (N_gt, 7)  → [x_min, y_min, x_max, y_max, class_id, difficult, crowd]
    pred_boxes: (N_pred, 6) → [x_min, y_min, x_max, y_max, class_id, confidence]
    
    Returns:
        y_true: list of matched GT class IDs
        y_pred: list of predicted class IDs
        matched_indices: list of (gt_idx, pred_idx)
    """
    y_true, y_pred = [], []
    matched_indices = []

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return y_true, y_pred, matched_indices

    gt_used = set()

    # Sort predictions by confidence descending (to prioritize higher confidence matches)
    pred_boxes = sorted(pred_boxes, key=lambda x: x[5], reverse=True)

    for pred_idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in gt_used:
                continue  # skip already matched ground truths

            iou = compute_iou(pred[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # If IoU exceeds threshold, consider it a valid match
        if best_iou >= iou_threshold:
            gt_used.add(best_gt_idx)
            matched_indices.append((best_gt_idx, pred_idx))
            y_true.append(int(gt_boxes[best_gt_idx][4]))
            y_pred.append(int(pred[4]))

    return y_true, y_pred, matched_indices


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", '-id', type=str, required=True)
    args = parser.parse_args()

    run_id = args.run_id
    RUN_DIR = os.path.join(PROJECT_ROOT, 'runs', f'run_{run_id}')
    TEST_PREDICTIONS_DIR = os.path.join(RUN_DIR, 'predictions')

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=12)
    test_data = load_label_data(os.path.join(PROJECT_ROOT, 'dataset', 'test'))

    y_true_all = []
    y_pred_all = []
    matched_indices_all = []

    # Loop through your test dataset
    for image_path, gt_boxes in test_data:
        img = cv2.imread(image_path)
        img_width, img_height = img.shape[:2]
        # gt_boxes shape: (N_gt, 7) # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        gt_boxes = [list(yolo_to_iou_format(box['box'], img_width, img_height)) + [box['label'], 0, 0] for box in gt_boxes]
        gt_boxes = np.array(gt_boxes)
        
        # pred_boxes shape: (N_pred, 6) # [xmin, ymin, xmax, ymax, class_id, confidence]
        pred_boxes = np.load(os.path.join(TEST_PREDICTIONS_DIR, os.path.basename(image_path).replace(".jpg", ".npy")))
        # If no predictions, create empty array with correct shape
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            print(f"No predictions or ground truth boxes for {image_path}")
            continue

        # Add this image
        metric_fn.add(pred_boxes, gt_boxes)

        # match detections to ground truth
        y_true, y_pred, matched_indices = match_detections_to_ground_truth(gt_boxes, pred_boxes)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
        matched_indices_all.extend(matched_indices)

    # After all images, compute mAP
    metrics = metric_fn.value(iou_thresholds=0.5)
    print(f"Final mAP: {metrics['mAP']}")

    # compute precision, recall, F1 score
    precision = precision_score(y_true_all, y_pred_all, average='macro')
    recall = recall_score(y_true_all, y_pred_all, average='macro')
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    accuracy = accuracy_score(y_true_all, y_pred_all)

    # convert y_true_all and y_pred_all to one-hot encoding
    y_true_all = label_binarize(y_true_all, classes=range(12))
    y_pred_all = label_binarize(y_pred_all, classes=range(12))
    auc = roc_auc_score(y_true_all, y_pred_all, multi_class='ovr')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")

    # save the metrics to a csv file
    with open(os.path.join(RUN_DIR, 'metrics.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['mAP', 'Precision', 'Recall', 'F1 score', 'Accuracy', 'AUC'])
        writer.writerow([metrics['mAP'], precision, recall, f1, accuracy, auc])