import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import csv
from sklearn.metrics import roc_auc_score

from .utils import compute_iou, BBox


def compute_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute Average Precision (AP) from precision and recall arrays.
    AP is the area under the precision-recall curve.
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
        
    Returns:
        Average Precision score
    """
    # Add sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_detection_metrics(
    detections: List[List],
    ground_truths: Dict[str, List[List]],
    iou_threshold: float = 0.5,
    num_classes: int = 12
) -> Dict[int, float]:
    """
    Compute detection performance (mAP) for each class.
    
    Args:
        detections: List of [image_id, class_id, x_min, y_min, x_max, y_max, confidence]
        ground_truths: Dict mapping image_id to list of [class_id, x_min, y_min, x_max, y_max]
        iou_threshold: IoU threshold for matching (default 0.5)
        num_classes: Number of classes (default 12)
        
    Returns:
        Dict mapping class_id to AP score
    """
    # Group predictions by class, sorted by confidence (descending)
    predictions_by_class = defaultdict(list)
    for det in detections:
        image_id, class_id, x_min, y_min, x_max, y_max, confidence = det
        predictions_by_class[class_id].append({
            'image_id': image_id,
            'box': (x_min, y_min, x_max, y_max),
            'confidence': confidence
        })
    
    # Sort predictions by confidence (descending) for each class
    for class_id in predictions_by_class:
        predictions_by_class[class_id].sort(key=lambda x: x['confidence'], reverse=True)
    
    # Group ground truth by class
    gt_by_class = defaultdict(lambda: defaultdict(list))
    for image_id, gt_boxes in ground_truths.items():
        for gt_box in gt_boxes:
            class_id, x_min, y_min, x_max, y_max = gt_box
            gt_by_class[class_id][image_id].append((x_min, y_min, x_max, y_max))
    
    # Compute AP for each class
    ap_scores = {}
    for class_id in range(num_classes):
        if class_id not in predictions_by_class and class_id not in gt_by_class:
            ap_scores[class_id] = 0.0
            continue
        
        # Get predictions and ground truth for this class
        predictions = predictions_by_class[class_id]
        gt_boxes = gt_by_class[class_id]
        
        if len(predictions) == 0:
            # No predictions for this class
            ap_scores[class_id] = 0.0
            continue
        
        if len(gt_boxes) == 0:
            # No ground truth for this class, all predictions are FP
            ap_scores[class_id] = 0.0
            continue
        
        # Track which ground truth boxes have been matched
        gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in gt_boxes.items()}
        
        # Lists to store TP/FP for each prediction
        tp_list = []
        fp_list = []
        
        # Process predictions in confidence order
        for pred in predictions:
            image_id = pred['image_id']
            pred_box = pred['box']
            
            if image_id not in gt_boxes:
                # No ground truth for this image, this is a FP
                tp_list.append(0)
                fp_list.append(1)
                continue
            
            # Compute IoU with all ground truth boxes for this image and class
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes[image_id]):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if this prediction is a TP or FP
            if best_iou >= iou_threshold and not gt_matched[image_id][best_gt_idx]:
                # True positive: IoU >= threshold and ground truth not matched yet
                tp_list.append(1)
                fp_list.append(0)
                gt_matched[image_id][best_gt_idx] = True
            else:
                # False positive: IoU < threshold or ground truth already matched
                tp_list.append(0)
                fp_list.append(1)
        
        # Compute cumulative TP and FP
        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)
        
        # Total number of ground truth boxes for this class
        total_gt = sum(len(boxes) for boxes in gt_boxes.values())
        
        # Compute precision and recall
        recalls = tp_cumsum / (total_gt + 1e-6)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP
        ap = compute_ap(precisions, recalls)
        ap_scores[class_id] = ap
    
    return ap_scores


def compute_classification_metrics(
    detections: List[List],
    ground_truths: Dict[str, List[List]],
    iou_threshold: float = 0.5,
    num_classes: int = 12
) -> Dict[str, Dict[int, float]]:
    """
    Compute classification performance metrics for each class.
    
    Args:
        detections: List of [image_id, class_id, x_min, y_min, x_max, y_max, confidence]
        ground_truths: Dict mapping image_id to list of [class_id, x_min, y_min, x_max, y_max]
        iou_threshold: IoU threshold for matching (default 0.5)
        num_classes: Number of classes (default 12)
        
    Returns:
        Dict with keys 'precision', 'recall', 'f1', 'accuracy', 'auc'
        Each value is a dict mapping class_id to metric value
    """
    # Group predictions by image
    predictions_by_image = defaultdict(list)
    for det in detections:
        image_id, class_id, x_min, y_min, x_max, y_max, confidence = det
        predictions_by_image[image_id].append({
            'class_id': class_id,
            'box': (x_min, y_min, x_max, y_max),
            'confidence': confidence
        })
    
    # For classification: match predictions to ground truth
    # Store matched pairs: (gt_class, pred_class, confidence)
    matched_pairs = []
    unmatched_gt = []  # List of (image_id, gt_class)
    
    for image_id, gt_boxes in ground_truths.items():
        preds = predictions_by_image.get(image_id, [])
        
        if len(preds) == 0:
            # No predictions, all ground truth are FN
            for gt_box in gt_boxes:
                gt_class = gt_box[0]
                unmatched_gt.append((image_id, gt_class))
            continue
        
        # Compute IoU matrix: preds x gt_boxes
        iou_matrix = np.zeros((len(preds), len(gt_boxes)))
        for i, pred in enumerate(preds):
            for j, gt_box in enumerate(gt_boxes):
                _, x_min, y_min, x_max, y_max = gt_box
                iou_matrix[i, j] = compute_iou(pred['box'], (x_min, y_min, x_max, y_max))
        
        # Match each ground truth box with the prediction with highest IoU
        # (greedy matching: each GT matched with at most one prediction)
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(preds)
        
        # Sort by IoU (descending) and match greedily
        matches = []
        for _ in range(min(len(preds), len(gt_boxes))):
            best_iou = -1
            best_pred_idx = -1
            best_gt_idx = -1
            
            for i in range(len(preds)):
                if pred_matched[i]:
                    continue
                for j in range(len(gt_boxes)):
                    if gt_matched[j]:
                        continue
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_pred_idx = i
                        best_gt_idx = j
            
            if best_iou >= iou_threshold:
                matches.append((best_pred_idx, best_gt_idx, best_iou))
                pred_matched[best_pred_idx] = True
                gt_matched[best_gt_idx] = True
            else:
                break
        
        # Record matched pairs
        for pred_idx, gt_idx, iou_val in matches:
            pred = preds[pred_idx]
            gt_box = gt_boxes[gt_idx]
            gt_class = gt_box[0]
            matched_pairs.append({
                'gt_class': gt_class,
                'pred_class': pred['class_id'],
                'confidence': pred['confidence']
            })
        
        # Record unmatched ground truth boxes
        for j, gt_box in enumerate(gt_boxes):
            if not gt_matched[j]:
                gt_class = gt_box[0]
                unmatched_gt.append((image_id, gt_class))
    
    # Compute TP, FP, FN for each class
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    # Process matched pairs
    for pair in matched_pairs:
        gt_class = pair['gt_class']
        pred_class = pair['pred_class']
        
        if gt_class == pred_class:
            # True positive
            tp[gt_class] += 1
        else:
            # False positive for predicted class, false negative for ground truth class
            fp[pred_class] += 1
            fn[gt_class] += 1
    
    # Process unmatched ground truth (all are FN)
    for _, gt_class in unmatched_gt:
        fn[gt_class] += 1
    
    # Compute metrics for each class
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    accuracy_dict = {}
    
    for class_id in range(num_classes):
        tp_val = tp[class_id]
        fp_val = fp[class_id]
        fn_val = fn[class_id]
        
        # Precision = TP / (TP + FP)
        precision = tp_val / (tp_val + fp_val + 1e-6)
        
        # Recall = TP / (TP + FN)
        recall = tp_val / (tp_val + fn_val + 1e-6)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Accuracy = TP / (TP + FP + FN) (TN is undefined)
        accuracy = tp_val / (tp_val + fp_val + fn_val + 1e-6)
        
        precision_dict[class_id] = precision
        recall_dict[class_id] = recall
        f1_dict[class_id] = f1
        accuracy_dict[class_id] = accuracy
    
    # Compute ROC AUC for each class (one-vs-rest)
    auc_dict = {}
    
    # Prepare data for ROC AUC: we need binary labels and scores for each class
    for class_id in range(num_classes):
        y_true = []
        y_scores = []
        
        # Add matched pairs
        for pair in matched_pairs:
            gt_class = pair['gt_class']
            pred_class = pair['pred_class']
            confidence = pair['confidence']
            
            # For one-vs-rest: label is 1 if gt_class == class_id, else 0
            label = 1 if gt_class == class_id else 0
            
            # Score for class_id:
            # - If prediction is for this class, use its confidence
            # - If prediction is for a different class, use (1 - confidence) as a proxy
            #   (lower confidence in the predicted class suggests lower confidence in other classes)
            if pred_class == class_id:
                score = confidence
            else:
                # If ground truth is this class but prediction is wrong, score should be low
                # If ground truth is not this class, score should also be low
                score = (1.0 - confidence) * 0.5  # Scale down to avoid extreme values
            
            y_true.append(label)
            y_scores.append(score)
        
        # Add unmatched ground truth (all are positive for their class, negative for others)
        for _, gt_class in unmatched_gt:
            label = 1 if gt_class == class_id else 0
            score = 0.0  # No prediction, so score is 0
            y_true.append(label)
            y_scores.append(score)
        
        # Compute AUC if we have both positive and negative samples
        if len(y_true) > 0 and len(set(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                # If all labels are the same, AUC is undefined
                auc = 0.5
        else:
            auc = 0.5  # Default to 0.5 if no variation
        
        auc_dict[class_id] = auc
    
    return {
        'precision': precision_dict,
        'recall': recall_dict,
        'f1': f1_dict,
        'accuracy': accuracy_dict,
        'auc': auc_dict
    }


def compute_all_metrics(
    detections: List[List],
    ground_truths: Dict[str, List[List]],
    iou_threshold: float = 0.5,
    class_names: List[str] = None,
    save_path: str = None
):
    """
    Evaluate detection and classification performance.
    
    Args:
        detections: List of [image_id, class_id, x_min, y_min, x_max, y_max, confidence]
        ground_truths: Dict mapping image_id to list of [class_id, x_min, y_min, x_max, y_max]
        iou_threshold: IoU threshold for matching (default 0.5)
        class_names: List of class names (default None, uses indices)
        save_path: Path to save metrics CSV file
    """
    num_classes = len(class_names) if class_names else 12
    
    # Compute detection metrics (mAP)
    ap_scores = compute_detection_metrics(
        detections, ground_truths, iou_threshold, num_classes
    )
    
    # Compute classification metrics
    cls_metrics = compute_classification_metrics(
        detections, ground_truths, iou_threshold, num_classes
    )
    
    # Prepare results
    results = []
    for class_id in range(num_classes):
        class_name = class_names[class_id] if class_names else f"Class_{class_id}"
        results.append({
            'class': class_name,
            'ap': ap_scores[class_id],
            'precision': cls_metrics['precision'][class_id],
            'recall': cls_metrics['recall'][class_id],
            'f1': cls_metrics['f1'][class_id],
            'auc': cls_metrics['auc'][class_id],
            'accuracy': cls_metrics['accuracy'][class_id]
        })
    
    # Compute averages
    avg_ap = np.mean([r['ap'] for r in results])
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_auc = np.mean([r['auc'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    
    results.append({
        'class': 'Average',
        'ap': avg_ap,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'auc': avg_auc,
        'accuracy': avg_accuracy
    })
    
    # Save to CSV if path provided
    if save_path:
        with open(save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Class', 'AP@0.5', 'Precision', 'Recall', 'F1', 'AUC', 'Accuracy'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'Class': r['class'],
                    'AP@0.5': r['ap'],
                    'Precision': r['precision'],
                    'Recall': r['recall'],
                    'F1': r['f1'],
                    'AUC': r['auc'],
                    'Accuracy': r['accuracy']
                })
    
    return results

