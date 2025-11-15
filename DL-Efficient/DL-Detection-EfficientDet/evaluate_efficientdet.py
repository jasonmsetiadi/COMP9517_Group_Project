import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from effdet import DetBenchPredict
import time
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import YOLODataset, collate_fn
from model import create_model, load_checkpoint
from detect_effdet import postprocess_detections


def compute_iou(box1, box2):
    """IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_ap(detections, gts, iou_thresh=0.5):
    """Compute Average Precision"""
    if len(gts) == 0 or len(detections) == 0:
        return 0.0
    
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    matched = set()
    
    for i, det in enumerate(detections):
        best_iou, best_idx = 0, -1
        for j, gt in enumerate(gts):
            iou = compute_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou, best_idx = iou, j
        
        if best_iou >= iou_thresh and best_idx not in matched:
            tp[i] = 1
            matched.add(best_idx)
        else:
            fp[i] = 1
    
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(gts)
    precisions = tp_cum / (tp_cum + fp_cum)
    
    # Compute AP (11-point interpolation)
    ap = sum([np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0 
              for t in np.linspace(0, 1, 11)]) / 11
    return ap


def evaluate_efficientdet(checkpoint_path, images_path, labels_path, dataset_name="Test"):
    """Complete evaluation"""
    print(f"\n{'='*70}")
    print(f"EFFICIENTDET EVALUATION - {dataset_name.upper()} SET")
    print(f"{'='*70}\n")
    
    device = torch.device(config.DEVICE)
    
    # Load model
    print("Loading model...")
    train_model = create_model(pretrained=False).to(device)
    train_model = load_checkpoint(train_model, checkpoint_path)
    train_model.eval()
    model = DetBenchPredict(train_model.model).to(device)
    model.eval()
    
    # Load dataset
    dataset = YOLODataset(images_path, labels_path, config.IMAGE_SIZE, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Collect predictions
    all_preds = {i: [] for i in range(config.NUM_CLASSES)}
    all_gts = {i: [] for i in range(config.NUM_CLASSES)}
    inference_times = []
    matched_predictions = []  # For classification
    
    print("Running inference...")
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            gt_boxes = targets[0]['bbox'].numpy()
            gt_labels = targets[0]['cls'].numpy()
            
            # Store ground truths
            for box, label in zip(gt_boxes, gt_labels):
                all_gts[int(label)].append({'bbox': box})
            
            # Inference
            start = time.time()
            detections = model(images)
            inference_times.append((time.time() - start) * 1000)
            
            img_size = (targets[0]['img_size'][0].item(), targets[0]['img_size'][1].item())
            boxes, scores, labels = postprocess_detections(
                detections, img_size, config.IMAGE_SIZE,
                config.CONF_THRESHOLD, config.NMS_THRESHOLD
            )
            
            # Store predictions
            for box, score, label in zip(boxes, scores, labels):
                pred_class = int(label) - 1
                if 0 <= pred_class < config.NUM_CLASSES:
                    all_preds[pred_class].append({'bbox': box, 'score': score})
                    
                    # Find best matching GT for classification
                    best_iou, best_gt_class = 0, -1
                    for gt_class in range(config.NUM_CLASSES):
                        for gt in all_gts[gt_class]:
                            iou = compute_iou(box, gt['bbox'])
                            if iou > best_iou:
                                best_iou, best_gt_class = iou, gt_class
                    
                    if best_iou >= 0.5 and best_gt_class != -1:
                        matched_predictions.append((pred_class, best_gt_class))
    
    print("\nComputing metrics...\n")
    
    # Detection metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    per_class_ap = {}
    
    for class_id, class_name in enumerate(config.CLASS_NAMES):
        ap = compute_ap(all_preds[class_id], all_gts[class_id])
        per_class_ap[class_name] = ap
        
        # Count TP, FP, FN
        preds = sorted(all_preds[class_id], key=lambda x: x['score'], reverse=True)
        gts = all_gts[class_id]
        matched = set()
        
        for pred in preds:
            best_iou, best_idx = 0, -1
            for j, gt in enumerate(gts):
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            
            if best_iou >= 0.5 and best_idx not in matched:
                total_tp += 1
                matched.add(best_idx)
            else:
                total_fp += 1
        
        total_fn += len(gts) - len(matched)
    
    det_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    det_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0
    mAP = np.mean(list(per_class_ap.values()))
    
    # Classification metrics
    if len(matched_predictions) > 0:
        clf_preds = np.array([p[0] for p in matched_predictions])
        clf_gts = np.array([p[1] for p in matched_predictions])
        
        clf_accuracy = accuracy_score(clf_gts, clf_preds)
        clf_precision, clf_recall, clf_f1, _ = precision_recall_fscore_support(
            clf_gts, clf_preds, average='weighted', zero_division=0
        )
        clf_cm = confusion_matrix(clf_gts, clf_preds, labels=list(range(config.NUM_CLASSES)))
    else:
        clf_accuracy = clf_precision = clf_recall = clf_f1 = 0
        clf_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES))
    
    # Create results table
    table_data = []
    for class_name in config.CLASS_NAMES:
        table_data.append({
            'Class': class_name,
            'AP': per_class_ap[class_name]
        })
    
    table_data.append({'Class': 'mAP', 'AP': mAP})
    table_data.append({'Class': 'Detection', 'AP': f'P:{det_precision:.3f} R:{det_recall:.3f} F1:{det_f1:.3f}'})
    table_data.append({'Class': 'Classification', 'AP': f'Acc:{clf_accuracy:.3f} P:{clf_precision:.3f} R:{clf_recall:.3f} F1:{clf_f1:.3f}'})
    
    df = pd.DataFrame(table_data)
    
    # Print results
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"mAP: {mAP:.4f}")
    print(f"\nDetection  - P: {det_precision:.4f}, R: {det_recall:.4f}, F1: {det_f1:.4f}")
    print(f"Classification - Acc: {clf_accuracy:.4f}, P: {clf_precision:.4f}, R: {clf_recall:.4f}, F1: {clf_f1:.4f}")
    print(f"{'='*70}\n")
    
    # Save results
    avg_time = np.mean(inference_times)
    
    results = {
        'mAP': float(mAP),
        'detection': {'precision': float(det_precision), 'recall': float(det_recall), 'f1': float(det_f1)},
        'classification': {'accuracy': float(clf_accuracy), 'precision': float(clf_precision), 
                          'recall': float(clf_recall), 'f1': float(clf_f1)},
        'per_class_AP': {name: float(ap) for name, ap in per_class_ap.items()},
        'timing': {'avg_ms': float(avg_time), 'total_s': float(sum(inference_times)/1000), 
                   'fps': float(1000/avg_time)}
    }
    
    # Save files
    prefix = f'efficientdet_{dataset_name.lower()}'
    
    df.to_csv(f'{prefix}_results.csv', index=False)
    
    with open(f'{prefix}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    pd.DataFrame([
        {'Metric': 'Average Inference (ms)', 'Value': avg_time},
        {'Metric': 'Total Time (s)', 'Value': sum(inference_times)/1000},
        {'Metric': 'FPS', 'Value': 1000/avg_time}
    ]).to_csv(f'{prefix}_timing.csv', index=False)
    
    # Confusion matrix
    if len(matched_predictions) > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(clf_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        plt.title(f'EfficientDet Classification - {dataset_name} Set\nAccuracy: {clf_accuracy:.4f}')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(f'{prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Results saved: {prefix}_results.csv, {prefix}_results.json")
    print(f"✓ Timing saved: {prefix}_timing.csv")
    print(f"✓ Confusion matrix: {prefix}_confusion_matrix.png\n")
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_efficientdet.py <checkpoint> [valid/test]")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'test'
    
    if dataset_type.lower() == 'test':
        images, labels, name = config.TEST_IMAGES, config.TEST_LABELS, "Test"
    else:
        images, labels, name = config.VALID_IMAGES, config.VALID_LABELS, "Validation"
    
    evaluate_efficientdet(checkpoint, images, labels, name)