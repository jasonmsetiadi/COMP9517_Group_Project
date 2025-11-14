"""
Complete Detection Evaluation - EfficientDet
Combines mAP + Precision/Recall/F1 + Table Output
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from effdet import DetBenchPredict
import sys
import json

import config
from dataset import YOLODataset, collate_fn
from model import create_model, load_checkpoint
from detect_effdet import postprocess_detections


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


def compute_ap(detections, ground_truths, iou_thresh=0.5):
    """Compute Average Precision"""
    if len(ground_truths) == 0:
        return 0.0
    if len(detections) == 0:
        return 0.0
    
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    gt_matched = set()
    
    for i, det in enumerate(detections):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            iou = calculate_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_thresh and best_gt_idx not in gt_matched:
            tp[i] = 1
            gt_matched.add(best_gt_idx)
        else:
            fp[i] = 1
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def compute_per_class_metrics(all_preds, all_gts, iou_thresh=0.5):
    """Compute per-class precision, recall, F1"""
    results = {}
    
    for class_id in range(config.NUM_CLASSES):
        class_name = config.CLASS_NAMES[class_id]
        preds = all_preds[class_id]
        gts = all_gts[class_id]
        
        if len(gts) == 0:
            results[class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ap': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0
            }
            continue
        
        if len(preds) == 0:
            results[class_name] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ap': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': len(gts)
            }
            continue
        
        # Sort predictions by score
        preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        tp = 0
        fp = 0
        gt_matched = set()
        
        for pred in preds_sorted:
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(gts):
                iou = calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_thresh and best_gt_idx not in gt_matched:
                tp += 1
                gt_matched.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(gts) - len(gt_matched)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute AP
        ap = compute_ap(preds, gts, iou_thresh)
        
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results


def evaluate_detection(checkpoint_path, images_path, labels_path, dataset_name="Test"):
    """Complete evaluation with table output"""
    print(f"\n{'='*70}")
    print(f"DETECTION EVALUATION - {dataset_name.upper()} SET")
    print(f"{'='*70}\n")
    
    device = torch.device(config.DEVICE)
    
    # Load model
    print("Loading model...")
    train_model = create_model(pretrained=False).to(device)
    train_model = load_checkpoint(train_model, checkpoint_path)
    train_model.eval()
    model = DetBenchPredict(train_model.model).to(device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Load dataset
    dataset = YOLODataset(images_path, labels_path, config.IMAGE_SIZE, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Collect predictions and ground truths
    all_preds = {i: [] for i in range(config.NUM_CLASSES)}
    all_gts = {i: [] for i in range(config.NUM_CLASSES)}
    
    print("Running inference...")
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            
            # Ground truth
            gt_boxes = targets[0]['bbox'].numpy()
            gt_labels = targets[0]['cls'].numpy()
            
            for box, label in zip(gt_boxes, gt_labels):
                all_gts[int(label)].append({'bbox': box})
            
            # Predictions
            detections = model(images)
            img_size = (targets[0]['img_size'][0].item(), targets[0]['img_size'][1].item())
            
            boxes, scores, labels = postprocess_detections(
                detections, img_size, config.IMAGE_SIZE,
                config.CONF_THRESHOLD, config.NMS_THRESHOLD
            )
            
            for box, score, label in zip(boxes, scores, labels):
                fixed_label = int(label) - 1  # Convert 1-indexed to 0-indexed
                if 0 <= fixed_label < config.NUM_CLASSES:
                    all_preds[fixed_label].append({'bbox': box, 'score': score})
    
    print("\nComputing metrics...\n")
    
    # Compute per-class metrics
    per_class_results = compute_per_class_metrics(all_preds, all_gts)
    
    # Compute overall metrics
    total_tp = sum([r['tp'] for r in per_class_results.values()])
    total_fp = sum([r['fp'] for r in per_class_results.values()])
    total_fn = sum([r['fn'] for r in per_class_results.values()])
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # Compute mAP
    mAP = np.mean([r['ap'] for r in per_class_results.values()])
    
    # Create results table
    table_data = []
    for class_id, class_name in enumerate(config.CLASS_NAMES):
        r = per_class_results[class_name]
        table_data.append({
            'Class': class_name,
            'Precision': r['precision'],
            'Recall': r['recall'],
            'AP': r['ap'],
            'F1': r['f1'],
            'Accuracy': r['precision']  # For detection, accuracy ≈ precision
        })
    
    # Add mAP row
    table_data.append({
        'Class': 'mAP',
        'Precision': np.nan,
        'Recall': np.nan,
        'AP': mAP,
        'F1': np.nan,
        'Accuracy': np.nan
    })
    
    # Add overall row
    table_data.append({
        'Class': 'Overall',
        'Precision': overall_precision,
        'Recall': overall_recall,
        'AP': np.nan,
        'F1': overall_f1,
        'Accuracy': overall_precision
    })
    
    df = pd.DataFrame(table_data)
    
    # Print results
    print(f"{'='*70}")
    print("RESULTS TABLE")
    print(f"{'='*70}")
    print(df.to_string(index=True))
    print(f"{'='*70}\n")
    
    # Save results
    csv_filename = f'detection_results_{dataset_name.lower()}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"✓ Results saved: {csv_filename}")
    
    # Save JSON
    json_data = {
        'dataset': dataset_name,
        'total_images': len(dataset),
        'mAP': float(mAP),
        'overall_metrics': {
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1)
        },
        'per_class_metrics': {
            name: {
                'precision': float(r['precision']),
                'recall': float(r['recall']),
                'ap': float(r['ap']),
                'f1': float(r['f1'])
            }
            for name, r in per_class_results.items()
        }
    }
    
    json_filename = f'detection_results_{dataset_name.lower()}.json'
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Results saved: {json_filename}\n")
    
    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate_detection_final.py <checkpoint_path> [valid/test]")
        print("\nExample:")
        print("  python evaluate_detection_final.py outputs/checkpoints/best_model.pth test")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'valid'
    
    if dataset_type.lower() == 'test':
        if not (hasattr(config, 'TEST_IMAGES') and hasattr(config, 'TEST_LABELS')):
            print("Error: TEST_IMAGES and TEST_LABELS not defined in config.py")
            sys.exit(1)
        images = config.TEST_IMAGES
        labels = config.TEST_LABELS
        name = "Test"
    else:
        images = config.VALID_IMAGES
        labels = config.VALID_LABELS
        name = "Validation"
    
    evaluate_detection(checkpoint, images, labels, name)