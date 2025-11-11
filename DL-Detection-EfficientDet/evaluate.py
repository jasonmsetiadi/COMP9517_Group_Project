"""
Evaluation script - compute mAP (FINAL FIX)
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from effdet import DetBenchPredict 
import sys # <-- Added import

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


# --- MODIFIED: Function now takes image/label paths ---
def evaluate_model(checkpoint_path, images_path, labels_path, dataset_name="Validation"):
    """Evaluate model and compute mAP"""
    device = torch.device(config.DEVICE)
    
    print(f"\n{'='*60}\nCalculating mAP on {dataset_name.upper()} SET\n{'='*60}\n")
    
    # --- Step 1: Load weights into training model ---
    print("Loading model for evaluation...")
    train_model = create_model(pretrained=False).to(device)
    train_model = load_checkpoint(train_model, checkpoint_path)
    train_model.eval()

    # --- Step 2: Create prediction model ---
    model = DetBenchPredict(train_model.model).to(device)
    model.eval()
    print("Created prediction model.")
    
    # --- MODIFIED: Uses the paths passed into the function (and fixes your bug) ---
    dataset = YOLODataset(images_path, labels_path, 
                          config.IMAGE_SIZE, augment=False)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    all_preds = {i: [] for i in range(config.NUM_CLASSES)}
    all_gts = {i: [] for i in range(config.NUM_CLASSES)}
    
    print(f"Evaluating on {len(dataset)} images...")
    
    for images, targets in tqdm(loader):
        images = images.to(device)
        
        with torch.no_grad():
            detections = model(images) 
        
        img_size_h, img_size_w = targets[0]['img_size'].numpy()
        
        boxes, scores, labels = postprocess_detections(
            detections, (img_size_h, img_size_w), 
            config.IMAGE_SIZE, config.CONF_THRESHOLD, config.NMS_THRESHOLD
        )
        
        # Store predictions
        for box, score, label in zip(boxes, scores, labels):
            fixed_label = int(label) - 1 # Fix 1-indexed to 0-indexed
            
            if fixed_label in all_preds:
                all_preds[fixed_label].append({'bbox': box, 'score': score})
        
        # Store ground truths
        gt_boxes = targets[0]['bbox'].numpy()
        gt_labels = targets[0]['cls'].numpy()
        for box, label in zip(gt_boxes, gt_labels):
            all_gts[int(label)].append({'bbox': box})
    
    # Compute AP per class
    aps = {}
    for class_id in range(config.NUM_CLASSES):
        ap = compute_ap(all_preds[class_id], all_gts[class_id])
        aps[config.CLASS_NAMES[class_id]] = ap
    
    # Compute mAP
    mean_ap = np.mean(list(aps.values()))
    
    print("\n" + "="*50)
    print(f"mAP@0.5: {mean_ap:.4f}")
    print("\nPer-class AP:")
    for name, ap in sorted(aps.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {ap:.4f}")
    print("="*50)
    
    return mean_ap, aps


# --- MODIFIED: This block now checks for 'test' or 'valid' ---
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <checkpoint_path> [valid/test]")
        print("Example (for final test report): python evaluate.py outputs/checkpoints/best_model.pth test")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'valid'

    if dataset_type.lower() == 'test':
        # Check config.py for test paths
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
    
    evaluate_model(checkpoint, images, labels, name)