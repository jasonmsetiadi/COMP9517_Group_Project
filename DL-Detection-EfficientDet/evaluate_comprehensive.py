import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from effdet import DetBenchPredict 

import config
from dataset import YOLODataset, collate_fn
from model import create_model, load_checkpoint
from detect_effdet import postprocess_detections

def compute_iou(box1, box2):
    """IoU between two boxes"""
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


def evaluate(checkpoint_path, images_path, labels_path, dataset_name="Test"):
    """Comprehensive evaluation"""
    print(f"\n{'='*60}\n{dataset_name.upper()} SET EVALUATION\n{'='*60}\n")
    
    device = torch.device(config.DEVICE)
    print("Loading model for evaluation...")
    train_model = create_model(pretrained=False).to(device)
    train_model = load_checkpoint(train_model, checkpoint_path)
    train_model.eval()
    model = DetBenchPredict(train_model.model).to(device)
    model.eval()
    print("Created prediction model.")
    
    dataset = YOLODataset(images_path, labels_path, config.IMAGE_SIZE, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Images: {len(dataset)}\n")
    
    # --- NEW LOGIC ---
    # We will store all pairs for the confusion matrix
    # 'background' will be class_id = config.NUM_CLASSES (e.g., 12)
    cm_pred_labels = []
    cm_gt_labels = []
    
    total_tp, total_fp, total_fn = 0, 0, 0
    iou_thresh = 0.5 # Using standard 0.5 IoU
    
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            gt_boxes = targets[0]['bbox'].numpy()
            gt_labels = targets[0]['cls'].numpy().astype(int) # 0-11
            orig_size = (targets[0]['img_size'][0].item(), targets[0]['img_size'][1].item())
            
            detections = model(images) 
            
            boxes, scores, labels = postprocess_detections(
                detections, orig_size, config.IMAGE_SIZE,
                config.CONF_THRESHOLD, config.NMS_THRESHOLD
            )
            
            pred_labels = labels.astype(int) - 1 # 0-11
            pred_boxes = boxes
            
            gt_matched_idx = set()
            pred_matched_idx = set()
            
            # --- Match predictions to ground truths ---
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                # Find all potential matches
                potential_matches = []
                for i in range(len(pred_boxes)):
                    for j in range(len(gt_boxes)):
                        iou = compute_iou(pred_boxes[i], gt_boxes[j])
                        if iou > iou_thresh:
                            potential_matches.append((iou, i, j))
                
                # Sort by best IoU first
                potential_matches.sort(key=lambda x: x[0], reverse=True)
                
                # Greedy matching
                for iou, pred_idx, gt_idx in potential_matches:
                    if pred_idx in pred_matched_idx or gt_idx in gt_matched_idx:
                        continue # This pred or gt is already matched
                    
                    pred_matched_idx.add(pred_idx)
                    gt_matched_idx.add(gt_idx)
                    
                    # This is a detection with good IoU.
                    # Add its (Pred, True) pair to the matrix
                    cm_pred_labels.append(pred_labels[pred_idx])
                    cm_gt_labels.append(gt_labels[gt_idx])

                    if pred_labels[pred_idx] == gt_labels[gt_idx]:
                        total_tp += 1 # Correct class
                    else:
                        total_fp += 1 # Classification error
            
            # --- Find False Positives (unmatched preds) ---
            for i in range(len(pred_boxes)):
                if i not in pred_matched_idx:
                    # This is a localization error (detected 'background')
                    total_fp += 1
                    cm_pred_labels.append(pred_labels[i])
                    cm_gt_labels.append(config.NUM_CLASSES) # 'background' class

            # --- Find False Negatives (unmatched gts) ---
            for j in range(len(gt_boxes)):
                if j not in gt_matched_idx:
                    # This is a missed detection (predicted 'background')
                    total_fn += 1
                    cm_pred_labels.append(config.NUM_CLASSES) # 'background' class
                    cm_gt_labels.append(gt_labels[j])

    # --- DETECTION METRICS (This is what you wanted to see) ---
    print(f"\n{'='*60}\nDETECTION METRICS (IoU=0.5, Conf={config.CONF_THRESHOLD})\n{'='*60}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nTrue Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    
    # --- CLASSIFICATION METRICS (This section now just saves the matrix) ---
    if len(cm_pred_labels) > 0:
        print(f"\n{'='*60}\nCLASSIFICATION METRICS\n{'='*60}")
        print("✓ Generating normalized confusion matrix...")
        
        # Define labels and names including the new 'background' class
        class_labels = list(range(config.NUM_CLASSES + 1))
        class_names = config.CLASS_NAMES + ['background']
        
        cm_norm = confusion_matrix(cm_gt_labels, cm_pred_labels, 
                                   labels=class_labels,
                                   normalize='true') # Normalize by True Label (Recall)

        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    vmin=0, vmax=1) 
                    
        plt.title(f'Confusion Matrix (Normalized by True Label) - {dataset_name} Set')
        plt.ylabel('Predicted Label') 
        plt.xlabel('True Label')
        plt.tight_layout()
        
        filename = f'confusion_matrix_normalized_{dataset_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Normalized confusion matrix saved: {filename}")
        plt.close()
    
    # --- SUMMARY FOR REPORT (This is what you wanted to see) ---
    print(f"\n{'='*60}\nSUMMARY FOR REPORT\n{'='*60}")
    print(f"{dataset_name} Set Results:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Calculate accuracy *only on TPs + Classification Errors*
    # This is the "Accuracy: 1.0000" you saw, which is not very useful
    # We will remove it to avoid confusion.
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <checkpoint_path> [valid/test]")
        print("Example: python your_script_name.py outputs/checkpoints/best_model.pth test")
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
    
    evaluate(checkpoint, images, labels, name)