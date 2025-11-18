"""
EfficientDet Evaluation - Compact Version
COMP9517 Group Project
"""
import torch, numpy as np, pandas as pd, json, time
from torch.utils.data import DataLoader
from tqdm import tqdm
from effdet import DetBenchPredict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
from dataset import YOLODataset, collate_fn
from model import create_model, load_checkpoint
from detect_effdet import postprocess_detections


def iou(b1, b2):
    """IoU between two boxes"""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter/union if union > 0 else 0


def compute_ap(dets, gts):
    """Compute AP for one class"""
    if not dets or not gts: return 0.0
    dets = sorted(dets, key=lambda x: x['score'], reverse=True)
    tp, fp, matched = np.zeros(len(dets)), np.zeros(len(dets)), set()
    
    for i, d in enumerate(dets):
        best_iou, best_j = max([(iou(d['bbox'], g['bbox']), j) for j, g in enumerate(gts)], default=(0, -1))
        if best_iou >= 0.5 and best_j not in matched:
            tp[i] = 1
            matched.add(best_j)
        else:
            fp[i] = 1
    
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    rec, prec = tp_cum/len(gts), tp_cum/(tp_cum+fp_cum)
    return sum([np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0 for t in np.linspace(0,1,11)])/11


def evaluate_efficientdet(ckpt, img_path, lbl_path, name="Test"):
    """Complete evaluation"""
    print(f"\n{'='*70}\nEFFICIENTDET EVALUATION - {name.upper()}\n{'='*70}\n")
    
    device = torch.device(config.DEVICE)
    train_model = create_model(pretrained=False).to(device)
    train_model = load_checkpoint(train_model, ckpt)
    train_model.eval()
    model = DetBenchPredict(train_model.model).to(device)
    model.eval()
    
    dataset = YOLODataset(img_path, lbl_path, config.IMAGE_SIZE, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Collect predictions
    all_preds = {i: [] for i in range(config.NUM_CLASSES)}
    all_gts = {i: [] for i in range(config.NUM_CLASSES)}
    inf_times = []
    matches = []  # For classification
    
    print("Running inference...")
    with torch.no_grad():
        for imgs, tgts in tqdm(loader):
            imgs = imgs.to(device)
            gt_boxes, gt_labels = tgts[0]['bbox'].numpy(), tgts[0]['cls'].numpy()
            
            for box, label in zip(gt_boxes, gt_labels):
                all_gts[int(label)].append({'bbox': box})
            
            start = time.time()
            dets = model(imgs)
            inf_times.append((time.time()-start)*1000)
            
            img_sz = (tgts[0]['img_size'][0].item(), tgts[0]['img_size'][1].item())
            boxes, scores, labels = postprocess_detections(dets, img_sz, config.IMAGE_SIZE, 
                                                          config.CONF_THRESHOLD, config.NMS_THRESHOLD)
            
            for box, score, label in zip(boxes, scores, labels):
                pred_cls = int(label)-1
                if 0 <= pred_cls < config.NUM_CLASSES:
                    all_preds[pred_cls].append({'bbox': box, 'score': score})
                    
                    # Find best GT match for classification
                    best_iou, best_gt = 0, -1
                    for gt_cls in range(config.NUM_CLASSES):
                        for gt in all_gts[gt_cls]:
                            cur_iou = iou(box, gt['bbox'])
                            if cur_iou > best_iou:
                                best_iou, best_gt = cur_iou, gt_cls
                    
                    if best_iou >= 0.5 and best_gt != -1:
                        matches.append((pred_cls, best_gt))
    
    print("\nComputing metrics...\n")
    
    # Per-class metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    per_class_aps = {}
    table_data = []
    
    for cls_id, cls_name in enumerate(config.CLASS_NAMES):
        ap = compute_ap(all_preds[cls_id], all_gts[cls_id])
        per_class_aps[cls_name] = ap
        
        # Calculate TP/FP/FN
        preds = sorted(all_preds[cls_id], key=lambda x: x['score'], reverse=True)
        gts = all_gts[cls_id]
        matched, tp, fp = set(), 0, 0
        
        for p in preds:
            best_iou, best_j = 0, -1
            for j, g in enumerate(gts):
                cur_iou = iou(p['bbox'], g['bbox'])
                if cur_iou > best_iou:
                    best_iou, best_j = cur_iou, j
            
            if best_iou >= 0.5 and best_j not in matched:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        
        fn = len(gts) - len(matched)
        total_tp, total_fp, total_fn = total_tp+tp, total_fp+fp, total_fn+fn
        
        prec = tp/(tp+fp) if tp+fp > 0 else 0
        rec = tp/(tp+fn) if tp+fn > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0
        
        table_data.append({
            'Class': cls_name,
            'Precision': f'{prec:.4f}',
            'Recall': f'{rec:.4f}',
            'AP': f'{ap:.4f}',
            'F1': f'{f1:.4f}',
            'Accuracy': f'{prec:.4f}'
        })
    
    mAP = np.mean(list(per_class_aps.values()))
    det_prec = total_tp/(total_tp+total_fp) if total_tp+total_fp > 0 else 0
    det_rec = total_tp/(total_tp+total_fn) if total_tp+total_fn > 0 else 0
    det_f1 = 2*det_prec*det_rec/(det_prec+det_rec) if det_prec+det_rec > 0 else 0
    
    # Classification metrics
    if matches:
        clf_preds = np.array([m[0] for m in matches])
        clf_gts = np.array([m[1] for m in matches])
        clf_acc = accuracy_score(clf_gts, clf_preds)
        clf_prec, clf_rec, clf_f1, _ = precision_recall_fscore_support(clf_gts, clf_preds, average='weighted', zero_division=0)
        clf_cm = confusion_matrix(clf_gts, clf_preds, labels=list(range(config.NUM_CLASSES)))
    else:
        clf_acc = clf_prec = clf_rec = clf_f1 = 0
        clf_cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES))
    
    # Add summary rows
    table_data.extend([
        {'Class': 'mAP', 'Precision': '-', 'Recall': '-', 'AP': f'{mAP:.4f}', 'F1': '-', 'Accuracy': '-'},
        {'Class': 'Overall (Detection)', 'Precision': f'{det_prec:.4f}', 'Recall': f'{det_rec:.4f}', 
         'AP': '-', 'F1': f'{det_f1:.4f}', 'Accuracy': f'{det_prec:.4f}'},
        {'Class': 'Overall (Classification)', 'Precision': f'{clf_prec:.4f}', 'Recall': f'{clf_rec:.4f}', 
         'AP': '-', 'F1': f'{clf_f1:.4f}', 'Accuracy': f'{clf_acc:.4f}'}
    ])
    
    df = pd.DataFrame(table_data)
    
    # Print
    print(f"{'='*70}\nRESULTS TABLE\n{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}\n")
    
    # Save files
    avg_time = np.mean(inf_times)
    prefix = f'efficientdet_{name.lower()}'
    
    df.to_csv(f'{prefix}_results.csv', index=False)
    
    with open(f'{prefix}_results.json', 'w') as f:
        json.dump({
            'mAP': float(mAP),
            'detection': {'precision': float(det_prec), 'recall': float(det_rec), 'f1': float(det_f1)},
            'classification': {'accuracy': float(clf_acc), 'precision': float(clf_prec), 
                              'recall': float(clf_rec), 'f1': float(clf_f1)},
            'timing': {'avg_ms': float(avg_time), 'total_s': float(sum(inf_times)/1000), 'fps': float(1000/avg_time)}
        }, f, indent=2)
    
    pd.DataFrame([
        {'Metric': 'Avg Inference (ms)', 'Value': avg_time},
        {'Metric': 'Total Time (s)', 'Value': sum(inf_times)/1000},
        {'Metric': 'FPS', 'Value': 1000/avg_time}
    ]).to_csv(f'{prefix}_timing.csv', index=False)
    
    # Confusion matrix
    if matches:
        plt.figure(figsize=(10, 8))
        sns.heatmap(clf_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
        plt.title(f'EfficientDet Classification - {name}\nAccuracy: {clf_acc:.4f}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Results: {prefix}_results.csv/json")
    print(f"✓ Timing: {prefix}_timing.csv")
    print(f"✓ Confusion matrix: {prefix}_confusion_matrix.png\n")
    
    return df


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate_efficientdet.py <checkpoint> [valid/test]")
        sys.exit(1)
    
    ckpt = sys.argv[1]
    dtype = sys.argv[2] if len(sys.argv) > 2 else 'test'
    
    if dtype.lower() == 'test':
        imgs, lbls, name = config.TEST_IMAGES, config.TEST_LABELS, "Test"
    else:
        imgs, lbls, name = config.VALID_IMAGES, config.VALID_LABELS, "Validation"
    
    evaluate_efficientdet(ckpt, imgs, lbls, name)