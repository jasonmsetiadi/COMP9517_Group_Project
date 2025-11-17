import numpy as np
import pandas as pd
from src.utils import compute_iou

def evaluate_class(detections, ground_truths, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x[-1], reverse=True)
    
    TP = []
    FP = []
    total_gts = sum(len(v) for v in ground_truths.values())
    
    gt_matched = {img_id: [False]*len(boxes) for img_id, boxes in ground_truths.items()}
    
    for det in detections:
        img_id, x1, y1, x2, y2, conf = det
        pred_box = [x1, y1, x2, y2]
        
        if img_id not in ground_truths:
            FP.append(1)
            TP.append(0)
            continue
        
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt_box in enumerate(ground_truths[img_id]):
            current_iou = compute_iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and not gt_matched[img_id][best_gt_idx]:
            TP.append(1)
            FP.append(0)
            gt_matched[img_id][best_gt_idx] = True
        else:
            TP.append(0)
            FP.append(1)

    return np.array(TP), np.array(FP), total_gts


def compute_classification_metrics(TP, FP, total_gts):
    tp = np.array(TP, dtype=float, copy=True)
    fp = np.array(FP, dtype=float, copy=True)
    cum_TP = np.cumsum(tp)
    cum_FP = np.cumsum(fp)
    cum_FN = np.maximum(total_gts - cum_TP, 0)

    precision = cum_TP / (cum_TP + cum_FP + 1e-6)
    recall = cum_TP / (total_gts + 1e-6)
    prec, rec = precision.copy(), recall.copy()
    f1 = (2 * prec * rec) / (prec + rec + 1e-6)
    accuracy = cum_TP / (cum_TP + cum_FP + cum_FN + 1e-6)

    recall_levels = np.linspace(0, 1, 11)
    precisions_interp = []
    for r in recall_levels:
        precisions_interp.append(np.max(precision[recall >= r]) if np.any(recall >= r) else 0)

    AP = np.mean(precisions_interp)

    return precision, recall, f1, accuracy, AP


def evaluate_map(detections, ground_truths, iou_threshold=0.5, class_names=None, save_path=None):
    classes = set()
    for gt_boxes in ground_truths.values():
        for box in gt_boxes:
            classes.add(box[0])
    for det in detections:
        classes.add(det[1])
    classes = sorted(classes)
    
    AP_per_class = {}
    precision_per_class = {}
    recall_per_class = {}
    
    results_rows = []
    
    total_tp_all = 0
    total_fp_all = 0
    total_fn_all = 0

    for cls in classes:
        cls_name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
        
        cls_gts = {}
        for img_id, boxes in ground_truths.items():
            cls_boxes = [box[1:] for box in boxes if box[0] == cls]
            if cls_boxes:
                cls_gts[img_id] = cls_boxes
        
        cls_dets = []
        for det in detections:
            img_id, class_id, x1, y1, x2, y2, conf = det
            if class_id == cls:
                cls_dets.append([img_id, x1, y1, x2, y2, conf])
        
        if len(cls_gts) == 0:
            AP_per_class[cls] = 0
            precision_per_class[cls] = np.array([])
            recall_per_class[cls] = np.array([])
            results_rows.append({
                "Class": cls_name,
                "Precision": 0,
                "Recall": 0,
                "AP": 0,
                "F1": 0,
                "Accuracy": 0
            })
            continue
        
        TP, FP, total_gts = evaluate_class(cls_dets, cls_gts, iou_threshold)
        precision, recall, f1_curve, accuracy_curve, AP = compute_classification_metrics(TP, FP, total_gts)

        AP_per_class[cls] = AP
        precision_per_class[cls] = precision
        recall_per_class[cls] = recall
        
        # Take last precision/recall values as representative (like YOLO does)
        prec_last = precision[-1] if len(precision) > 0 else 0
        rec_last = recall[-1] if len(recall) > 0 else 0
        f1_last = f1_curve[-1] if len(f1_curve) > 0 else 0
        accuracy_last = accuracy_curve[-1] if len(accuracy_curve) > 0 else 0
        tp_final = int(np.sum(TP))
        fp_final = int(np.sum(FP))
        fn_final = int(total_gts - tp_final)

        total_tp_all += tp_final
        total_fp_all += fp_final
        total_fn_all += fn_final
        
        results_rows.append({
            "Class": cls_name,
            "Precision": prec_last,
            "Recall": rec_last,
            "AP": AP,
            "F1": f1_last,
            "Accuracy": accuracy_last
        })
    
    mAP = np.mean(list(AP_per_class.values())) if AP_per_class else 0

    overall_precision = total_tp_all / (total_tp_all + total_fp_all + 1e-6) if (total_tp_all + total_fp_all) > 0 else 0
    overall_recall = total_tp_all / (total_tp_all + total_fn_all + 1e-6) if (total_tp_all + total_fn_all) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6)) if (overall_precision + overall_recall) > 0 else 0
    overall_accuracy = total_tp_all / (total_tp_all + total_fp_all + total_fn_all + 1e-6) if (total_tp_all + total_fp_all + total_fn_all) > 0 else 0

    # Add mAP row
    results_rows.append({
        "Class": "mAP",
        "Precision": np.nan,
        "Recall": np.nan,
        "AP": mAP,
        "F1": np.nan,
        "Accuracy": np.nan
    })

    # Add overall metrics row
    results_rows.append({
        "Class": "Overall",
        "Precision": overall_precision,
        "Recall": overall_recall,
        "AP": np.nan,
        "F1": overall_f1,
        "Accuracy": overall_accuracy
    })
    results_df = pd.DataFrame(results_rows)

    
    # Print table similar to YOLO output
    print("\nPerformance Metrics:")
    print(results_df.to_string(index=False, float_format="{:.3f}".format))
    
    # Save to CSV if requested
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"\nSaved metrics table to: {save_path}")
    
    return AP_per_class, precision_per_class, recall_per_class, mAP, results_df
