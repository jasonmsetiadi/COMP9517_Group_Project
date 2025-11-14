"""
Complete Classification Evaluation - EfficientNet
Outputs results table like detection version
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import json

import config_clf as config
from dataset_clf import InsectDataset
from train_clf import create_model


def evaluate_classification(checkpoint_path, data_dir, dataset_name="Test"):
    """Complete evaluation with table output"""
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION EVALUATION - {dataset_name.upper()} SET")
    print(f"{'='*70}\n")
    
    device = torch.device(config.DEVICE)
    
    # Load model
    print("Loading model...")
    model = create_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    # Load dataset
    dataset = InsectDataset(data_dir, augment=False)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Dataset: {len(dataset)} images\n")
    
    # Predict
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            start = time.time()
            outputs = model(images)
            batch_time = (time.time() - start) * 1000  # milliseconds
            inference_times.append(batch_time / images.size(0))  # ms per image
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print("\nComputing metrics...\n")
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Get classes present in data
    unique_labels = sorted(set(all_labels) | set(all_preds))
    class_names_present = [config.CLASS_NAMES[i] for i in unique_labels]
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=unique_labels, zero_division=0
    )
    
    # Overall weighted metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    
    # Per-class accuracy (correct predictions / total for that class)
    per_class_accuracy = []
    for label in unique_labels:
        class_mask = (all_labels == label)
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == label).sum() / class_mask.sum()
        else:
            class_acc = 0.0
        per_class_accuracy.append(class_acc)
    
    # Create results table
    table_data = []
    for i, label in enumerate(unique_labels):
        class_name = config.CLASS_NAMES[label]
        table_data.append({
            'Class': class_name,
            'Precision': per_class_precision[i],
            'Recall': per_class_recall[i],
            'AP': per_class_precision[i],  # For classification, AP ≈ Precision
            'F1': per_class_f1[i],
            'Accuracy': per_class_accuracy[i]
        })
    
    # Add mAP row (mean of per-class precision)
    mAP = np.mean(per_class_precision)
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
        'Accuracy': accuracy
    })
    
    df = pd.DataFrame(table_data)
    
    # Print results
    print(f"{'='*70}")
    print("RESULTS TABLE")
    print(f"{'='*70}")
    print(df.to_string(index=True))
    print(f"{'='*70}\n")
    
    # Save results
    csv_filename = f'classification_results_{dataset_name.lower()}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"✓ Results saved: {csv_filename}")
    
    # Save JSON
    avg_inference_ms = np.mean(inference_times)
    total_inference_s = np.sum(inference_times) / 1000
    
    json_data = {
        'dataset': dataset_name,
        'total_images': len(dataset),
        'mAP': float(mAP),
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'f1': float(overall_f1),
            'auc': float(auc)
        },
        'per_class_metrics': {
            config.CLASS_NAMES[unique_labels[i]]: {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i]),
                'accuracy': float(per_class_accuracy[i])
            }
            for i in range(len(unique_labels))
        },
        'timing': {
            'average_prediction_duration_per_image_seconds': float(avg_inference_ms / 1000),
            'total_prediction_duration_seconds': float(total_inference_s),
            'average_inference_ms': float(avg_inference_ms),
            'fps': float(1000 / avg_inference_ms)
        }
    }
    
    json_filename = f'classification_results_{dataset_name.lower()}.json'
    with open(json_filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Results saved: {json_filename}")
    
    # Save timing to separate CSV
    timing_df = pd.DataFrame([{
        'Metric': 'Average Prediction Duration (seconds/image)',
        'Value': float(avg_inference_ms / 1000)
    }, {
        'Metric': 'Total Prediction Duration (seconds)',
        'Value': float(total_inference_s)
    }, {
        'Metric': 'Average Inference (ms/image)',
        'Value': float(avg_inference_ms)
    }, {
        'Metric': 'FPS',
        'Value': float(1000 / avg_inference_ms)
    }, {
        'Metric': 'Total Images',
        'Value': len(dataset)
    }])
    
    timing_csv = f'classification_timing_{dataset_name.lower()}.csv'
    timing_df.to_csv(timing_csv, index=False)
    print(f"✓ Timing saved: {timing_csv}")
    
    # Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)
    
    # Calculate precision and recall per class
    cm_precision = cm.diagonal() / cm.sum(axis=0)
    cm_recall = cm.diagonal() / cm.sum(axis=1)
    
    # Create annotations
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            if i == j:  # Diagonal
                annotations[i, j] = f'{count}\nP: {cm_precision[j]:.2f}\nR: {cm_recall[i]:.2f}'
            else:
                annotations[i, j] = f'{count}'
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
               xticklabels=class_names_present,
               yticklabels=class_names_present,
               ax=ax, cbar_kws={'label': 'Count'},
               annot_kws={'fontsize': 9})
    
    ax.set_title(f'Confusion Matrix - {dataset_name} Set\nAccuracy: {accuracy:.4f}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label\n\nDiagonal cells show: Count, Precision (P), Recall (R)', 
                 fontsize=12)
    
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{dataset_name.lower()}_clf.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {cm_filename}\n")
    plt.close()
    
    return df


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate_classification_final.py <checkpoint_path> [valid/test]")
        print("\nExample:")
        print("  python evaluate_classification_final.py outputs_classification/checkpoints/best_model.pth test")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'test'
    
    if dataset_type.lower() == 'test':
        data_dir = config.TEST_DIR
        name = "Test"
    else:
        data_dir = config.VALID_DIR
        name = "Validation"
    
    evaluate_classification(checkpoint, data_dir, name)