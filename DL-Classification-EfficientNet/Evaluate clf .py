"""
Evaluate classification model - ALL METRICS
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys

import config_clf as config
from dataset_clf import InsectDataset
from train_clf import create_model


def evaluate(checkpoint_path, data_dir, dataset_name="Test"):
    """Comprehensive evaluation"""
    print(f"\n{'='*60}\n{dataset_name.upper()} SET EVALUATION\n{'='*60}\n")
    
    device = torch.device(config.DEVICE)
    
    # Load model
    model = create_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded\n")
    
    # Load dataset
    dataset = InsectDataset(data_dir, augment=False)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Images: {len(dataset)}\n")
    
    # Predict
    all_preds = []
    all_labels = []
    all_probs = []
    inference_times = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            start = time.time()
            outputs = model(images)
            inference_times.append((time.time() - start) * 1000 / images.size(0))
            
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    print(f"\n{'='*60}\nMETRICS\n{'='*60}")
    
    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        print(f"AUC: {auc:.4f}")
    except:
        auc = 0
        print(f"AUC: N/A")
    
    # Per-class metrics
    print(f"\n{'='*60}\nPER-CLASS METRICS\n{'='*60}")
    print(classification_report(all_labels, all_preds, 
                                target_names=config.CLASS_NAMES,
                                zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=config.CLASS_NAMES,
               yticklabels=config.CLASS_NAMES)
    plt.title(f'Confusion Matrix - {dataset_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filename = f'confusion_matrix_{dataset_name.lower()}_clf.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {filename}")
    plt.close()
    
    # Timing
    avg_time = np.mean(inference_times)
    print(f"\n{'='*60}\nTIMING\n{'='*60}")
    print(f"Inference: {avg_time:.2f} ms/image")
    print(f"FPS: {1000/avg_time:.1f}")
    
    # Summary
    print(f"\n{'='*60}\nSUMMARY FOR REPORT\n{'='*60}")
    print(f"{dataset_name} Set Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Inference: {avg_time:.2f} ms/image")
    print(f"{'='*60}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'inference_ms': avg_time
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python evaluate_clf.py <checkpoint_path> [valid/test]")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    dataset_type = sys.argv[2] if len(sys.argv) > 2 else 'test'
    
    if dataset_type.lower() == 'test':
        data_dir = config.TEST_DIR
        name = "Test"
    else:
        data_dir = config.VALID_DIR
        name = "Validation"
    
    evaluate(checkpoint, data_dir, name)