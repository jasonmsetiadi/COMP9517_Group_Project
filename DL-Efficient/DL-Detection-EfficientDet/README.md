# EfficientDet for Insect Detection

**COMP9517 Group Project - Deep Learning Method**

End-to-end pipeline for training and evaluating EfficientDet on agricultural insect images. Scripts are intended to be run from the project root.

**Performance:** 40.3% mAP, 89.0% recall, 391 ms/image

---

## Prerequisites

* Dataset organized with `images/` and `labels/` subdirectories under `dataset/train`, `dataset/valid`, and `dataset/test`
* Labels in YOLO format (`.txt` files: `class_id x_center y_center width height`)
* Python dependencies from `requirements.txt` installed in your environment
* Update dataset paths in `config.py` to match your directory structure

---

## Training Pipeline

**1. Train the model**
```bash
python train_efficientdet.py
```

Trains EfficientDet-D0 for 50 epochs on 11,502 images with ImageNet pre-trained backbone. Saves best model to `outputs_EfficientDet/checkpoints/efficientdet_best.pth` and training time to `efficientdet_training_time.json`.

---

## Evaluation Pipeline

**1. Run evaluation on test set**
```bash
python evaluate_efficientdet.py outputs_EfficientDet/checkpoints/efficientdet_best.pth test
```

Evaluates on 546 test images, computing detection metrics (mAP, precision, recall, F1) and classification metrics (accuracy, precision, recall, F1).

**Outputs:**
- `efficientdet_test_results.csv` - Per-class performance table
- `efficientdet_test_results.json` - Complete metrics in JSON format
- `efficientdet_test_timing.csv` - Inference timing statistics
- `efficientdet_test_confusion_matrix.png` - Classification confusion matrix

**2. (Optional) Evaluate on validation set**
```bash
python evaluate_efficientdet.py outputs_EfficientDet/checkpoints/efficientdet_best.pth valid
```

Generates `efficientdet_validation_*` output files.

---

## Configuration

Key settings in `config.py`:
```python
IMAGE_SIZE = 512          # Input resolution
NUM_CLASSES = 12          # 12 insect types
BATCH_SIZE = 8            # Reduce if GPU memory limited
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
CONF_THRESHOLD = 0.3      # Detection confidence threshold
NMS_THRESHOLD = 0.5       # Non-Maximum Suppression threshold
```

---

## Results Summary

**Overall Performance (Test Set):**

| Metric | Detection | Classification* |
|--------|-----------|----------------|
| mAP@0.5 | 40.3% | — |
| Precision | 1.1% | 24.0% |
| Recall | 89.0% | 12.0% |
| F1 Score | 2.2% | 12.7% |
| Inference Time | 391 ms/image (2.56 FPS) | — |

*Classification metrics computed on correctly localized boxes (IoU ≥ 0.5)

**Best Classes:** Bees (50.1% AP), Wasps (49.0%), Moths (47.8%)  
**Worst Classes:** Earthworms (22.9% AP), Grasshoppers (25.1%)

See `efficientdet_test_results.csv` for complete per-class breakdown.

---

## Implementation Details

- **Architecture:** EfficientNet-B0 backbone + BiFPN + detection head
- **Training:** AdamW optimizer, focal loss + smooth L1 loss, 50 epochs
- **Augmentation:** Random horizontal flip, brightness/contrast adjustment (±20%)
- **Inference:** Confidence filtering (0.3) + NMS (0.5)

---


