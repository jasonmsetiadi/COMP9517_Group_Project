# COMP9517_Group_Project - EfficientDet Object Detection for AgroPest-12
The primary method implemented uses the EfficientDet-D0 object detection model, trained using PyTorch and the effdet library.

# Quick Start
Install
pip install -r requirements.txt
Train
python train_efficientdet.py
Detect
python detect_effdet.py outputs/checkpoints/best_model.pth test.jpg output.jpg
Evaluate
python evaluate.py outputs/checkpoints/best_model.pth

# 🚀 Features
Model: EfficientDet-D0 (easily configurable to other variants).
Framework: PyTorch.
Dataset: Handles standard YOLOv5 format (via dataset.py).
Training: Full training script with validation loss, checkpoint saving, and data augmentation via albumentations.
Evaluation: Calculates mAP@0.5 and per-class AP scores.
Inference: Script to run detection on single images and draw bounding boxes.

# Configuration
Model variant (d0, d1, d2, etc.)
Batch size, epochs, learning rate
Dataset paths
Detection thresholds