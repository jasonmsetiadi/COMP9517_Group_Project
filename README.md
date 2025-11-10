# COMP9517_Group_Project - EfficientDet Object Detection for AgroPest-12
# Quick Start
# Install
pip install -r requirements.txt

# Train
python train_efficientdet.py

# Detect
python detect_effdet.py outputs/checkpoints/best_model.pth test.jpg output.jpg

# Evaluate
python evaluate.py outputs/checkpoints/best_model.pth

# Key Features
✓ YOLO format support
✓ Data augmentation
✓ Training with validation
✓ mAP evaluation
✓ Clean, readable code

# Configuration

Model variant (d0, d1, d2, etc.)
Batch size, epochs, learning rate
Dataset paths
Detection thresholds