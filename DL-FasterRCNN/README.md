# Faster R-CNN Baseline

End-to-end Faster R-CNN pipeline for the AgroPest-12 dataset. Everything runs from inside `DL-FasterRCNN/` and assumes the shared dataset tree under `dataset/{train,valid,test}/(images|labels)` (YOLO txt format).

## Training
1. (Optional) adjust constants in `train_frcnn.py` (`DATA_ROOT`, `NUM_CLASSES`, epochs, LR, batch size, device).
2. Run training:
   ```bash
   python train_frcnn.py
   ```
3. Checkpoint saved to `../results/fasterrcnn/fasterrcnn_final.pth`.

## Evaluation
1. Ensure the trained checkpoint path in `evaluate_frcnn.py` matches the file produced above.
2. Run evaluation on the validation split:
   ```bash
   python evaluate_frcnn.py
   ```
3. Script prints per-class Precision / Recall / AP / F1 / Accuracy plus overall metrics. Requires `pandas` for nicely formatted tables.

## Single-Image Inference
```bash
python detect_frcnn.py --image /path/to/image.jpg \
                       --output fasterrcnn_detect.jpg \
                       --score-thresh 0.5
```
Outputs annotated image and logs the top-k detections with confidence scores.

## Tips
- Training defaults to `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")`. Adjust to other backbones if needed.
- Scripts automatically choose GPU (`cuda`) when available, otherwise CPU.
- To experiment with different datasets, update `DATA_ROOT` in all three scripts or add CLI arguments.

