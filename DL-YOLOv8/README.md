# YOLOv8 Experiments

This folder contains all code and artifacts for the Ultralytics YOLOv8 pipelines (baseline training, fine-tuning weak classes, continued training, test-time prediction, metric export, and timing studies).

## Layout
- `codes/` – launcher scripts
  - `train_continue_yolo.py` – resume/continue full training run
  - `train_tune_weakclasses.py` – over-sample weak classes and fine-tune
  - `predict_test_fulltry3.py` – run inference on the held-out test set
  - `csv_table_test_result.py` – export per-class metrics to CSV
  - `test_time.py` – measure end-to-end inference latency
- `weights/` – checkpoints such as `best.pt`
- `all_test_results/`, `some_correct_bounded_images/`, `some_Incorrect_bounded_images/` – qualitative outputs
- `table_by_classes/result_table.csv` – summarized class metrics

## Workflows
### Continue / Resume Baseline Training
```bash
python codes/train_continue_yolo.py
```
- Automatically picks up from `runs/debug/agropest_yolov8n_fulltry3/weights/{last,best}.pt`.
- Saves metrics, validation plots, and report images back into the same run directory.

### Fine-Tune Weak Classes
```bash
python codes/train_tune_weakclasses.py
```
- Reads class IDs from `data.yaml`, builds an oversampled list of weak-class images, and launches a short fine-tuning run (`NEW_RUN = agropest_yolov8n_fulltry3_tune1` by default).

### Predict Test Split
```bash
python codes/predict_test_fulltry3.py
```
- Loads `best.pt`, runs inference on all test images, and moves bounded images + YOLO txt outputs into `runs/predict/test_fulltry3/{bounded images,labels}`.

### Export Metrics
```bash
python codes/csv_table_test_result.py
```
- Calls `model.val(split="test")`, aggregates Precision/Recall/AP/F1/Accuracy per class, and saves `class_metrics_test.csv` alongside the YOLO validation artifacts.

### Measure Inference Time
```bash
python codes/test_time.py
```
- Counts images, measures total `model.val(split="test")` duration, and writes timing stats to `runs/detect/test_time/test_time_stats.json`.
