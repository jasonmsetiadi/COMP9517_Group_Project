# Scripts Workflow

This directory houses the end-to-end pipeline for training and evaluating the object detection system. The scripts are intended to be run from the project root.

## Prerequisites
- Training data under `dataset/train` and test data under `dataset/test`, each with `images`, `labels`, and (after step 1) `proposals`.
- Python dependencies from `requirements.txt` installed in your environment.

## Training Pipeline
1. **Generate region proposals for train images**
   ```bash
   python scripts/run_selective_search.py --mode train
   ```
   Saves `*.txt` proposal files to `dataset/train/proposals`.

2. **Compute IoU overlap with ground truth**
   ```bash
   python scripts/train_compute_iou.py
   ```
   Produces `*.npy` overlap matrices in `dataset/train/overlap`.

3. **Prepare per-class training sets**
   ```bash
   python scripts/prepare_training_set.py
   ```
   Creates a timestamped run directory at `runs/run_<timestamp>/classes/`, storing per-class `data.npy` and `labels.npy`. Record the generated `<timestamp>`; it becomes the `run_id` for downstream steps.

4. **Train binary SVMs**
   ```bash
   python scripts/train_binary_svm.py --run_id <timestamp>
   ```
   Loads the prepared features and writes `svm.pkl` models to `runs/run_<timestamp>/classes/class_<id>/`.

## Prediction & Evaluation Pipeline
1. **Generate region proposals for test images**
   ```bash
   python scripts/run_selective_search.py --mode test
   ```
   Stores proposals in `dataset/test/proposals`.

2. **Run test-time predictions**
   ```bash
   python scripts/run_test_predictions.py --run_id <timestamp>
   ```
   Loads the trained SVMs for the specified `run_id` and writes detection arrays to `runs/run_<timestamp>/predictions/`.

3. **Compute evaluation metrics**
   ```bash
   python scripts/compute_metrics.py --run_id <timestamp>
   ```
   Aggregates metrics (mAP, precision, recall, F1, accuracy, AUC), printing them to stdout and saving them to `runs/run_<timestamp>/metrics.csv`.

Re-run individual stages as needed; each stage reads from the previous stage's outputs.

