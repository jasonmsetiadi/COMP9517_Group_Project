# COMP9517 Crop Pest Detection

A comprehensive comparison of machine learning and deep learning approaches for crop pest detection using the AgroPest-12 dataset. This project implements five different methods: two traditional ML approaches using multiple feature descriptors (HOG, SIFT, LBP, Color features) with SVM and Random Forest classifiers, and three deep learning models (Faster R-CNN, EfficientDet, YOLOv8).

## Team Members

| Name | Student ID | Responsibilities | Contact |
|------|-----------|------------------|---------|
| Jason | z5611110 | HOG+SVM | [jason.setiadi@unsw.edu.au] |
| Ben | z5360027 | HOG+RF | [b.laphai@student.unsw.edu.au] |
| Justine | z5423358 | Faster-RCNN | [siyeon.kim@student.unsw.edu.au] |
| Mike | z5698637 | EfficientDet | [xiaojun.guo@student.unsw.edu.au] |
| Michael | z5540434 | YOLOv8 | [feiyang.wang@student.unsw.edu.au] |

## Project Structure

```
COMP9517_Group_Project/
├── dataset/              # AgroPest-12 dataset (train/valid/test splits)
├── ML-HOG-SVM/          # Machine Learning: Multiple features (HOG, SIFT, LBP, Color) + SVM classifier
├── ML-HOG-RF/           # Machine Learning: Multiple features (HOG, SIFT, LBP, Color) + Random Forest classifier
├── DL-FasterRCNN/       # Deep Learning: Faster R-CNN detection model
├── DL-EfficientDet/     # Deep Learning: EfficientDet detection model
└── DL-YOLOv8/           # Deep Learning: YOLOv8 detection model
```

### Approach Descriptions

#### Machine Learning Approaches

- **ML-SVM**: Uses multiple feature descriptors (HOG, SIFT, LBP, and Color features) for feature extraction with Selective Search for region proposal generation, followed by binary SVM classifiers for each pest class.

- **ML-RF**: Similar to ML-SVM but uses Random Forest classifier instead. Employs sliding window detection with multiple feature descriptors (HOG, SIFT, LBP, and Color features) and trained RF models.

#### Deep Learning Approaches

- **DL-FasterRCNN**: Two-stage detector using ResNet backbone with Region Proposal Network (RPN) and Fast R-CNN head for classification and bounding box regression.

- **DL-EfficientDet**: Single-stage detector based on EfficientNet backbone with BiFPN feature pyramid network, optimized for efficiency and accuracy balance.

- **DL-YOLOv8**: State-of-the-art single-stage detector using YOLOv8 architecture (Ultralytics), providing fast inference with competitive accuracy.

## Dataset Setup

1. **Download the dataset** from [Kaggle: AgroPest-12 Dataset](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset).

2. **Organize the dataset** in the following structure:
   ```
   dataset/
   ├── data.yaml
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

3. **Label Format**: Labels are provided in YOLO format:
   - Each label file corresponds to an image file (`.txt` extension)
   - Each line in a label file follows: `class_id x_center y_center width height`
   - All values are normalized between 0 and 1, where:
     - `x_center` and `y_center` represent the center of the bounding box (as a fraction of image width and height, respectively)
     - `width` and `height` are the size of the bounding box (as a fraction of image width and height)

4. **Pest Classes**: The dataset contains 12 pest classes:
   - Ants, Bees, Beetles, Caterpillars, Earthworms, Earwigs, Grasshoppers, Moths, Slugs, Snails, Wasps, and others.

## Results

Each approach stores its results in its respective directory:

### ML-HOG-SVM
- Results stored in `ML-HOG-SVM/runs/run_<timestamp>/`
  - `metrics.csv` - Evaluation metrics
  - `predictions/` - Detection outputs
  - `classes/` - Per-class models and features

### ML-HOG-RF
- Models: `ML-HOG-RF/src/models/random_forest_crops.pkl`
- Results: `ML-HOG-RF/src/results/`

### DL-FasterRCNN
- Model checkpoints: `DL-FasterRCNN/results/fasterrcnn/`
- Sample detections: `DL-FasterRCNN/Results/`

### DL-EfficientDet
- Model checkpoints: `DL-EfficientDet/outputs_EfficientDet/checkpoints/`
- Evaluation results: `DL-EfficientDet/results/`
  - `efficientdet_test_results.csv` - Per-class metrics
  - `efficientdet_test_results.json` - Complete metrics
  - `efficientdet_test_confusion_matrix.png` - Confusion matrix
  - `efficientdet_test_timing.csv` - Inference timing

### DL-YOLOv8
- Model weights: `DL-YOLOv8/weights/`
- Test results: `DL-YOLOv8/all_test_results/`
  - `bounded images/` - Visualized detections
  - `labels/` - Detection labels
  - `reports/` - Evaluation plots
- Class breakdown: `DL-YOLOv8/table_by_classes/result_table.csv`