# COMP9517 Crop Pest Detection

## Team Members

| Name | Student ID | Responsibilities | Contact |
|------|-----------|------------------|---------|
| Justine | z5423358 |  | [siyeon.kim@student.unsw.edu.au] |
| Ben | z5360027 |  | [b.laphai@student.unsw.edu.au] |
| Jason | z5611110 |  | [jason.setiadi@unsw.edu.au] |
| Mike | z5698637 |  | [xiaojun.guo@student.unsw.edu.au] |
| Michael | z5540434 |  | [feiyang.wang@student.unsw.edu.au] |

## Setup

### Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- See requirements.txt for dependencies

### Installation
```bash
git clone <repo-url>
pip install -r requirements.txt
```

### Dataset Setup

- Download the dataset from [Kaggle: AgroPest-12 Dataset](https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset).
- Labels are provided in YOLO format:
  - Each label line follows: `class_id x_center y_center width height`
  - All values are normalized between 0 and 1, where:
    - `x_center` and `y_center` represent the center of the bounding box (as a fraction of image width and height, respectively).
    - `width` and `height` are the size of the bounding box (as a fraction of image width and height).

## Quick Start

## Results
Key results are stored in results/:
- metrics/ - Quantitative evaluation metrics
- visualizations/ - Plots and figures for report
- explanations/ - Explainability outputs

## Citation
[List key papers and tools used]