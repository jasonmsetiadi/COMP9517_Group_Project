# COMP9517_Group_Project

Currently the directory structure should look like this
```bash   
COMP9517_Group_Project/
│
├── ML-RF/
│   ├── README.md
│   ├── crop_sw.py
│   ├── src/
│   │   ├── train_model.py
│   │   ├── evaluate_classifier.py
│   │   ├── sliding_window_detector.py
│   |   ├── tools/
│   |   │   ├── extract_features.py
│   |   │   ├── helpers.py
│   │   |   ├── nms.py
│   └── .gitignore
```

# Steps to Run Code
Step 0: Transfer the raw images into a folder called **archive** from the kaggle datset in the **ML-RF** directory which should have the following format as shown below.

```bash
COMP9517_Group_Project/
│
├── ML-RF/
    ├── archive/
│       ├── train/
│         ├── images/
│         └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│   └── data.yaml
│   ├── README.md
│   ├── crop_sw.py
│   ├── src/
│   │   ├── train_model.py
│   │   ├── evaluate_classifier.py
│   │   ├── sliding_window_detector.py
│   |   ├── tools/
│   |   │   ├── extract_features.py
│   |   │   ├── helpers.py
│   │   |   ├── nms.py
│   └── .gitignore
```
Step 1: Install packages
```bash
pip install -r requirements.txt
```

Step 2: In the ML-RF directory, run the following to crop the images using the YOLO for training:
```bash
python crop_sw.py
```
The current structure should look like the following:

```bash
COMP9517_Group_Project/
│
├── ML-RF/
│   ├── archive/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── test/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── data.yaml
│   │
│   ├── data/                    ← **NEW: cropped datasets**
│   │   ├── train/
│   │   │   ├── Ants/
│   │   │   ├── Bees/
│   │   │   ├── Beetles/
│   │   │   ├── ...
│   │   │   └── Weevils/
│   │   ├── valid/
│   │   │   ├── Ants/
│   │   │   ├── Bees/
│   │   │   ├── Beetles/
│   │   │   ├── ...
│   │   │   └── Weevils/
│   │   └── test/
│   │       ├── Ants/
│   │       ├── Bees/
│   │       ├── Beetles/
│   │       ├── ...
│   │       └── Weevils/
│   │
│   ├── README.md
│   ├── crop_sw.py
│   │
│   ├── src/
│   │   ├── train_model.py
│   │   ├── evaluate_classifier.py
│   │   ├── sliding_window_detector.py
│   │   └── tools/
│   │       ├── extract_features.py
│   │       ├── helpers.py
│   │       └── nms.py
│   │
│   └── .gitignore

```

Step 3: Now go to the **ML-RF/src** directory and run the following to train the Random Forest classifier model with the cropped images
```bash
python train_model.py
```
This should create a folder in the **ML-RF/src** called models and save model to **ML-RF/src/models/random_forest.pkl**

Step 4: Run the following in the **ML-RF/src** directory to evaluate the classifier model using the original images
```bash
python evaluate_classifier.py
```
Step 5: Run the following in the **ML-RF/src** to run the sliding window detector with the results displayed in the src/results folder.
```bash
python sliding_window_detector.py
```
