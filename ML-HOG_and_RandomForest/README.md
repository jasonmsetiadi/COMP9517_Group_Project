# COMP9517_Group_Project

Currently the directory structure should look like this
```bash   
COMP9517_Group_Project/
│
├── ML-HOG_and_RandomForest/
│   ├── README.md
│   ├── crop_yolo.py
│   ├── src/
│   │   ├── train_classifier.py
│   │   ├── evaluate_classifier.py
│   │   ├── sliding_window_detector.py
│   |   ├── tools/
│   |   │   ├── extract_features.py
│   |   │   ├── helpers.py
│   │   |   ├── nms.py
│   └── .gitignore
```

# Steps to Run Code
Step 0: Transfer the raw images into a folder called **archive** in the **ML-HOG_and_RandomForest** directory which should have the following format as shown below.

```bash
COMP9517_Group_Project/
│
├── ML-HOG_and_RandomForest/
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
│   ├── crop_yolo.py
│   ├── src/
│   │   ├── train_classifier.py
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

Step 2: In the ML-HOG_and_RandomForest directory, run the following to crop the images using the YOLO for training:
```bash
python crop_yolo.py
```

Step 3: Now go to the **ML-HOG_and_RandomForest/src** directory and run the following to train the Random Forest classifier model with the cropped images
```bash
python train_classifier.py
```
This should create a folder in the **ML-HOG_and_RandomForest/src** called models and save model to **ML-HOG_and_RandomForest/src/models/random_forest_crops.pkl**

Step 4: Run the following in the **ML-HOG_and_RandomForest/src** directory to evaluate the classifier model using the original images
```bash
python evaluate_classifier.py
```
Step 5: Run the following in the **ML-HOG_and_RandomForest/src** to run the sliding window detector with the results displayed in the src/results folder.
```bash
python sliding_window_detector.py
```
