# src/sliding_window_detector.py

import os
import cv2
import joblib
import numpy as np
from tools.extract_features import extract_combined_features
from tools.helpers import load_class_names
from tools.nms import non_max_suppression  # make sure you have this

# -------------------------------
# Paths and parameters
# -------------------------------
MODEL_PATH = "models/random_forest_crops.pkl"  # trained on cropped RGB images
TEST_IMG_DIR = "../archive/test/images"
OUTPUT_DIR = "results"
WINDOW_SIZE = 128
STEP_SIZE = 32  # sliding window step
CONF_THRESHOLD = 0.5

# -------------------------------
# Load trained model and class names
# -------------------------------
clf = joblib.load(MODEL_PATH)
class_names = load_class_names("../archive/data.yaml")  # returns list of class names

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Sliding window generator
# -------------------------------
def sliding_window(image, step_size=32, window_size=(128,128)):
    """Yield (x, y, window)"""
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# -------------------------------
# Detect insects on an image
# -------------------------------
def detect_insects(image):
    bboxes = []
    scores = []
    predicted_classes = []

    for (x, y, window) in sliding_window(image, step_size=STEP_SIZE, window_size=(WINDOW_SIZE, WINDOW_SIZE)):
        if window.shape[0] != WINDOW_SIZE or window.shape[1] != WINDOW_SIZE:
            continue

        # Use same feature extraction as training
        features = extract_combined_features([window])
        pred = clf.predict(features)[0]
        prob = max(clf.predict_proba(features)[0])

        if prob >= CONF_THRESHOLD:
            bboxes.append([x, y, x + WINDOW_SIZE, y + WINDOW_SIZE])
            scores.append(prob)
            predicted_classes.append(pred)

    if len(bboxes) == 0:
        return [], [], []

    # Apply Non-Maximum Suppression
    final_boxes, final_scores, final_classes = non_max_suppression(
        np.array(bboxes), np.array(scores), np.array(predicted_classes), overlapThresh=0.3
    )
    return final_boxes, final_scores, final_classes

# -------------------------------
# Process all test images
# -------------------------------
for img_file in os.listdir(TEST_IMG_DIR):
    if not img_file.endswith(".jpg"):
        continue
    img_path = os.path.join(TEST_IMG_DIR, img_file)
    image = cv2.imread(img_path)
    if image is None:
        continue

    bboxes, scores, classes = detect_insects(image)

    if len(bboxes) == 0:
        # Skip saving images with no detections
        print(f"Processed {img_file}, 0 insects detected. Skipping save.")
        continue

    # Draw boxes
    for (box, cls) in zip(bboxes, classes):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, class_names[cls], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Save output
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), image)
    print(f"Processed {img_file}, {len(bboxes)} insects detected.")

print("✅ Detection complete. Check the 'results/' folder.")
