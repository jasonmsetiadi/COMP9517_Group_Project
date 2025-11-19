import os
import cv2
import joblib
import numpy as np
from tools.extract_features import extract_combined_features
from tools.helpers import load_class_names
from tools.nms import non_max_suppression

MODEL_PATH = "models/random_forest.pkl"
TEST_IMG_DIR = "../archive/test/images"
OUTPUT_DIR = "results"
WINDOW_SIZE = 128
STEP_SIZE = 32
CONF_THRESHOLD = 0.5

saved = joblib.load(MODEL_PATH)
clf = saved["model"]
pca = saved["pca"]
class_names = saved["class_names"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sliding_window(image, step_size, window_size):
    h, w = image.shape[:2]
    for y in range(0, h - window_size[1] + 1, step_size):
        for x in range(0, w - window_size[0] + 1, step_size):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]

def detect_insects(image):
    bboxes, scores, classes = [], [], []

    for x, y, window in sliding_window(image, STEP_SIZE, (WINDOW_SIZE, WINDOW_SIZE)):
        feats = extract_combined_features([window])
        feats = pca.transform(feats)

        prob_vec = clf.predict_proba(feats)[0]
        prob = np.max(prob_vec)
        pred = np.argmax(prob_vec)

        if prob >= CONF_THRESHOLD:
            bboxes.append([x, y, x + WINDOW_SIZE, y + WINDOW_SIZE])
            scores.append(prob)
            classes.append(pred)

    if len(bboxes) == 0:
        return [], [], []

    bboxes, scores, classes = non_max_suppression(
        np.array(bboxes), np.array(scores), np.array(classes), overlapThresh=0.3
    )
    return bboxes, scores, classes

for img_file in os.listdir(TEST_IMG_DIR):
    if not img_file.endswith(".jpg"):
        continue

    img = cv2.imread(os.path.join(TEST_IMG_DIR, img_file))
    if img is None:
        continue

    bboxes, scores, preds = detect_insects(img)

    if len(bboxes) == 0:
        continue

    for box, cls in zip(bboxes, preds):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_names[cls], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_file), img)
 'src/results/' folder")
