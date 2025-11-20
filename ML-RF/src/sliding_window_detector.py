import os
import cv2
import joblib
import numpy as np
from tools.extract_features import extract_combined_features
from tools.helpers import load_class_names
from tools.nms import non_max_suppression

saved = joblib.load("models/random_forest.pkl")
clf = saved["model"]
pca = saved["pca"]
class_names = saved["class_names"]

os.makedirs("results", exist_ok=True)

def sliding_window(image):
    for y in range(0, image.shape[0] - 128 + 1, 32):
        for x in range(0, image.shape[1] - 128 + 1, 32):
            yield x, y, image[y:y+128, x:x+128]

def detect_insects(image):
    bboxes, scores, predicted_classes = [], [], []

    for x, y, window in sliding_window(image):
        if window.shape[:2] != (128, 128):
            continue

        feats = extract_combined_features([window])
        feats_pca = pca.transform(feats)

        pred = clf.predict(feats_pca)[0]
        prob = max(clf.predict_proba(feats_pca)[0])

        if prob >= 0.5:
            bboxes.append([x, y, x + 128, y + 128])
            scores.append(prob)
            predicted_classes.append(pred)

    if len(bboxes) == 0:
        return [], [], []

    final_boxes, final_scores, final_classes = non_max_suppression(
        np.array(bboxes), np.array(scores), np.array(predicted_classes), 0.3
    )
    return final_boxes, final_scores, final_classes

for img_file in os.listdir("../archive/test/images"):
    if not img_file.endswith(".jpg"):
        continue

    img_path = os.path.join("../archive/test/images", img_file)
    image = cv2.imread(img_path)
    if image is None:
        continue

    bboxes, scores, classes = detect_insects(image)

    if len(bboxes) == 0:
        print(f"{img_file}, 0 insects detected.")
        continue

    for (box, cls) in zip(bboxes, classes):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, class_names[cls], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imwrite(os.path.join("results", img_file), image)
    print(f" {img_file}, {len(bboxes)} insects detected.")

print("Detection complete, check the 'src/results/' folder")

