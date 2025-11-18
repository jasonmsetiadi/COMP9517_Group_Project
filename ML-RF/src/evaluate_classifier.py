import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
from tools.helpers import load_cropped_dataset
from tools.extract_features import extract_combined_features

from tools.helpers import load_full_images

# Paths
model_path = "models/random_forest_crops.pkl"
test_img_path = "../archive/test/images"
test_label_path = "../archive/test/labels"

# Load model
clf = joblib.load(model_path)

# Load full images for testing
X_test_imgs, y_test, class_names = load_full_images(test_img_path, test_label_path)
print(f"Test samples: {len(X_test_imgs)}")


# Extract features
print("Extracting features for test set...")
X_test = extract_combined_features(X_test_imgs)
print(f"Feature vector size: {X_test.shape[1]}")


# Predict & evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=class_names,
    cmap=plt.cm.Blues
)
plt.xticks(rotation=45)
plt.show()



