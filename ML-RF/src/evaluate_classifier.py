import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
from tools.helpers import load_full_images
from tools.extract_features import extract_combined_features

# Load saved model 
model_path = "models/random_forest.pkl"

saved = joblib.load(model_path)

clf = saved["model"]          
pca = saved["pca"]            
class_names = saved["class_names"]


# Load test data
test_img_path = "../archive/test/images"
test_label_path = "../archive/test/labels"

X_test_imgs, y_test, _ = load_full_images(test_img_path, test_label_path)
print(f"Test samples: {len(X_test_imgs)}")


# Extract features

X_test_features = extract_combined_features(X_test_imgs)
print("Raw feature dimension:", X_test_features.shape[1])
X_test = pca.transform(X_test_features)


# Predict & evaluate
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=class_names))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=class_names,
    cmap=plt.cm.Blues,
    normalize="true"   
)

plt.xticks(rotation=45)
plt.show()
