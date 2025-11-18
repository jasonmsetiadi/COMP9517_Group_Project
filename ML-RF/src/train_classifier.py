import os
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from tools.helpers import load_cropped_dataset
from tools.extract_features import extract_combined_features
import numpy as np
import joblib



# Paths to cropped images
train_path = "../data/crops/train"
val_path   = "../data/crops/val"

# Load cropped dataset
X_train_imgs, y_train, class_names = load_cropped_dataset(train_path)
X_val_imgs, y_val, _ = load_cropped_dataset(val_path)
print(f"Training samples: {len(X_train_imgs)}, Validation samples: {len(X_val_imgs)}")

# handle class imbalance
X_train_flat = [img.flatten() for img in X_train_imgs]
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train_flat, y_train)

X_train_imgs = [np.array(x, dtype=np.uint8).reshape(128,128,3) for x in X_train_res]

# Extract combined features
print("Extracting features...")
X_train = extract_combined_features(X_train_imgs)
X_val = extract_combined_features(X_val_imgs)
print(f"Feature vector size: {X_train.shape[1]}")


# Train Random Forest
clf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
clf.fit(X_train, y_train_res)

# Save model
os.makedirs("models", exist_ok=True)
model_path = "models/random_forest_crops.pkl"
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")
