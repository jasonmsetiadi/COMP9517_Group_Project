import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from tools.helpers import load_cropped_dataset
from tools.extract_features import extract_combined_features


train_path = "../data/train"
valid_path   = "../data/valid"
X_train_imgs, y_train, class_names = load_cropped_dataset(train_path)
X_val_imgs, y_val, _ = load_cropped_dataset(valid_path)

print(f"Training samples: {len(X_train_imgs)}, Validation samples: {len(X_val_imgs)}")

# Extract features 
X_train_features = extract_combined_features(X_train_imgs)
X_val_features   = extract_combined_features(X_val_imgs)

print("raw feature dimension:", X_train_features.shape[1])

# Reduce feature dimension using PCA
pca = PCA(n_components=200, random_state=42)
X_train_pca = pca.fit_transform(X_train_features)
X_val_pca   = pca.transform(X_val_features)

print("Feature dimension after PCA:", X_train_pca.shape[1])


ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train_pca, y_train)

print("training samples:", len(X_train_balanced))

# train rf, lighter to prevent RAM issues)
clf = RandomForestClassifier(
    n_estimators=120,      
    max_depth=25,          
    class_weight="balanced",
    random_state=42,
    n_jobs=-1              
)

clf.fit(X_train_balanced, y_train_balanced)

os.makedirs("models", exist_ok=True)

model_path = "models/random_forest.pkl"
joblib.dump({
    "model": clf,
    "pca": pca,
    "class_names": class_names
}, model_path)

print(f"Model saved to {model_path}")



