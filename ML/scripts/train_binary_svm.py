"""script to train the binary SVM model for each class"""

import os
import sys
import argparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import pickle
import json
import time

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import CLASS_TO_ID

def _write_timings(timings_path, data):
    if os.path.exists(timings_path):
        with open(timings_path, "r", encoding="utf-8") as f:
            timings = json.load(f)
            timings.update(data)
            json.dump(timings, f, indent=2)
    else:
        os.makedirs(os.path.dirname(timings_path), exist_ok=True)
        with open(timings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def initialize_model():
    linear_svm = LinearSVC(
        penalty='l2',
        loss='squared_hinge',
        dual=True,
        tol=0.0001,
        C=0.001,
        class_weight='balanced',
        verbose=1,
        random_state=42,
        max_iter=10000
    )
    model = make_pipeline(Normalizer(norm='l2'), linear_svm)
    return model

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", '-id', type=str, required=True)
    args = parser.parse_args()

    run_id = args.run_id
    RUN_DIR = os.path.join(PROJECT_ROOT, 'runs', f'run_{run_id}', 'classes')
    timings_per_class = {}

    # load training data
    for class_id in CLASS_TO_ID.values():
        start_ts_class = time.time()
        X = np.load(os.path.join(RUN_DIR, f"class_{class_id}", "data.npy"))
        y = np.load(os.path.join(RUN_DIR, f"class_{class_id}", "labels.npy"))
        print(f"Loaded training data for class {class_id}: {X.shape}, {y.shape}")

        print(f"Training binary SVM model for class {class_id}")
        model = initialize_model()
        model.fit(X, y)

        # save trained model as pickle file
        with open(os.path.join(RUN_DIR, f"class_{class_id}", "svm.pkl"), "wb") as f:
            pickle.dump(model, f)
        print(f"Saved binary SVM model for class {class_id}")

        # log training duration for this class
        train_duration_class = time.time() - start_ts_class
        timings_per_class[f"class_{class_id}"] = train_duration_class

    # log training duration
    timings = {
        "total_train_duration_seconds": sum(timings_per_class.values()),
        "per_class_train_duration_seconds": timings_per_class,
    }
    _write_timings(os.path.join(PROJECT_ROOT, 'runs', f'run_{run_id}', "timings.json"), timings)