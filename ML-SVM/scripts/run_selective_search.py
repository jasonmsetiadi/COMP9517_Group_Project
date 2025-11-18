"""script to run selective search on the training data and save the proposals"""

import os
import sys
import argparse

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_label_data
from src.selective_search import selective_search_rcnn_pipeline

TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'dataset', 'train')
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'dataset', 'test')
TRAIN_PROPOSALS_DIR = os.path.join(TRAIN_DATA_DIR, 'proposals')
TEST_PROPOSALS_DIR = os.path.join(TEST_DATA_DIR, 'proposals')

if __name__ == "__main__":
    # parse arguments to decide whether to run on train or test data
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", '-m', type=str, required=True)
    args = parser.parse_args()
    mode = args.mode

    if mode == "train":
        data = load_label_data(TRAIN_DATA_DIR)
    elif mode == "test":
        data = load_label_data(TEST_DATA_DIR)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Loaded {len(data)} images")
    for i, (img_path, bboxes) in enumerate(data):
        save_path = os.path.join(TRAIN_PROPOSALS_DIR if mode == "train" else TEST_PROPOSALS_DIR, os.path.basename(img_path).replace('.jpg', '.txt'))
        boxes = selective_search_rcnn_pipeline(img_path, save_boxes_path=save_path)
        print(f"Processed {i+1} of {len(data)} images")