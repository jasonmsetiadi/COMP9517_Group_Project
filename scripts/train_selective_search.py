"""script to run selective search on the training data and save the proposals"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_label_data
from src.selective_search import selective_search_rcnn_pipeline

TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'train')
TRAIN_PROPOSALS_DIR = os.path.join(TRAIN_DATA_DIR, 'proposals')

if __name__ == "__main__":
    data = load_label_data(TRAIN_DATA_DIR)
    print(f"Loaded {len(data)} images")
    for i, (img_path, bboxes) in enumerate(data):
        boxes = selective_search_rcnn_pipeline(img_path, save_boxes_path=os.path.join(TRAIN_PROPOSALS_DIR, os.path.basename(img_path).replace('.jpg', '.txt')))
        print(f"Processed {i+1} of {len(data)} images")