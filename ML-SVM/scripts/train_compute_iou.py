"""script to compute overlap between proposals and ground truth boxes and save the results"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_label_data, compute_image_max_iou, yolo_to_iou_format, ss_to_iou_format
import numpy as np
import cv2

TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'dataset', 'train')
TRAIN_PROPOSALS_DIR = os.path.join(TRAIN_DATA_DIR, 'proposals')
TRAIN_OVERLAP_DIR = os.path.join(TRAIN_DATA_DIR, 'overlap')

if __name__ == "__main__":
    data = load_label_data(TRAIN_DATA_DIR)
    print(f"Loaded {len(data)} images")
    os.makedirs(TRAIN_OVERLAP_DIR, exist_ok=True)
    for i, (img_path, bboxes) in enumerate(data):
        img_width, img_height = cv2.imread(img_path).shape[:2]
        proposals = open(os.path.join(TRAIN_PROPOSALS_DIR, os.path.basename(img_path).replace('.jpg', '.txt')), "r").read()
        proposals = [tuple(map(float, line.split(" "))) for line in proposals.split("\n") if line]
       
        # convert proposals to IoU format
        proposals = [ss_to_iou_format(proposal) for proposal in proposals]
        
        # convert ground truth boxes to IoU format
        for box in bboxes:
            box["box"] = yolo_to_iou_format(box["box"], img_width, img_height)

        max_iou = compute_image_max_iou(proposals, bboxes, len(CLASS_TO_ID))

        # save the results as .npy file
        np.save(os.path.join(TRAIN_OVERLAP_DIR, os.path.basename(img_path).replace('.jpg', '.npy')), max_iou)
        print(f"Processed {i+1} of {len(data)} images")