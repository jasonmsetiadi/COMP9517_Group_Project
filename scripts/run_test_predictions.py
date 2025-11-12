"""script to run predictions on the test data"""

import os
import sys
import argparse
import numpy as np
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import pickle
from src.preprocessing import warp_region, extract_features
from src.utils import ss_to_iou_format, compute_iou
import cv2
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'test')
TEST_PROPOSAL_DIR = os.path.join(TEST_DATA_DIR, 'proposals')

def classify_region_proposals(img_path, region_proposals, class_svms, 
                              score_threshold=None, nms_threshold=0.3):
    """
    Classify region proposals using trained SVMs.
    
    Args:
        region_proposals: List of bounding boxes from selective search
        cnn_model: Pre-trained CNN for feature extraction
        class_svms: List of trained SVMs
        score_threshold: Minimum confidence score to keep detection
        nms_threshold: IoU threshold for non-maximum suppression
    
    Returns:
        detections: List of (bbox, class_id, confidence_score)
    """
    # Step 1: Extract features for all proposals
    features = []
    img = cv2.imread(img_path)
    for box in region_proposals:
        warp = warp_region(img, ss_to_iou_format(box), output_size=(96, 96))
        feature = extract_features(warp)
        features.append(feature)
    features = np.array(features)
    
    # Step 2: Score each proposal with all class SVMs
    num_classes = len(class_svms)
    num_proposals = len(region_proposals)
    
    # scores[i, j] = confidence score for proposal i, class j
    scores = np.zeros((num_proposals, num_classes))
    
    for class_id, svm in enumerate(class_svms):
        # Get decision function scores (not binary predictions)
        scores[:, class_id] = svm.decision_function(features)
    
    # Step 3: For each proposal, find the best class
    detections = [] # [xmin, ymin, xmax, ymax, class_id, confidence]

    for i, bbox in enumerate(region_proposals):
        # Get best class and its score
        class_scores = scores[i, :]
        best_class_id = np.argmax(class_scores)
        confidence_score = class_scores[best_class_id]
        
        # Keep only if above threshold
        # if confidence_score >= score_threshold:
        detection = list(bbox) 
        detection.append(best_class_id) 
        detection.append(confidence_score) 
        detections.append(detection)
    
    # Step 4: Apply NMS per class
    final_detections = []
    
    for class_id in range(num_classes):
        class_dets = [d for d in detections if d[4] == class_id]
        if len(class_dets) == 0:
            continue
        
        boxes = [d[:4] for d in class_dets]
        scores = [d[5] for d in class_dets]
        
        kept_indices = non_max_suppression(boxes, scores, nms_threshold)
        
        for idx in kept_indices:
            final_detections.append(class_dets[idx])
    
    return np.array(final_detections)


def non_max_suppression(boxes, scores, iou_threshold):
    """
    Apply non-maximum suppression.
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        kept_indices: Indices of boxes to keep
    """
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    
    kept_indices = []
    
    while len(sorted_indices) > 0:
        # Keep the highest scoring box
        current_idx = sorted_indices[0]
        kept_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = np.array([compute_iou(current_box, box) for box in remaining_boxes])
        
        # Keep only boxes with IoU less than threshold
        keep_mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return kept_indices

def load_trained_svm_models(model_dirs):
    trained_models = []
    for model_dir in os.listdir(model_dirs):
        if os.path.isdir(os.path.join(model_dirs, model_dir)):
            trained_models.append(os.path.join(model_dirs, model_dir, "svm.pkl"))

    # load the trained models
    trained_models = [pickle.load(open(model, "rb")) for model in trained_models]
    return trained_models

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", '-id', type=str, required=True)
    args = parser.parse_args()

    run_id = args.run_id
    RUN_DIR = os.path.join(PROJECT_ROOT, 'runs', f'run_{run_id}')
    TEST_PREDICTIONS_DIR = os.path.join(RUN_DIR, 'predictions')

    # iterate over each directory in model_dirs and load the svm.pkl file
    trained_models = load_trained_svm_models(os.path.join(RUN_DIR, 'classes'))

    # get proposal path given image path. end file extensioon is txt
    for i, proposal_path in enumerate(os.listdir(TEST_PROPOSAL_DIR)):
        proposal_bboxes = []
        with open(os.path.join(TEST_PROPOSAL_DIR, proposal_path), "r") as f:
            for line in f:
                proposal_bboxes.append(list(map(float, line.split())))
        img_path = os.path.join(TEST_DATA_DIR, 'images', os.path.basename(proposal_path).replace(".txt", ".jpg"))
        preds = classify_region_proposals(img_path, proposal_bboxes, trained_models, 0.3)
        # save as .npy file
        if not os.path.exists(TEST_PREDICTIONS_DIR):
            os.makedirs(TEST_PREDICTIONS_DIR)
        np.save(os.path.join(TEST_PREDICTIONS_DIR, os.path.basename(proposal_path).replace(".txt", ".npy")), preds)
        print(f"Saved predictions for {proposal_path}, {preds.shape} predictions, {i+1} of {len(os.listdir(TEST_PROPOSAL_DIR))} images")