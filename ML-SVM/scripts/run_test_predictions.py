"""script to run predictions on the test data"""

import os
import sys
import argparse
import multiprocessing as mp
import numpy as np
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import json
import time
import pickle
from src.preprocessing import warp_region, extract_features
from src.utils import ss_to_iou_format, compute_iou
import cv2
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'dataset', 'test')
TEST_PROPOSAL_DIR = os.path.join(TEST_DATA_DIR, 'proposals')

_CLASS_SVMS = None
_PREDICTIONS_DIR = None
_SCORE_THRESHOLD = None
_NMS_THRESHOLD = None


def _write_timings(timings_path, data):
    if os.path.exists(timings_path):
        with open(timings_path, "r", encoding="utf-8") as f:
            timings = json.load(f)
    else:
        timings = {}
    timings.update(data)

    os.makedirs(os.path.dirname(timings_path), exist_ok=True)
    with open(timings_path, "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)

def classify_region_proposals(
    img_path,
    region_proposals,
    class_svms=None,
    score_threshold=None,
    nms_threshold=0.3,
):
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
    global _CLASS_SVMS

    if class_svms is None:
        if _CLASS_SVMS is None:
            raise RuntimeError("SVM models are not loaded in this worker.")
        class_svms = _CLASS_SVMS

    # Step 1: Extract features for all proposals
    features = []
    img = cv2.imread(img_path)
    for box in region_proposals:
        warp = warp_region(img, ss_to_iou_format(box), output_size=(96, 96))
        feature = extract_features(warp, use_sift=True, use_hog=False, use_lbp=True, use_color=True)
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

    global _CLASS_SVMS, _PREDICTIONS_DIR, _SCORE_THRESHOLD, _NMS_THRESHOLD
    _CLASS_SVMS = load_trained_svm_models(classes_dir)
    _PREDICTIONS_DIR = predictions_dir
    _SCORE_THRESHOLD = score_threshold
    _NMS_THRESHOLD = nms_threshold

def _process_single_image(proposal_filename):
    global _PREDICTIONS_DIR, _SCORE_THRESHOLD, _NMS_THRESHOLD
    start_ts_image = time.time()

    proposal_bboxes = []
    proposal_path = os.path.join(TEST_PROPOSAL_DIR, proposal_filename)
    with open(proposal_path, "r") as f:
        for line in f:
            proposal_bboxes.append(list(map(float, line.split())))

    img_path = os.path.join(
        TEST_DATA_DIR, "images", os.path.basename(proposal_filename).replace(".txt", ".jpg")
    )
    preds = classify_region_proposals(
        img_path,
        proposal_bboxes,
        score_threshold=_SCORE_THRESHOLD,
        nms_threshold=_NMS_THRESHOLD,
    )

    os.makedirs(_PREDICTIONS_DIR, exist_ok=True)
    np.save(
        os.path.join(
            _PREDICTIONS_DIR, os.path.basename(proposal_filename).replace(".txt", ".npy")
        ),
        preds,
    )

    duration = time.time() - start_ts_image
    return proposal_filename, preds.shape[0] if preds.size else 0, duration

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", "-id", type=str, required=True)
    parser.add_argument(
        "--num_workers",
        "-j",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to CPU count).",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Minimum confidence score to keep a detection.",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.3,
        help="IoU threshold used for non-maximum suppression.",
    )
    args = parser.parse_args()

    run_id = args.run_id
    RUN_DIR = os.path.join(PROJECT_ROOT, 'runs', f'run_{run_id}')
    TEST_PREDICTIONS_DIR = os.path.join(RUN_DIR, 'predictions')

    classes_dir = os.path.join(RUN_DIR, "classes")
    proposal_files = sorted(f for f in os.listdir(TEST_PROPOSAL_DIR) if f.endswith(".txt"))
    total_files = len(proposal_files)

    if total_files == 0:
        print("No proposal files found; nothing to process.")
        sys.exit(0)

    os.makedirs(TEST_PREDICTIONS_DIR, exist_ok=True)

    worker_count = args.num_workers
    if worker_count is None or worker_count < 1:
        worker_count = mp.cpu_count() or 1

    durations = []

    with mp.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(classes_dir, TEST_PREDICTIONS_DIR, args.score_threshold, args.nms_threshold),
    ) as pool:
        for idx, (proposal_filename, num_preds, duration) in enumerate(
            pool.imap_unordered(_process_single_image, proposal_files), start=1
        ):
            durations.append(duration)
            print(
                f"Saved predictions for {proposal_filename}, "
                f"{num_preds} predictions, {idx} of {total_files} images",
                flush=True,
            )

    # log prediction duration
    timings = {
        "average_prediction_duration_per_image_seconds": float(np.mean(durations)),
        "total_prediction_duration_seconds": float(np.sum(durations)),
    }
    _write_timings(os.path.join(RUN_DIR, "timings.json"), timings)