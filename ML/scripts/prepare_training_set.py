"""script to prepare the training set for the model"""
import os
import sys
import argparse
import multiprocessing as mp
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2
from src.preprocessing import warp_region, extract_features
from src.utils import yolo_to_iou_format, ss_to_iou_format, load_label_data, CLASS_TO_ID

TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'dataset', 'train')
TRAIN_PROPOSALS_DIR = os.path.join(TRAIN_DATA_DIR, 'proposals')
TRAIN_OVERLAP_DIR = os.path.join(TRAIN_DATA_DIR, 'overlap')


def negative_bbox_random_sampling(img_path, class_id, num_pos, threshold=0.3):
    # define the number of negative samples
    num_samples = 1 if num_pos == 0 else min(int(num_pos * 0.1), 1)

    # get indices of negative samples
    file_name = os.path.basename(img_path)
    overlap_path = os.path.join(TRAIN_OVERLAP_DIR, file_name.replace(".jpg", ".npy"))
    overlap_matrix = np.load(overlap_path)
    class_id_column = overlap_matrix[:, class_id]
    negative_population = np.where(class_id_column < threshold)[0]
    sampled_indices = np.random.choice(
        negative_population,
        size=min(num_samples, len(negative_population)),
        replace=False,
    )

    # get proposal bboxes
    proposal_path = os.path.join(TRAIN_PROPOSALS_DIR, file_name.replace(".jpg", ".txt"))
    proposal_bboxes = []
    with open(proposal_path, "r") as f:
        for line in f:
            proposal_bboxes.append(list(map(float, line.split())))
    proposal_bboxes = np.array(proposal_bboxes)
    negative_samples = [ss_to_iou_format(sample) for sample in proposal_bboxes[sampled_indices].tolist()]

    return negative_samples


def positive_bbox_sampling(img_path, labels, class_id, img_width, img_height, threshold=1):
    positive_samples = []

    # get positive samples from labels
    for label in labels:
        if label['label'] == class_id:
            pos_box = yolo_to_iou_format(label['box'], img_width, img_height)
            positive_samples.append(pos_box)

    # get indices of negative samples
    file_name = os.path.basename(img_path)
    overlap_path = os.path.join(TRAIN_OVERLAP_DIR, file_name.replace(".jpg", ".npy"))
    overlap_matrix = np.load(overlap_path)
    class_id_column = overlap_matrix[:, class_id]
    sampled_indices = np.where(class_id_column > threshold)[0]

    # get proposal bboxes
    proposal_path = os.path.join(TRAIN_PROPOSALS_DIR, file_name.replace(".jpg", ".txt"))
    proposal_bboxes = []
    with open(proposal_path, "r") as f:
        for line in f:
            proposal_bboxes.append(list(map(float, line.split())))
    proposal_bboxes = np.array(proposal_bboxes)
    positive_samples.extend([ss_to_iou_format(sample) for sample in proposal_bboxes[sampled_indices].tolist()])

    return positive_samples


def get_training_data(data, class_id, run_dir, region_size=(96, 96), save=True):
    X, y = [], []
    num_pos, num_neg = 0, 0

    for i in range(len(data)):
        img_path, bboxes = data[i]
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]

        # positive samples
        pos_boxes = positive_bbox_sampling(img_path, bboxes, class_id, img_width, img_height, threshold=0.7)
        num_pos += len(pos_boxes)
        for pos_box in pos_boxes:
            warp = warp_region(img, pos_box, output_size=region_size)
            features = extract_features(warp)
            X.append(features)
            y.append(1)

        # negative samples
        neg_boxes = negative_bbox_random_sampling(img_path, class_id, len(pos_boxes))
        num_neg += len(neg_boxes)
        for neg_box in neg_boxes:
            warp = warp_region(img, neg_box, output_size=region_size)
            features = extract_features(warp)
            X.append(features)
            y.append(0)

        if (i + 1) % 100 == 0 and (num_pos + num_neg) > 0:
            print(
                f"[class {class_id}] Processed {i + 1} of {len(data)} images. "
                f"Total training samples: {len(X)} "
                f"({num_pos / (num_pos + num_neg) * 100:.2f}% positive, "
                f"{num_neg / (num_pos + num_neg) * 100:.2f}% negative)",
                flush=True,
            )

    # save the training data
    X, y = np.array(X), np.array(y)
    if save:
        class_dir = os.path.join(run_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)
        np.save(os.path.join(class_dir, "data.npy"), X)
        np.save(os.path.join(class_dir, "labels.npy"), y)

    return X, y


def _prepare_single_class(class_name, class_id, data, run_dir, region_size):
    print(f"Preparing training data for {class_name} class (ID: {class_id})", flush=True)
    X, y = get_training_data(data, class_id, run_dir, region_size=region_size, save=True)
    num_pos = int(np.sum(y))
    num_neg = len(y) - num_pos
    print(
        f"Finished {class_name} class (ID: {class_id}) - samples: {len(X)} "
        f"(positives: {num_pos}, negatives: {num_neg})",
        flush=True,
    )
    return class_name, class_id, len(X), num_pos, num_neg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers",
        "-j",
        type=int,
        default=None,
        help="Number of parallel workers (defaults to min(cpu_count, number of classes))",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run identifier. If not provided, uses current timestamp.",
    )
    args = parser.parse_args()

    # define path for the run
    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(PROJECT_ROOT, 'runs', f'run_{run_id}', 'classes')
    os.makedirs(run_dir, exist_ok=True)

    data = load_label_data(TRAIN_DATA_DIR)

    class_entries = list(CLASS_TO_ID.items())
    worker_count = args.num_workers
    if worker_count is None:
        cpu_count = mp.cpu_count() or 1
        worker_count = min(len(class_entries), cpu_count)
    elif worker_count < 1:
        raise ValueError("num_workers must be a positive integer")

    print(f"Preparing training data in {run_dir}")
    print(f"Using {worker_count} worker(s) for {len(class_entries)} class(es).", flush=True)

    with mp.Pool(processes=worker_count) as pool:
        results = pool.starmap(
            _prepare_single_class,
            [
                (class_name, class_id, data, run_dir, (96, 96))
                for class_name, class_id in class_entries
            ],
        )

    print("Completed preparing training data:")
    for class_name, class_id, total, pos, neg in results:
        print(
            f" - {class_name} (ID: {class_id}): {total} samples "
            f"(positives: {pos}, negatives: {neg})"
        )
    print(f"Run ID: {run_id}")
