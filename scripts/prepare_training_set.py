import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2
from src.preprocessing import warp_region, extract_features
from src.utils import yolo_to_iou_format, ss_to_iou_format, load_label_data, CLASS_TO_ID

TRAIN_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'train')
TRAIN_PROPOSALS_DIR = os.path.join(TRAIN_DATA_DIR, 'proposals')
TRAIN_OVERLAP_DIR = os.path.join(TRAIN_DATA_DIR, 'overlap')

def negative_bbox_random_sampling(img_path, class_id, num_pos, threshold=0.1):
    # define the number of negative samples
    num_samples = 2 if num_pos == 0 else num_pos*2

    # get indices of negative samples
    file_name = os.path.basename(img_path)
    overlap_path = os.path.join(TRAIN_OVERLAP_DIR, file_name.replace(".jpg", ".npy"))
    overlap_matrix = np.load(overlap_path)
    class_id_column = overlap_matrix[:, class_id]
    negative_population = np.where(class_id_column < threshold)[0]
    sampled_indices = np.random.choice(negative_population, size=min(num_samples, len(negative_population)), replace=False)

    # get proposal bboxes
    proposal_path = os.path.join(TRAIN_PROPOSALS_DIR, file_name.replace(".jpg", ".txt"))
    proposal_bboxes = []
    with open(proposal_path, "r") as f:
        for line in f:
            proposal_bboxes.append(list(map(float, line.split())))
    proposal_bboxes = np.array(proposal_bboxes)
    negative_samples = [ss_to_iou_format(sample) for sample in proposal_bboxes[sampled_indices].tolist()]

    return negative_samples

def positive_bbox_sampling(image, labels, class_id):
    positive_samples = []
    image_width, image_height = image.shape[:2]
    for label in labels:
        if label['label'] == class_id:
            pos_box = yolo_to_iou_format(label['box'], image_width, image_height)
            positive_samples.append(pos_box)
    return positive_samples

def get_training_data(data, class_id, save=True):
    X, y = [], []
    num_pos, num_neg = 0, 0

    for i in range(len(data)):
        img_path , bboxes = data[i]
        img = cv2.imread(img_path)

        # positive samples
        pos_boxes = positive_bbox_sampling(img, bboxes, class_id)
        num_pos += len(pos_boxes)
        for pos_box in pos_boxes:
            warp = warp_region(img, pos_box, output_size=(224, 224))
            features = extract_features(warp)
            X.append(features)
            y.append(1)
        
        # negative samples
        neg_boxes = negative_bbox_random_sampling(img_path, class_id, len(pos_boxes))
        num_neg += len(neg_boxes)
        for neg_box in neg_boxes:
            warp = warp_region(img, neg_box, output_size=(224, 224))
            features = extract_features(warp)
            X.append(features)
            y.append(0)

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} of {len(data)} images. Total training samples for class {class_id}: {len(X)} ({num_pos/(num_pos + num_neg)*100:.2f}% positive, {num_neg/(num_pos + num_neg)*100:.2f}% negative)")
        
    # save the training data
    X, y = np.array(X), np.array(y)
    if save:
        dir = f"training_data/{class_id}"
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.save(os.path.join(dir, "data.npy"), X)
        np.save(os.path.join(dir, "labels.npy"), y)
        
    return X, y


if __name__ == "__main__":
    data = load_label_data(TRAIN_DATA_DIR)
    for class_name, class_id in CLASS_TO_ID.items():
        print(f"Preparing training data for {class_name} class (ID: {class_id})")
        X, y = get_training_data(data, class_id, save=True)
