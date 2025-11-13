import os
import numpy as np
from typing import Dict, List, Tuple

CLASS_TO_ID = {
    "Ants": 0,
    "Bees": 1,
    "Beetles": 2,
    "Caterpillars": 3,
    "Earthworms": 4,
    "Earwigs": 5,
    "Grasshoppers": 6,
    "Moths": 7,
    "Slugs": 8,
    "Snails": 9,
    "Wasps": 10,
    "Weevils": 11,
}
BBox = Tuple[float, float, float, float]  # IoU format (x_min, y_min, x_max, y_max)

def load_label_data(dir, max_samples=None):
    """
    Load label data from a directory
    
    Args:
        dir: Directory containing the images and labels
        max_samples: Maximum number of images to load (None for all)
        
    Returns:
        List of tuples containing the image path and the list of labels
    """
    # sort from ascending order
    image_files = os.listdir(os.path.join(dir, 'images'))
    
    # Limit the number of samples if specified
    if max_samples:
        image_files = image_files[:max_samples]
    
    data = []
    for file_name in image_files:
        file_name = file_name.rsplit(".", 1)[0]
        image_path = os.path.join(dir, 'images', file_name + ".jpg")
        label_path = os.path.join(dir, 'labels', file_name + ".txt")
        text_label = open(label_path, "r").read()

        # parse label
        label = []
        for line in text_label.split("\n"):
            if line:
                # first value is class label, rest are box coordinates
                label.append({
                    "box": tuple(map(float, line.split(" ")[1:5])),
                    "label": int(line.split(" ")[0])
                })
        data.append((image_path, label))
    return data

def yolo_to_iou_format(yolo_box, img_width, img_height):
    """
    Convert a YOLO-format box to IoU standard format (x_min, y_min, x_max, y_max).

    Args:
        yolo_box: tuple/list (x_center, y_center, w, h), normalized [0, 1]
        img_width: image width in pixels
        img_height: image height in pixels

    Returns:
        (x_min, y_min, x_max, y_max) in absolute pixel coordinates
    """
    x_center, y_center, w, h = yolo_box

    # Convert from normalized to absolute pixel coordinates
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height

    # Compute corners
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    return (x_min, y_min, x_max, y_max)

def ss_to_iou_format(ss_box):
    """
    Convert a Selective Search box to IoU standard format (x_min, y_min, x_max, y_max).

    Args:
        ss_box: tuple/list (x, y, w, h)

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    x, y, w, h = ss_box

    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    return (x_min, y_min, x_max, y_max)


def compute_iou(boxA: BBox, boxB: BBox) -> float:
    """
    Compute IoU between two boxes in (x_min, y_min, x_max, y_max) format.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)



# function that processes a single image
# gets region proposals from the image along with the ground truth boxes and classes
# computes the max IoU for each proposal across all ground truth boxes
# returns matrix of (num_proposals, num_classes) with the max IoU for each proposal and class
def compute_image_max_iou(proposals: List[BBox], labels: List[Dict], num_classes: int) -> np.ndarray:
    """
    Compute the max IoU for each proposal across all ground truth boxes
    Both proposals and ground truth boxes are in IoU format (x_min, y_min, x_max, y_max)
    """
    # initialize the max IoU matrix
    max_iou = np.zeros((len(proposals), num_classes))
    # iterate over each ground truth box
    for label in labels:
        ground_truth_box = label["box"]
        ground_truth_class = label["label"]
        # compute the IoU between all proposals and this ground truth box
        ious = np.array([compute_iou(proposal, ground_truth_box) for proposal in proposals])
        # update the max IoU for the proposal and class
        max_iou[:, ground_truth_class] = np.maximum(max_iou[:, ground_truth_class], ious)
    return max_iou
