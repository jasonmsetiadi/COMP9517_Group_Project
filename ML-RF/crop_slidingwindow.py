import os
import cv2
import yaml
import numpy as np

WINDOW_SIZE = 128
STEP_SIZE = 32
POS_IOU_THRESHOLD = 0.2
MAX_CROPS_PER_INSECT = 5   

# Load class names
with open("archive/data.yaml") as f:
    data = yaml.safe_load(f)
class_names = data['names']

def sliding_window(image, step_size, window_size):
    h, w = image.shape[:2]
    for y in range(0, h - window_size + 1, step_size):
        for x in range(0, w - window_size + 1, step_size):
            yield x, y, image[y:y+window_size, x:x+window_size]


def compute_iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter)


def load_gt_boxes(label_path, w, h):
    boxes, classes = [], []

    if not os.path.exists(label_path):
        return boxes, classes

    with open(label_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())
            cls = int(cls)

            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            boxes.append([x1, y1, x2, y2])
            classes.append(cls)

    return boxes, classes


# Save YOLO ground-truth crops 
def save_direct_gt_crops(img_dir, label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # create class folders
    for cls in class_names:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    print(f"Generating GT crops → {out_dir}")

    for img_file in os.listdir(img_dir):
        if not img_file.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(img_dir, img_file))
        if img is None:
            continue

        h, w = img.shape[:2]
        gt_boxes, gt_classes = load_gt_boxes(
            os.path.join(label_dir, img_file.replace(".jpg", ".txt")), 
            w, h
        )

        # save 1 clean crop per insect
        for idx, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]

            if crop.size != 0:
                crop = cv2.resize(crop, (128,128))
                cls = gt_classes[idx]

                save_path = os.path.join(
                    out_dir, class_names[cls],
                    f"{img_file.replace('.jpg','')}_{idx}.jpg"
                )
                cv2.imwrite(save_path, crop)



# Generate sliding-window crops 
def generate_train_sliding_window(img_dir, label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # create class folders
    for cls in class_names:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    print(f"Generating → {out_dir}")

    for img_file in os.listdir(img_dir):
        if not img_file.endswith(".jpg"):
            continue

        img = cv2.imread(os.path.join(img_dir, img_file))
        if img is None:
            continue

        h, w = img.shape[:2]
        gt_boxes, gt_classes = load_gt_boxes(
            os.path.join(label_dir, img_file.replace(".jpg", ".txt")),
            w, h
        )

        # track counts to limit duplicates
        crop_count = {i: 0 for i in range(len(gt_boxes))}

        # sliding window pass
        for x, y, crop in sliding_window(img, STEP_SIZE, WINDOW_SIZE):
            crop_box = [x, y, x+WINDOW_SIZE, y+WINDOW_SIZE]

            best_iou = 0
            best_idx = None

            for idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(crop_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            # save crop if IoU high enough
            if best_iou >= POS_IOU_THRESHOLD and crop_count[best_idx] < MAX_CROPS_PER_INSECT:
                cls = gt_classes[best_idx]
                crop = cv2.resize(crop, (128,128))

                save_path = os.path.join(
                    out_dir, class_names[cls],
                    f"{img_file.replace('.jpg','')}_{best_idx}_{crop_count[best_idx]}.jpg"
                )
                cv2.imwrite(save_path, crop)
                crop_count[best_idx] += 1

        # ensure every insect gets at least 1 crop
        for idx, box in enumerate(gt_boxes):
            if crop_count[idx] == 0:
                x1, y1, x2, y2 = box
                direct_crop = img[y1:y2, x1:x2]

                if direct_crop.size > 0:
                    direct_crop = cv2.resize(direct_crop, (128,128))
                    cls = gt_classes[idx]

                    save_path = os.path.join(
                        out_dir, class_names[cls],
                        f"{img_file.replace('.jpg','')}_{idx}_fallback.jpg"
                    )
                    cv2.imwrite(save_path, direct_crop)

generate_train_sliding_window("archive/train/images", "archive/train/labels", "data/train")
save_direct_gt_crops("archive/valid/images", "archive/valid/labels", "data/valid")
save_direct_gt_crops("archive/test/images", "archive/test/labels", "data/test")
print("Dataset created.")

