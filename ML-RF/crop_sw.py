import os
import cv2
import yaml
import numpy as np

# Load class names
with open("archive/data.yaml") as f:
    data = yaml.safe_load(f)
class_names = data['names']


def sliding_window(image):
    h, w = image.shape[:2]
    for y in range(0, h - 128 + 1, 32):        # window size 128, stride 32
        for x in range(0, w - 128 + 1, 32):
            yield x, y, image[y:y+128, x:x+128]


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



def save_direct_gt_crops(img_dir, label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for cls in class_names:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    print(f"crops → {out_dir}")

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

        for idx, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            crop = img[y1:y2, x1:x2]

            if crop.size != 0:
                crop = cv2.resize(crop, (128,128))
                cls = gt_classes[idx]

                cv2.imwrite(
                    os.path.join(out_dir, class_names[cls],
                    f"{img_file.replace('.jpg','')}_{idx}.jpg"),
                    crop
                )



def generate_train_sliding_window(img_dir, label_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for cls in class_names:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    print(f"crops → {out_dir}")

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

        crop_count = {i: 0 for i in range(len(gt_boxes))}

        for x, y, crop in sliding_window(img):
            crop_box = [x, y, x+128, y+128]

            best_iou = 0
            best_idx = None

            for idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(crop_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= 0.2 and crop_count[best_idx] < 5:
                cls = gt_classes[best_idx]
                crop = cv2.resize(crop, (128,128))

                cv2.imwrite(
                    os.path.join(out_dir, class_names[cls],
                    f"{img_file.replace('.jpg','')}_{best_idx}_{crop_count[best_idx]}.jpg"),
                    crop
                )

                crop_count[best_idx] += 1


        for idx, box in enumerate(gt_boxes):
            if crop_count[idx] == 0:
                x1, y1, x2, y2 = box
                fallback = img[y1:y2, x1:x2]

                if fallback.size > 0:
                    fallback = cv2.resize(fallback, (128,128))
                    cls = gt_classes[idx]

                    cv2.imwrite(
                        os.path.join(out_dir, class_names[cls],
                        f"{img_file.replace('.jpg','')}_{idx}_fallback.jpg"),
                        fallback
                    )

generate_train_sliding_window("archive/train/images", "archive/train/labels", "data/train")
save_direct_gt_crops("archive/valid/images", "archive/valid/labels", "data/valid")
save_direct_gt_crops("archive/test/images", "archive/test/labels", "data/test")

print("Dataset created.")
