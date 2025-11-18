import os
import cv2
import yaml

# Load class names from data.yml
with open("archive/data.yaml") as f:
    data = yaml.safe_load(f)

class_names = data['names']

# Crops insects from images using YOLO labels and saves them in class-specific folder
def crop_yolo_set(img_dir, label_dir, save_dir):
   
    os.makedirs(save_dir, exist_ok=True)
    for cls in class_names:
        os.makedirs(os.path.join(save_dir, cls), exist_ok=True)

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, label_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(label_path) as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:5])
            x1 = int((x_center - bw/2) * w)
            y1 = int((y_center - bh/2) * h)
            x2 = int((x_center + bw/2) * w)
            y2 = int((y_center + bh/2) * h)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (128,128))
            save_path = os.path.join(save_dir, class_names[cls_id], f"{os.path.basename(img_path).split('.')[0]}_{idx}.jpg")
            cv2.imwrite(save_path, crop)

# Process train, val, test sets
crop_yolo_set("archive/train/images", "archive/train/labels", "data/crops/train")
crop_yolo_set("archive/valid/images", "archive/valid/labels", "data/crops/val")
crop_yolo_set("archive/test/images", "archive/test/labels", "data/crops/test")

print("Crops created for train, val, and test sets")
