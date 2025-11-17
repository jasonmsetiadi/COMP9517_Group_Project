import os
import cv2
import yaml

def load_cropped_dataset(base_path):
    """
    Load cropped images and their class labels from folder structure:
    base_path/
        class_1/
        class_2/
        ...
    Returns:
        X: list of images
        y: list of class indices
        class_names: list of class names
    """
    class_names = sorted(os.listdir(base_path))
    X = []
    y = []
    for idx, cls in enumerate(class_names):
        cls_path = os.path.join(base_path, cls)
        for file in os.listdir(cls_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(cls_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    X.append(img)
                    y.append(idx)
    return X, y, class_names

def load_full_images(img_folder, label_folder):
    """
    Load full images and their class labels from YOLO-style label files.
    Assumes:
        - img_folder contains .jpg images
        - label_folder contains .txt labels with class IDs (first number on each line)
    Returns:
        X: list of images
        y: list of class indices (take first label per image for simplicity)
        class_names: list of class names (sorted)
    """
    import yaml
    # Load class names from data.yaml (assumes in archive folder)
    class_names = load_class_names("../archive/data.yaml")

    X = []
    y = []

    for label_file in os.listdir(label_folder):
        if not label_file.endswith(".txt"):
            continue
        img_file = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_folder, img_file)
        if not os.path.exists(img_path):
            continue

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Read label file
        with open(os.path.join(label_folder, label_file)) as f:
            lines = f.readlines()
            if len(lines) == 0:
                continue
            # Take the first class label
            class_id = int(lines[0].split()[0])
            X.append(img)
            y.append(class_id)

    return X, y, class_names

def load_class_names(yaml_path):
    """
    Load class names from a YOLO-style data.yaml file.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data['names']


