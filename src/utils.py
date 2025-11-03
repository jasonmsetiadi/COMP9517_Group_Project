import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'train', 'images')
TRAIN_LABELS_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'train', 'labels')


def load_label_data(max_samples=None):
    """
    Load training data
    
    Args:
        max_samples: Maximum number of images to load (None for all)
    """
    # sort from ascending order
    train_image_files = os.listdir(TRAIN_IMAGE_DIR)
    
    # Limit the number of samples if specified
    if max_samples:
        train_image_files = train_image_files[:max_samples]
    
    data = []
    for file_name in train_image_files:
        file_name = file_name.rsplit(".", 1)[0]
        image_path = os.path.join(TRAIN_IMAGE_DIR, file_name + ".jpg")
        label_path = os.path.join(TRAIN_LABELS_DIR, file_name + ".txt")
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