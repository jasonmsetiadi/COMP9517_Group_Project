import os

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