import os
import random
import shutil
from pathlib import Path

def make_imbalanced_dataset(
    input_dir,
    output_dir,
    imbalance_ratio=0.3,
    seed=42
):
    """
    Create an imbalanced dataset by randomly reducing images from some classes.

    Parameters:
        input_dir (str): Path to the original dataset (each class has its own folder).
        output_dir (str): Path to save the imbalanced dataset.
        imbalance_ratio (float): The ratio of samples to keep for minority classes (0–1).
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all class folders
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes.")

    # Randomly select half of them to become minority classes
    minority_classes = random.sample(class_dirs, len(class_dirs) // 2)
    print(f"Minority classes: {[c.name for c in minority_classes]}")

    for class_dir in class_dirs:
        images = list(class_dir.glob('*'))
        random.shuffle(images)

        # If class is minority, keep fewer samples
        if class_dir in minority_classes:
            num_keep = int(len(images) * imbalance_ratio)
        else:
            num_keep = len(images)

        target_dir = output_path / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images[:num_keep]:
            shutil.copy(img_path, target_dir / img_path.name)

        print(f"Class '{class_dir.name}': kept {num_keep}/{len(images)} images")

    print(f"\n✅ Imbalanced dataset created at: {output_path}")


if __name__ == "__main__":
    # Example usage:
    make_imbalanced_dataset(
        input_dir="datasets/AgroPest12/train",      # change to your dataset path
        output_dir="datasets/AgroPest12_imbalanced", 
        imbalance_ratio=0.3                         # keep 30% of some classes
    )
