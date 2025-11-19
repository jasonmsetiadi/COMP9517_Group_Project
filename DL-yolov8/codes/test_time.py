import time
import json
from ultralytics import YOLO
from pathlib import Path
import yaml
import os



MODEL_PATH = "runs/debug/agropest_yolov8n_fulltry3/weights/best.pt"
TEST_DIR = "/Users/wang/9517/group_project/AgroPest-12/test"
DATA_YAML  = "/Users/wang/9517/group_project/AgroPest-12/data.yaml"
DEVICE = "mps"



def count_images(dir_path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    n = 0
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                n += 1
    return n


def main():
    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    test_dir = TEST_DIR

    n_test_images = count_images(test_dir)

    print(f"Found {n_test_images} test images.")

    model = YOLO(MODEL_PATH)

    print("Start timing test inference…")
    t0 = time.time()
    
    model.val(
        data=DATA_YAML,
        split="test",
        device=DEVICE,
        save_json=False,
        plots=False,
        exist_ok=True,
    )

    t1 = time.time()

    total_test_time = t1 - t0
    avg_per_image   = total_test_time / n_test_images

    print(f"\nTotal prediction duration:  {total_test_time:.4f} seconds")
    print(f"Average per image:          {avg_per_image:.6f} seconds")


    result = {
        "average_prediction_duration_per_image_seconds": avg_per_image,
        "total_prediction_duration_seconds": total_test_time
    }

    out_path = "/Users/wang/9517/group_project/runs/detect/test_time/test_time_stats.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved test_time_stats.json to: {out_path}")


if __name__ == "__main__":
    main()
