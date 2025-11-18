import os
import shutil
from ultralytics import YOLO

MODEL_PATH = "/Users/wang/9517/group_project/runs/debug/agropest_yolov8n_fulltry3/weights/best.pt"
SOURCE_DIR = "/Users/wang/9517/group_project/AgroPest-12/test/images"
SAVE_ROOT  = "/Users/wang/9517/group_project/runs/predict"
RUN_NAME   = "test_fulltry3"
model = YOLO(MODEL_PATH)

results = model.predict(
    source=SOURCE_DIR,
    project=SAVE_ROOT,
    name=RUN_NAME,
    imgsz=512,
    conf=0.25,
    iou=0.7,
    save=True,
    save_txt=True,
    save_conf=False,
    exist_ok=True,
    verbose=True
)

output_dir = os.path.join(SAVE_ROOT, RUN_NAME)
images_dir = os.path.join(output_dir, "bounded images")
labels_dir = os.path.join(output_dir, "labels")

os.makedirs(images_dir, exist_ok=True)

for f in os.listdir(output_dir):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        src = os.path.join(output_dir, f)
        dst = os.path.join(images_dir, f)
        shutil.move(src, dst)

print(f"\nbounded_test_images:{images_dir}")
print(f"test_labels:{labels_dir}")

