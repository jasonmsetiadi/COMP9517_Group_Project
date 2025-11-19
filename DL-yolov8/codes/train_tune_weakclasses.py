import os
import yaml
import glob
import collections
from ultralytics import YOLO


DATA_YAML = "/Users/wang/9517/group_project/AgroPest-12/data.yaml"
PROJECT = "/Users/wang/9517/group_project/runs/debug"
BASE_RUN = "agropest_yolov8n_fulltry3"
NEW_RUN = BASE_RUN + "_tune1"
model = YOLO(os.path.join(PROJECT, BASE_RUN, "weights", "best.pt"))

WEAK_CLASS_NAMES = ["Slugs", "Caterpillars", "Earthworms"]


EPOCHS = 15
BATCH = 4
IMGSZ = 512
DEVICE = "mps"
LR0 = 3e-4
LRF = 0.01
OPTIMIZER = "AdamW"
CLS_GAIN = 0.75
COS_LR = True
SAVE_PERIOD = 1


with open(DATA_YAML, "r") as f:
    data_yaml = yaml.safe_load(f)

name2id = {name: i for i, name in enumerate(data_yaml["names"])}
WEAK_CLASS_IDS = [name2id[n] for n in WEAK_CLASS_NAMES if n in name2id]

print("\n==============================")
print("weakclass map: (from data.yaml)")
for n in WEAK_CLASS_NAMES:
    if n in name2id:
        print(f"  {n} → ID {name2id[n]}")
    else:
        print(f" didn't find class in data: {n}")
print("==============================\n")


train_labels_root = "/Users/wang/9517/group_project/AgroPest-12/train/labels"
balanced_txt_path = os.path.join(PROJECT, NEW_RUN, "train_balanced.txt")

os.makedirs(os.path.dirname(balanced_txt_path), exist_ok=True)

counter = collections.Counter()
weak_image_paths = []

label_files = glob.glob(os.path.join(train_labels_root, "**", "*.txt"), recursive=True)
for label_path in label_files:
    try:
        with open(label_path) as f:
            cls_ids = [int(line.split()[0]) for line in f if line.strip()]
        if any(cid in WEAK_CLASS_IDS for cid in cls_ids):
            # image path
            image_path = label_path.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"
            weak_image_paths.append(image_path)
            for cid in cls_ids:
                if cid in WEAK_CLASS_IDS:
                    counter[cid] += 1
    except Exception as e:
        print(f"can't read label file: {label_path} ({e})")

print("weakclass statistics：")
for cid in WEAK_CLASS_IDS:
    cname = list(name2id.keys())[cid]
    print(f"  {cname:<15} (ID {cid}): {counter[cid]} 张")

print(f"\nweakcalss amount: {len(weak_image_paths)} 张")

if len(weak_image_paths) > 0:
    print("first 5:")
    for p in weak_image_paths[:5]:
        print(" ", p)
else:
    print("didn't find any weakclass")


with open(balanced_txt_path, "w") as f:
    for p in weak_image_paths:
        f.write(p + "\n")

print(f"\nbalanced txt path: {balanced_txt_path}\n")


model_path = os.path.join(PROJECT, BASE_RUN, "weights", "best.pt")
assert os.path.exists(model_path), f"didn't find weights: {model_path}"

print("======================================================================")
print("tuen start（stat with best.pt，no resume）")
print(f"base run: {BASE_RUN}")
print(f"new run  : {NEW_RUN}")
print("======================================================================")

model = YOLO(model_path)

results = model.train(
    data=DATA_YAML,
    project=PROJECT,
    name=NEW_RUN,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,
    device=DEVICE,
    lr0=LR0,
    lrf=LRF,
    optimizer=OPTIMIZER,
    cls=CLS_GAIN,
    cos_lr=COS_LR,
    exist_ok=True,
    plots=False,
    save_period=0,
    workers=0,
    max_det=100,
    patience=50,
    verbose=True,
    resume=True
)

print("\ntune start log path:：")
print(f"   {os.path.join(PROJECT, NEW_RUN)}")
print("======================================================================")
