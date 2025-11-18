import os
import shutil
from datetime import datetime
from ultralytics import YOLO


DATA_YAML = "/Users/wang/9517/group_project/AgroPest-12/data.yaml"
PROJECT   = "/Users/wang/9517/group_project/runs/debug"
RUN_NAME  = "agropest_yolov8n_fulltry3"
INIT_WEIGHTS = "yolov8n.pt"
DEVICE = "mps"

EPOCHS  = 50
BATCH   = 8
IMGSZ   = 640
WORKERS = 4
SAVE_PERIOD = 1
DETERMINISTIC = False


def copy_if_exists(src: str, dst_dir: str):
    if os.path.exists(src):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def main():
    print("=" * 70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] YOLOv8 Launcher")
    print(f"Project: {PROJECT}")
    print(f"Run name: {RUN_NAME}")
    print("=" * 70)

    run_dir     = os.path.join(PROJECT, RUN_NAME)
    weights_dir = os.path.join(run_dir, "weights")
    last_ckpt   = os.path.join(weights_dir, "last.pt")
    best_ckpt   = os.path.join(weights_dir, "best.pt")
    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    resume_flag = False
    if os.path.exists(last_ckpt):
        print(f"resume：found {last_ckpt}")
        model = YOLO(last_ckpt)
        resume_flag = True
    elif os.path.exists(best_ckpt):
        print(f"did not found last.pt，start with best.pt：{best_ckpt}")
        model = YOLO(best_ckpt)
    else:
        print(f"：start with initial weights {INIT_WEIGHTS}")
        model = YOLO(INIT_WEIGHTS)

    print(f"Resume mode: {resume_flag}")

    results = model.train(
        data=DATA_YAML,
        project=PROJECT,
        name=RUN_NAME,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        workers=WORKERS,
        device=DEVICE,
        resume=resume_flag,
        exist_ok=True,
        save_period=SAVE_PERIOD,
        deterministic=DETERMINISTIC,
        verbose=True,
    )

    print("\n validation")
    val_results = model.val(
        data=DATA_YAML,
        device=DEVICE,
        project=run_dir,
        name="val",
        exist_ok=True,
        split="val",
        save_json=False,
        plots=True,
    )
    print("validation finished")

    print("\ncopy images to report")
    copy_if_exists(os.path.join(run_dir, "results.png"), reports_dir)
    copy_if_exists(os.path.join(run_dir, "labels.jpg"), reports_dir)

    val_dir = os.path.join(run_dir, "val")
    copy_if_exists(os.path.join(val_dir, "confusion_matrix.png"), reports_dir)
    copy_if_exists(os.path.join(val_dir, "confusion_matrix_normalized.png"), reports_dir)

    print(f"  run_dir: {run_dir}")
    print(f"  weights_dir: {weights_dir}/best.pt, {weights_dir}/last.pt")
    print(f"  validation_dir: {val_dir}")
    print(f"  report_dir: {reports_dir}")
    print("  have copied: results.png, labels.jpg, confusion_matrix.png, confusion_matrix_normalized.png")

if __name__ == "__main__":
    main()
