from ultralytics import YOLO
from pathlib import Path
import pandas as pd

# DEFINITION OF TP FP FN
# TP = ...  True Positives
# FP = ...  False Positives
# FN = ...  False Negatives
# precision = TP / (TP + FP)
# recall    = TP / (TP + FN)
# accuracy  = TP / (TP + FP + FN)

def main():
    # load model
    model_path = "runs/debug/agropest_yolov8n_fulltry3/weights/best.pt"
    data_yaml = "AgroPest-12/data.yaml"

    model = YOLO(model_path)

    # test
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        iou=0.5,
        conf=0.001,
        save=False
    )

    print("result saved to：", metrics.save_dir)

    names = metrics.names

    # YOLO metrics
    p    = metrics.box.p.tolist()
    r    = metrics.box.r.tolist()
    ap50 = metrics.box.ap50.tolist()
    f1   = metrics.box.f1.tolist()

    nt_per_class = metrics.nt_per_class

    rows = []

    for cls_idx, cls_name in names.items():
        P = float(p[cls_idx])
        R = float(r[cls_idx])
        N = float(nt_per_class[cls_idx])

        acc = 0.0
        if N > 0 and P > 0.0 and R > 0.0:
            TP = R * N
            FP = TP * (1.0 / P - 1.0)
            FN = TP * (1.0 / R - 1.0)
            acc = TP / (TP + FP + FN)

        rows.append({
            "Class": cls_name,
            "Precision": P,
            "Recall": R,
            "AP": float(ap50[cls_idx]),
            "F1": float(f1[cls_idx]),
            "Accuracy": acc,
        })

    df = pd.DataFrame(rows)

    # average row
    overall = {
        "Class": "Overall",
        "Precision": df["Precision"].mean(),
        "Recall": df["Recall"].mean(),
        "AP": df["AP"].mean(),
        "F1": df["F1"].mean(),
        "Accuracy": df["Accuracy"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([overall])], ignore_index=True)

    df = df.applymap(lambda x: f"{x:.6f}" if isinstance(x, float) else x)

    # save path
    save_dir = Path(metrics.save_dir)
    csv_path = save_dir / "class_metrics_test.csv"
    df.to_csv(csv_path, index=False)

    print("class_metrics_test.csv saved to：", csv_path)


if __name__ == "__main__":
    main()
