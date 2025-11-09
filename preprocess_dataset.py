# preprocess_dataset.py

import cv2, numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _read_yolo_labels(txt):
    boxes, clses = [], []
    if not txt.exists():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    for ln in txt.read_text().splitlines():
        if not ln.strip(): continue
        c, xc, yc, w, h = map(float, ln.split()[:5])
        clses.append(int(c))
        boxes.append([xc, yc, w, h])
    return np.asarray(boxes, np.float32), np.asarray(clses, np.int64)

def _yolo_to_xyxy_abs(boxes, W, H):
    if boxes.size == 0:
        return np.zeros((0, 4), np.float32)
    xc, yc, w, h = boxes[:,0]*W, boxes[:,1]*H, boxes[:,2]*W, boxes[:,3]*H
    x1 = np.clip(xc - w/2, 0, W-1)
    y1 = np.clip(yc - h/2, 0, H-1)
    x2 = np.clip(xc + w/2, 0, W-1)
    y2 = np.clip(yc + h/2, 0, H-1)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

class YoloDetDataset(Dataset):
    def __init__(self, root, img_size=640, is_train=False, num_classes=12):
        root = Path(root)
        self.img_dir = root / "images"
        self.lab_dir = root / "labels"
        self.paths = sorted([p for p in self.img_dir.rglob("*") if p.suffix.lower() in IMG_EXT])
        self.img_size = img_size
        self.is_train = is_train

        if is_train:
            self.tfms = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.2),
                    A.LongestMaxSize(max_size=img_size),
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, fill=0), # Use 'value'
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
            )
        else:
            self.tfms = A.Compose(
                [
                    A.LongestMaxSize(max_size=img_size),
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=0), # Use 'value'
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        ip = self.paths[i]
        lp = self.lab_dir / ip.relative_to(self.img_dir).with_suffix(".txt")

        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None: # Handle corrupted images
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            boxes_xyxy = np.zeros((0, 4), np.float32)
            clses = np.zeros((0,), np.int64)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            boxes_norm, clses = _read_yolo_labels(lp)
            boxes_xyxy = _yolo_to_xyxy_abs(boxes_norm, W, H)

        class_labels = clses.tolist()
        bboxes_list = boxes_xyxy.tolist()

        out = self.tfms(image=img, bboxes=bboxes_list, class_labels=class_labels)
        img_t = out["image"]
        bxs_t = np.array(out["bboxes"], np.float32)
        cls_t = np.array(out["class_labels"], np.int64)

        # --- This is the simple target format we need ---
        target = {
            "bbox": torch.tensor(bxs_t, dtype=torch.float32),
            "cls": torch.tensor(cls_t, dtype=torch.int64),
            "img_size": torch.tensor((self.img_size, self.img_size), dtype=torch.float32),
            "img_scale": torch.tensor(1.0, dtype=torch.float32)
        }
        
        return img_t, target

def collate_fn(batch):
    imgs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    imgs = torch.stack(imgs, dim=0)
    
    return imgs, tgts