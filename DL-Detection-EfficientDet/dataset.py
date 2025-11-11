"""
Dataset loader for YOLO format
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

import config


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, image_size=512, augment=False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.augment = augment
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.endswith(('.jpg', '.png'))]
        
        # Use subset for faster training if configured
        import config
        if hasattr(config, 'USE_SUBSET') and config.USE_SUBSET:
            subset_size = int(len(self.image_files) * config.SUBSET_SIZE)
            self.image_files = self.image_files[:subset_size]
            print(f"  → Using {config.SUBSET_SIZE*100:.0f}% subset: {len(self.image_files)} images")
        
        # Augmentation
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Load labels (YOLO format)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        boxes, labels = [], []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:5])
                    
                    # Convert YOLO to x1,y1,x2,y2
                    x1 = (x_c - w/2) * orig_w
                    y1 = (y_c - h/2) * orig_h
                    x2 = (x_c + w/2) * orig_w
                    y2 = (y_c + h/2) * orig_h
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        # Apply augmentation
        if self.transform and len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            labels = np.array(transformed['labels'])
        
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= (self.image_size / orig_w)
            boxes[:, [1, 3]] *= (self.image_size / orig_h)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # Create target
        target = {
            'bbox': torch.from_numpy(boxes).float(),
            'cls': torch.from_numpy(labels).long(),
            'img_scale': torch.tensor([1.0]),
            'img_size': torch.tensor([self.image_size, self.image_size])
        }
        
        return image, target


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets