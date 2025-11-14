"""
Dataset loader for classification
Assumes images are organized: dataset/class_name/image.jpg
"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config_clf as config


class InsectDataset(Dataset):
    """Classification dataset"""
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.augment = augment
        self.samples = []
        
        # Find all images organized by class
        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                # Try finding images with class prefix (e.g., ants-1.jpg)
                all_files = [f for f in os.listdir(root_dir) if f.startswith(class_name)]
                for img_file in all_files:
                    self.samples.append((os.path.join(root_dir, img_file), class_idx))
            else:
                # Images in class folders
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_file), class_idx))
        
        # Subset for quick testing
        if config.USE_SUBSET:
            n = int(len(self.samples) * config.SUBSET_SIZE)
            self.samples = self.samples[:n]
        
        # Transforms
        if augment:
            self.transform = A.Compose([
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        transformed = self.transform(image=image)
        image = transformed['image']
        
        return image, label