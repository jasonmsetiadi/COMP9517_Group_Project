
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


class YOLODataset(Dataset):
    
    def __init__(self, images_dir, labels_dir, image_size=512, transforms=None, is_train=True):

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.transforms = transforms
        self.is_train = is_train
        
        # Get list of image files
        self.image_files = []
        for fname in os.listdir(images_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.image_files.append(fname)
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]
        
        # Load YOLO format labels
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format (x_center, y_center, w, h) to (x_min, y_min, x_max, y_max)
                    # YOLO coordinates are normalized, so multiply by image dimensions
                    x_min = (x_center - width / 2) * orig_w
                    y_min = (y_center - height / 2) * orig_h
                    x_max = (x_center + width / 2) * orig_w
                    y_max = (y_center + height / 2) * orig_h
                    
                    # Clip to image boundaries
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(orig_w, x_max)
                    y_max = min(orig_h, y_max)
                    
                    # Ensure valid box
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id)
        
        # Convert to numpy arrays
        if len(boxes) == 0:
            # Empty image - create dummy box
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        
        # Apply augmentations
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Resize boxes accordingly
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (self.image_size / orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (self.image_size / orig_h)
        
        # Convert image to tensor and normalize
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        # Create target dict
        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.from_numpy(labels).long(),
            'image_id': torch.tensor([idx])
        }
        
        return image, target


def get_train_transforms():
    """Get training augmentation transforms"""
    if not config.USE_AUGMENTATION:
        return None
    
    return A.Compose([
        A.HorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
        A.VerticalFlip(p=config.VERTICAL_FLIP_PROB),
        A.Rotate(limit=config.ROTATION_LIMIT, p=config.AUGMENTATION_PROB),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=config.BRIGHTNESS_CONTRAST_PROB
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=config.HUE_SATURATION_PROB
        ),
        A.RandomGamma(p=config.RANDOM_GAMMA_PROB),
        A.Blur(blur_limit=3, p=config.BLUR_PROB),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        min_visibility=0.3
    ))


def get_valid_transforms():
    """Get validation transforms (no augmentation)"""
    return None


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Handles variable number of boxes per image
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_dataloaders(batch_size=8, num_workers=4):
    # Create datasets
    train_dataset = YOLODataset(
        images_dir=config.TRAIN_IMAGES,
        labels_dir=config.TRAIN_LABELS,
        image_size=config.IMAGE_SIZE,
        transforms=get_train_transforms(),
        is_train=True
    )
    
    valid_dataset = YOLODataset(
        images_dir=config.VALID_IMAGES,
        labels_dir=config.VALID_LABELS,
        image_size=config.IMAGE_SIZE,
        transforms=get_valid_transforms(),
        is_train=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Valid dataset: {len(valid_dataset)} images")
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    
    return train_loader, valid_loader


if __name__ == '__main__':
    """Test dataset loading"""
    print("Testing dataset loading...")
    
    # Create dataset
    dataset = YOLODataset(
        images_dir=config.TRAIN_IMAGES,
        labels_dir=config.TRAIN_LABELS,
        image_size=config.IMAGE_SIZE,
        transforms=get_train_transforms()
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    image, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Number of boxes: {len(target['boxes'])}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")
    
    # Test dataloader
    train_loader, valid_loader = create_dataloaders(batch_size=4, num_workers=0)
    
    images, targets = next(iter(train_loader))
    print(f"\nBatch images shape: {images.shape}")
    print(f"Number of targets: {len(targets)}")