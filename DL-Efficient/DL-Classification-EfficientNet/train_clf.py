"""
Train EfficientNet for insect classification WITH TIMING
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import time
import json

import config_clf as config
from dataset_clf import InsectDataset


def create_model():
    """Create EfficientNet model"""
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=config.PRETRAINED,
        num_classes=config.NUM_CLASSES
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch with per-class timing"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-class training time
    class_times = {i: 0.0 for i in range(config.NUM_CLASSES)}
    class_counts = {i: 0 for i in range(config.NUM_CLASSES)}
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        batch_start = time.time()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        batch_time = time.time() - batch_start
        
        # Accumulate time per class
        for label in labels.cpu().numpy():
            class_times[label] += batch_time / len(labels)
            class_counts[label] += 1
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total, class_times


def validate(model, loader, criterion, device):
    """Validate"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading datasets...")
    train_dataset = InsectDataset(config.TRAIN_DIR, augment=True)
    valid_dataset = InsectDataset(config.VALID_DIR, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, num_workers=config.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Valid: {len(valid_dataset)} images\n")
    
    # Create model
    print(f"Creating {config.MODEL_NAME}...")
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training
    best_acc = 0
    start_time = time.time()
    
    # Accumulate per-class training times across all epochs
    total_class_times = {i: 0.0 for i in range(config.NUM_CLASSES)}
    
    print(f"\nTraining for {config.NUM_EPOCHS} epochs...\n")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_acc, class_times = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
        # Accumulate class times
        for cls_id, cls_time in class_times.items():
            total_class_times[cls_id] += cls_time
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}")
        
        # Save best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': valid_acc,
            }, f"{config.CHECKPOINT_DIR}/best_model.pth")
            print(f"✓ Best model saved! Acc: {valid_acc:.4f}\n")
        else:
            print()
    
    training_time = time.time() - start_time
    
    print(f"\nTraining complete! Time: {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Save training timing information
    timing_info = {
        "total_train_duration_seconds": training_time,
        "total_train_duration_hours": training_time / 3600,
        "per_class_train_duration_seconds": {
            f"class_{i}_{config.CLASS_NAMES[i]}": total_class_times[i]
            for i in range(config.NUM_CLASSES)
        },
        "epochs": config.NUM_EPOCHS,
        "best_validation_accuracy": float(best_acc)
    }
    
    timing_file = f"{config.CHECKPOINT_DIR}/training_timing.json"
    with open(timing_file, 'w') as f:
        json.dump(timing_info, f, indent=2)
    
    print(f"\n✓ Training timing saved: {timing_file}")
    print(f"\nPer-class training time:")
    for i in range(config.NUM_CLASSES):
        print(f"  {config.CLASS_NAMES[i]:<15}: {total_class_times[i]:.2f}s")
    print(f"\nTotal: {training_time:.2f}s ({training_time/3600:.2f} hours)")


if __name__ == '__main__':
    main()