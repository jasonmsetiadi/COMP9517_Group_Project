"""
Train EfficientNet for insect classification
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import time

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
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


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
    
    print(f"\nTraining for {config.NUM_EPOCHS} epochs...\n")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)
        
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
    
    training_time = (time.time() - start_time) / 3600
    print(f"\nTraining complete! Time: {training_time:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()