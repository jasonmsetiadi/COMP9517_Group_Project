"""
Training script for EfficientDet
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import config
from dataset import YOLODataset, collate_fn
from model import create_model, save_checkpoint


def train_epoch(model, dataloader, optimizer, device, accum_steps=4):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for idx, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        
        # Convert targets to the format expected by DetBenchTrain
        batch_target = {
            'bbox': [],
            'cls': [],
            'img_scale': [],
            'img_size': []
        }
        
        for t in targets:
            batch_target['bbox'].append(t['bbox'].to(device))
            batch_target['cls'].append(t['cls'].to(device))
            batch_target['img_scale'].append(t['img_scale'].to(device))
            batch_target['img_size'].append(t['img_size'].to(device))
        
        loss_dict = model(images, batch_target)
        loss = loss_dict['loss'] / accum_steps  # Scale loss
        loss.backward()
        
        # Update weights every accum_steps
        if (idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps
    
    # Final update if needed
    if len(dataloader) % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate model"""
    model.train()  # Keep in train mode for loss computation
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            
            # Convert targets to the format expected by DetBenchTrain
            batch_target = {
                'bbox': [],
                'cls': [],
                'img_scale': [],
                'img_size': []
            }
            
            for t in targets:
                batch_target['bbox'].append(t['bbox'].to(device))
                batch_target['cls'].append(t['cls'].to(device))
                batch_target['img_scale'].append(t['img_scale'].to(device))
                batch_target['img_size'].append(t['img_size'].to(device))
            
            loss_dict = model(images, batch_target)
            loss = loss_dict['loss']
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = YOLODataset(config.TRAIN_IMAGES, config.TRAIN_LABELS, 
                                config.IMAGE_SIZE, augment=True)
    valid_dataset = YOLODataset(config.VALID_IMAGES, config.VALID_LABELS, 
                                config.IMAGE_SIZE, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn, 
                             num_workers=config.NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, 
                             shuffle=False, collate_fn=collate_fn, 
                             num_workers=config.NUM_WORKERS)
    
    print(f"Train: {len(train_dataset)} images, Valid: {len(valid_dataset)} images")
    
    # Create model
    model = create_model(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, accum_steps=4)
        valid_loss = validate(model, valid_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            save_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, save_path)
            print(f"✓ New best model! Loss: {best_loss:.4f}")
    
    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, config.NUM_EPOCHS-1, final_path)
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()