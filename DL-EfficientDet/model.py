"""
EfficientDet model setup
"""
import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

import config


def create_model(pretrained=True):
    """Create EfficientDet model"""
    effdet_config = get_efficientdet_config(config.MODEL_NAME)
    effdet_config.num_classes = config.NUM_CLASSES
    effdet_config.image_size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    net = EfficientDet(effdet_config, pretrained_backbone=pretrained)
    net.class_net = HeadNet(effdet_config, num_outputs=config.NUM_CLASSES)
    
    model = DetBenchTrain(net, effdet_config)
    print(f"Created {config.MODEL_NAME} with {config.NUM_CLASSES} classes")
    return model


def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Saved checkpoint: {save_path}")


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model