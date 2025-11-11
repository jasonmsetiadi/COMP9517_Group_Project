"""
Configuration for EfficientNet Classification
"""
import os
import torch

# Paths
DATASET_ROOT = r"C:\Users\20695\Desktop\COMP9517\datasets"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train", "images")
VALID_DIR = os.path.join(DATASET_ROOT, "valid", "images")
TEST_DIR = os.path.join(DATASET_ROOT, "test", "images")

OUTPUT_DIR = "outputs_classification"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Classes (12 insects)
CLASS_NAMES = ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
               'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']
NUM_CLASSES = 12

# Model
MODEL_NAME = 'efficientnet_b0'
IMAGE_SIZE = 224
PRETRAINED = True

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

# Subset (for testing)
USE_SUBSET = False  # Set True for quick test
SUBSET_SIZE = 0.1
NUM_EPOCHS = 30

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'