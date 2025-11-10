"""
Configuration for EfficientDet Detection
"""
import os
import torch

# Dataset paths
DATASET_ROOT = r"C:\Users\20695\Desktop\COMP9517\datasets"
TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "train", "labels")
VALID_IMAGES = os.path.join(DATASET_ROOT, "valid", "images")
VALID_LABELS = os.path.join(DATASET_ROOT, "valid", "labels")

# Output paths
OUTPUT_DIR = r"C:\Users\20695\Desktop\COMP9517\COMP9517_Group_Project\outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Class names (12 insect classes)
CLASS_NAMES = ['ants', 'bees', 'beetle', 'catterpillar', 'earthworms', 'earwig',
               'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']
NUM_CLASSES = 12

# Fast training option - Set to True for quick testing
USE_SUBSET = False  # Set to True to use only part of data
SUBSET_SIZE = 1   # Use 10% of data (only if USE_SUBSET=True)

# Model settings
MODEL_NAME = 'tf_efficientdet_d0'
IMAGE_SIZE = 512

# Training parameters
BATCH_SIZE = 2  # Reduced for CPU training
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Set to 0 for CPU to avoid multiprocessing issues

# Detection settings
CONF_THRESHOLD = 0.001
NMS_THRESHOLD = 0.5

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)