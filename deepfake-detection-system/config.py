"""
This file contains the central configuration for the deepfake detection project.
It defines default paths, model parameters, and training settings.
Modifying these values here will affect the entire project.
"""

import os

# --- Project Root ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Dataset Paths ---
# It's recommended to place your datasets in the 'data' directory.
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FACEFORENSICS_PATH = os.path.join(DATA_DIR, 'FaceForensics++')
CELEB_DF_PATH = os.path.join(DATA_DIR, 'Celeb-DF')
DFDC_PATH = os.path.join(DATA_DIR, 'DFDC')

# --- Output Paths ---
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# --- Model Parameters ---
# Default model to use (can be 'XceptionNet' or 'MesoNet')
DEFAULT_MODEL = 'MesoNet'

# Image dimensions
# Note: XceptionNet was trained on 299x299, but can be adapted. MesoNet uses 256x256.
IMAGE_SIZE = 256
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Number of frames to extract from each video
NUM_FRAMES = 10

# --- Training Parameters ---
# Number of training epochs
EPOCHS = 10

# Batch size for training and evaluation
BATCH_SIZE = 32

# Learning rate for the optimizer
LEARNING_RATE = 1e-3

# Number of top layers to unfreeze for fine-tuning XceptionNet
NUM_LAYERS_TO_UNFREEZE = 30

# --- Frontend Settings ---
# Confidence threshold for classifying as 'DEEPFAKE'
PREDICTION_THRESHOLD = 0.5


if __name__ == '__main__':
    # Print the configuration to verify paths
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Weights Directory: {WEIGHTS_DIR}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"Image Size: {IMAGE_SIZE}")

