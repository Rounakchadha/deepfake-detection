#!/usr/bin/env python3
"""
train.py — Launch script for training the Deepfake Detection model.

Usage:
    cd /Users/rounakchadha/Desktop/deepfake
    source venv/bin/activate
    python train.py

Dataset structure expected at data/:
    data/
    ├── FaceForensics++/
    │   ├── REAL/   ← extracted real face frames (JPG/PNG)
    │   └── FAKE/   ← extracted deepfake frames (JPG/PNG)
    ├── Celeb-DF/
    │   ├── REAL/
    │   └── FAKE/
    └── DFDC/
        ├── REAL/
        └── FAKE/

After training, the model is saved to:
    checkpoints/best_model.pth

Then restart the backend and it will automatically load the trained weights:
    uvicorn backend.api:app --reload --port 8000
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.transfer_model import TransferDeepfakeModel
from models.training import train_model, get_device
from data_pipeline.dataset_loader import get_dataloader
from data_pipeline.augmentation import get_train_transforms, get_val_transforms
from torch.utils.data import random_split
from data_pipeline.dataset_loader import DeepfakeDataset

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "data"                         # Root directory of datasets
DATASET_NAMES = ["FaceForensics++"]       # Which datasets to train on
BATCH_SIZE = 32                           # Reduce to 16 on low RAM
EPOCHS = 20                               # Start with 10 for quick test
LEARNING_RATE = 1e-4                      # Adam LR for classifier head
FREEZE_BASE_EPOCHS = 5                    # Epochs to train only the head
UNFREEZE_AFTER = 5                        # Then fine-tune last N backbone blocks
VAL_SPLIT = 0.15                          # 15% for validation
NUM_WORKERS = 4                           # Dataloader workers (reduce to 0 on Mac)
CHECKPOINT_DIR = "checkpoints"
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Deepfake Detection — Training Pipeline")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Datasets   : {DATASET_NAMES}")
    print(f"  Epochs     : {EPOCHS} (freeze base for first {FREEZE_BASE_EPOCHS})")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"{'='*60}\n")

    # ── Load full dataset ──────────────────────────────────────────────────────
    print("Loading dataset...")
    full_dataset = DeepfakeDataset(
        data_dir=DATA_DIR,
        dataset_names=DATASET_NAMES,
        transform=get_train_transforms()
    )

    if len(full_dataset) == 0:
        print("\n❌ ERROR: No images found.")
        print(f"   Expected structure:   {DATA_DIR}/FaceForensics++/REAL/  and  .../FAKE/")
        print("   See NOVELTY.md for dataset download instructions.")
        sys.exit(1)

    # ── Train / Val split ─────────────────────────────────────────────────────
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val transforms to val split (no augmentation)
    val_dataset.dataset.transform = get_val_transforms()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type != 'mps')
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type != 'mps')
    )

    print(f"  Train samples : {train_size}")
    print(f"  Val samples   : {val_size}\n")

    # ── Phase 1: Train only the classifier head (backbone frozen) ─────────────
    print(f"Phase 1: Training classifier head ({FREEZE_BASE_EPOCHS} epochs, backbone frozen)...")
    model = TransferDeepfakeModel(
        target_model='efficientnet_b0',
        freeze_base=True,
        pretrained=True   # Downloads ImageNet weights for backbone (runs once)
    )

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=FREEZE_BASE_EPOCHS,
        lr=LEARNING_RATE,
        device=device,
        save_dir=CHECKPOINT_DIR
    )

    # ── Phase 2: Fine-tune last N backbone blocks ──────────────────────────────
    if EPOCHS > FREEZE_BASE_EPOCHS:
        remaining = EPOCHS - FREEZE_BASE_EPOCHS
        print(f"\nPhase 2: Fine-tuning top backbone blocks ({remaining} epochs)...")
        model.unfreeze_base_model(num_layers=3)  # Unfreeze last 3 blocks

        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=remaining,
            lr=LEARNING_RATE * 0.1,  # Lower LR for fine-tuning
            device=device,
            save_dir=CHECKPOINT_DIR
        )

    print(f"\n✅ Training complete!")
    print(f"   Best model saved to: {CHECKPOINT_DIR}/best_model.pth")
    print(f"\nRestart the backend to load trained weights:")
    print(f"   uvicorn backend.api:app --reload --port 8000")


if __name__ == "__main__":
    main()
