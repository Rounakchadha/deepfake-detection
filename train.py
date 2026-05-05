#!/usr/bin/env python3
"""
train.py — Fixed Deepfake Detection Training Pipeline
======================================================
For Colab: mount Drive, set DATA_DIR, then run all cells.

Dataset structure:
    DATA_DIR/
    ├── REAL/   ← real face images (JPG/PNG)
    └── FAKE/   ← deepfake/AI-generated face images (JPG/PNG)

Output: checkpoints/best_model.pth
"""

import os, sys, time, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR        = os.environ.get("DATA_DIR", "data")
CHECKPOINT_DIR  = os.environ.get("CHECKPOINT_DIR", "checkpoints")
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "16"))  # 16 safe for M2 8GB, 64 on Colab T4
EPOCHS          = 25
LR_HEAD         = 3e-4            # Head-only phase learning rate
LR_FINETUNE     = 3e-5            # Fine-tune phase (10x smaller)
FREEZE_EPOCHS   = 5               # Epochs with backbone frozen
VAL_SPLIT       = 0.15
NUM_WORKERS     = int(os.environ.get("NUM_WORKERS", "4"))
IMG_SIZE        = 224
SEED            = 42
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)

# ── Transforms (separate for train and val) ───────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
from torchvision.datasets import ImageFolder

def _is_valid_image(path):
    """Skip macOS metadata files (._*) and other non-images."""
    return not os.path.basename(path).startswith('._')

class SplitableImageFolder(ImageFolder):
    """ImageFolder that allows setting per-split transforms after random_split."""
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, is_valid_file=_is_valid_image)
        self._override_transform = None

    def __getitem__(self, index):
        path, label = self.samples[index]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        t = self._override_transform if self._override_transform else self.transform
        if t:
            img = t(img)
        return img, label


def get_dataloaders(data_dir):
    """
    Loads dataset with correct train/val transforms.
    BUG FIX: We create two separate dataset instances so val_dataset.transform
    changes don't bleed into train_dataset (the old random_split bug).
    """
    # Accept both REAL/FAKE (uppercase) and real/fake (lowercase)
    real_dir = (os.path.join(data_dir, 'REAL') if os.path.exists(os.path.join(data_dir, 'REAL'))
                else os.path.join(data_dir, 'real'))
    fake_dir = (os.path.join(data_dir, 'FAKE') if os.path.exists(os.path.join(data_dir, 'FAKE'))
                else os.path.join(data_dir, 'fake'))
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError(
            f"Expected REAL/ and FAKE/ (or real/ and fake/) inside {data_dir}/\n"
            f"Found: {os.listdir(data_dir)}"
        )

    # Build a clean symlink-based directory with ONLY REAL/FAKE to avoid
    # ImageFolder picking up stray subdirs (e.g. demo_raw, real_vs_fake)
    import tempfile
    clean_dir = tempfile.mkdtemp(prefix="deepfake_train_")
    os.symlink(real_dir, os.path.join(clean_dir, os.path.basename(real_dir)))
    os.symlink(fake_dir, os.path.join(clean_dir, os.path.basename(fake_dir)))
    data_dir = clean_dir

    # Load once to get indices, then split
    full = SplitableImageFolder(data_dir, transform=None)
    n = len(full)
    if n == 0:
        raise ValueError("No images found. Check DATA_DIR structure.")

    val_n   = int(n * VAL_SPLIT)
    train_n = n - val_n

    # Reproducible split
    gen = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = random_split(range(n), [train_n, val_n], generator=gen)

    # Two separate dataset instances — each with its own transform
    train_ds = SplitableImageFolder(data_dir, transform=train_transforms)
    val_ds   = SplitableImageFolder(data_dir, transform=val_transforms)

    train_subset = Subset(train_ds, train_idx.indices)
    val_subset   = Subset(val_ds,   val_idx.indices)

    # Class balance stats
    labels = [full.targets[i] for i in train_idx.indices]
    n_real = labels.count(full.class_to_idx.get('REAL', 0))
    n_fake = labels.count(full.class_to_idx.get('FAKE', 1))
    print(f"  Train: {train_n} images  (REAL={n_real}, FAKE={n_fake})")
    print(f"  Val  : {val_n} images")
    print(f"  Classes: {full.class_to_idx}")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Compute pos_weight for imbalanced datasets (important!)
    pos_weight = torch.tensor([n_real / (n_fake + 1e-6)])  # weight for FAKE class
    return train_loader, val_loader, pos_weight, full.class_to_idx


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    from torchvision import models
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Unfreeze all — we use a small LR instead of freezing (more stable)
    for p in m.parameters():
        p.requires_grad_(True)
    # Replace head: 1280 → 256 → 1  (deeper head, better generalisation)
    in_features = m.classifier[1].in_features  # 1280
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return m


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, leave=False)
    for imgs, labels in bar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            out  = model(imgs)
            loss = criterion(out, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        pred   = (torch.sigmoid(out) > 0.5).float()
        correct += (pred == labels).sum().item()
        total  += labels.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device).float().unsqueeze(1)
        out    = model(imgs)
        loss   = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        pred   = (torch.sigmoid(out) > 0.5).float()
        correct += (pred == labels).sum().item()
        total  += labels.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))

    print(f"\n{'='*60}")
    print(f"  Deepfake Detector — Training  |  Device: {device}")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Val: {VAL_SPLIT*100:.0f}%")
    print(f"{'='*60}\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    train_loader, val_loader, pos_weight, class_map = get_dataloaders(DATA_DIR)
    pos_weight = pos_weight.to(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Weighted BCE handles class imbalance automatically
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Two-phase training ────────────────────────────────────────────────────
    # Phase 1: freeze backbone, train only the new head fast
    for name, p in model.named_parameters():
        if 'classifier' not in name:
            p.requires_grad_(False)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS)
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc  = 0.0
    best_state    = None
    history       = []

    print(f"\nPhase 1 — Head warmup ({FREEZE_EPOCHS} epochs, backbone frozen)")
    for epoch in range(FREEZE_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"  Ep {epoch+1:02d}/{FREEZE_EPOCHS} | "
              f"Train {tr_acc:.3f} ({tr_loss:.4f}) | "
              f"Val {vl_acc:.3f} ({vl_loss:.4f}) | "
              f"LR {lr_now:.2e} | {elapsed:.0f}s")
        history.append({'epoch': epoch+1, 'phase': 1, 'val_acc': vl_acc, 'val_loss': vl_loss})
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
            print(f"  ✓ New best val_acc: {best_val_acc:.4f}")

    # Phase 2: unfreeze entire model, fine-tune with lower LR
    print(f"\nPhase 2 — Full fine-tune ({EPOCHS - FREEZE_EPOCHS} epochs)")
    for p in model.parameters():
        p.requires_grad_(True)

    # Separate LRs: small for backbone, larger for head (discriminative LR)
    head_params     = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR_FINETUNE},
        {'params': head_params,     'lr': LR_FINETUNE * 5},
    ], weight_decay=1e-4)

    remaining = EPOCHS - FREEZE_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-7)

    patience, no_improve = 5, 0
    for epoch in range(remaining):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Ep {FREEZE_EPOCHS+epoch+1:02d}/{EPOCHS} | "
              f"Train {tr_acc:.3f} ({tr_loss:.4f}) | "
              f"Val {vl_acc:.3f} ({vl_loss:.4f}) | {elapsed:.0f}s")
        history.append({'epoch': FREEZE_EPOCHS+epoch+1, 'phase': 2,
                        'val_acc': vl_acc, 'val_loss': vl_loss})

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
            torch.save(best_state, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  ✓ Saved best — val_acc: {best_val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping (no improvement for {patience} epochs)")
                break

    # Always save final best
    if best_state:
        torch.save(best_state, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))

    # Save training history
    with open(os.path.join(CHECKPOINT_DIR, 'training_history.json'), 'w') as f:
        json.dump({'best_val_acc': best_val_acc, 'class_map': class_map,
                   'history': history}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"  Saved to          : {CHECKPOINT_DIR}/best_model.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
