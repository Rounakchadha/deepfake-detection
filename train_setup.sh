#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# train_setup.sh — Download dataset, organize, and train EfficientNet
# Run: bash train_setup.sh
# Takes: ~20min download + ~4-6h training on M2
# ──────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSD_DIR="/Volumes/ADATA SE880/deepfake_data"
VENV="$PROJECT_DIR/venv/bin/python3"
KAGGLE="$PROJECT_DIR/venv/bin/kaggle"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Deepfake Detector — Full Setup & Training"
echo "══════════════════════════════════════════════════════"

# ── Step 1: SSD check ─────────────────────────────────────────────────────────
if [ ! -d "$SSD_DIR" ]; then
    echo "Creating dataset dir on SSD..."
    mkdir -p "$SSD_DIR"
fi

# ── Step 2: Download dataset (skip if already done) ───────────────────────────
REAL_DIR="$SSD_DIR/REAL"
FAKE_DIR="$SSD_DIR/FAKE"

if [ -d "$REAL_DIR" ] && [ -d "$FAKE_DIR" ]; then
    REAL_COUNT=$(ls "$REAL_DIR" | wc -l | tr -d ' ')
    FAKE_COUNT=$(ls "$FAKE_DIR" | wc -l | tr -d ' ')
    if [ "$REAL_COUNT" -gt "1000" ] && [ "$FAKE_COUNT" -gt "1000" ]; then
        echo "Dataset already organized: REAL=$REAL_COUNT, FAKE=$FAKE_COUNT — skipping download"
    else
        echo "Incomplete dataset found, re-downloading..."
        rm -rf "$SSD_DIR/real_vs_fake" 2>/dev/null || true
        NEEDS_DOWNLOAD=1
    fi
else
    NEEDS_DOWNLOAD=1
fi

if [ "${NEEDS_DOWNLOAD:-0}" = "1" ]; then
    echo ""
    echo "Downloading 140K Real and Fake Faces dataset (~4GB)..."
    echo "This will take 10-25 minutes depending on your internet speed."
    echo ""
    $KAGGLE datasets download \
        -d xhlulu/140k-real-and-fake-faces \
        -p "$SSD_DIR" \
        --unzip
    echo "Download complete."
fi

# ── Step 3: Organize into REAL/ and FAKE/ ─────────────────────────────────────
# Dataset structure: real_vs_fake/real-vs-fake/{train,valid}/{real,fake}/
RAW_TRAIN="$SSD_DIR/real_vs_fake/real-vs-fake/train"
RAW_VALID="$SSD_DIR/real_vs_fake/real-vs-fake/valid"

if [ ! -d "$REAL_DIR" ] || [ "$(ls $REAL_DIR 2>/dev/null | wc -l)" -lt "1000" ]; then
    echo ""
    echo "Organizing dataset into REAL/ and FAKE/ structure..."
    mkdir -p "$REAL_DIR" "$FAKE_DIR"

    echo "  Copying train/real → REAL/  (60K images)..."
    cp "$RAW_TRAIN/real/"* "$REAL_DIR/" 2>/dev/null && echo "  Done."

    echo "  Copying valid/real → REAL/  (10K images)..."
    cp "$RAW_VALID/real/"* "$REAL_DIR/" 2>/dev/null && echo "  Done."

    echo "  Copying train/fake → FAKE/  (60K images)..."
    cp "$RAW_TRAIN/fake/"* "$FAKE_DIR/" 2>/dev/null && echo "  Done."

    echo "  Copying valid/fake → FAKE/  (10K images)..."
    cp "$RAW_VALID/fake/"* "$FAKE_DIR/" 2>/dev/null && echo "  Done."

    REAL_COUNT=$(ls "$REAL_DIR" | wc -l | tr -d ' ')
    FAKE_COUNT=$(ls "$FAKE_DIR" | wc -l | tr -d ' ')
    echo ""
    echo "Dataset ready: REAL=$REAL_COUNT, FAKE=$FAKE_COUNT"
fi

# ── Step 4: Run training ───────────────────────────────────────────────────────
echo ""
echo "Starting EfficientNet training..."
echo "Weights will be saved to: $CHECKPOINT_DIR/best_model.pth"
echo ""

mkdir -p "$CHECKPOINT_DIR"

DATA_DIR="$SSD_DIR" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
    $VENV "$PROJECT_DIR/train.py"

# ── Step 5: Auto-activate EfficientNet blending in inference.py ───────────────
echo ""
echo "Activating EfficientNet + ViT ensemble in inference.py..."
sed -i '' 's/EFFICIENTNET_WEIGHT = 0.0/EFFICIENTNET_WEIGHT = 0.3/' \
    "$PROJECT_DIR/backend/inference.py" 2>/dev/null && \
    echo "  Done — EFFICIENTNET_WEIGHT set to 0.3 (30% EfficientNet, 70% ViT)" || \
    echo "  (inference.py already updated or not found)"

# ── Step 6: Restart backend ───────────────────────────────────────────────────
echo "Restarting backend to load new model..."
pkill -f uvicorn 2>/dev/null; sleep 2
nohup "$PROJECT_DIR/venv/bin/python3" -m uvicorn backend.api:app \
    --host 0.0.0.0 --port 8000 \
    > "$PROJECT_DIR/backend.log" 2>&1 &
sleep 5
curl -s http://localhost:8000/ > /dev/null && echo "  Backend restarted OK" || echo "  Backend restart failed — check backend.log"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ALL DONE!"
echo "  Model saved to: $CHECKPOINT_DIR/best_model.pth"
echo "  EfficientNet ensemble: ACTIVE (30% weight)"
echo "  Backend: running on http://localhost:8000"
echo "  Frontend: run ./start.sh if not already running"
echo "══════════════════════════════════════════════════════"
