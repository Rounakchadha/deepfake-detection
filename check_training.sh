#!/bin/bash
# check_training.sh — Check status of the download/training pipeline
SSD="/Volumes/ADATA SE880/deepfake_data"
LOG="$SSD/training_run.log"
CKPT="/Users/rounakchadha/Desktop/projects/deepfake/checkpoints/best_model.pth"

echo "══════════════════════════════════════"
echo "  Training Pipeline Status"
echo "══════════════════════════════════════"

# Check if pipeline process is still running
if ps aux | grep -q "[t]rain_setup.sh"; then
    echo "  Pipeline: RUNNING"
elif ps aux | grep -q "[k]aggle datasets download"; then
    echo "  Pipeline: Downloading..."
elif ps aux | grep -q "[t]rain.py"; then
    echo "  Pipeline: Training..."
else
    echo "  Pipeline: Not running (completed or not started)"
fi

# Dataset status
if [ -d "$SSD/REAL" ] && [ -d "$SSD/FAKE" ]; then
    REAL=$(ls "$SSD/REAL" 2>/dev/null | wc -l | tr -d ' ')
    FAKE=$(ls "$SSD/FAKE" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Dataset:  REAL=$REAL, FAKE=$FAKE images"
elif [ -d "$SSD/real_vs_fake" ]; then
    echo "  Dataset:  Extracted (organizing in progress)"
else
    SSD_SIZE=$(du -sh "$SSD" 2>/dev/null | cut -f1)
    echo "  Dataset:  Downloading... (SSD data size: $SSD_SIZE)"
fi

# Checkpoint status
if [ -f "$CKPT" ]; then
    AGE=$(( ($(date +%s) - $(stat -f %m "$CKPT")) / 60 ))
    SIZE=$(du -sh "$CKPT" | cut -f1)
    echo "  Model:    checkpoints/best_model.pth exists ($SIZE, ${AGE}m ago)"
else
    echo "  Model:    Not yet saved (training in progress or not started)"
fi

# Last few log lines
echo ""
echo "── Recent log output ──────────────────"
tail -15 "$LOG" 2>/dev/null || echo "  (no log yet)"
echo "═══════════════════════════════════════"
