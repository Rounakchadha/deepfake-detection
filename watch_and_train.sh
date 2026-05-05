#!/bin/bash
# watch_and_train.sh
# Waits for the zip to finish downloading, then extracts and trains.
# Run: bash watch_and_train.sh  (leave terminal open or run in background)

SSD="/Volumes/ADATA SE880/deepfake_data"
ZIP="$SSD/140k-real-and-fake-faces.zip"
PROJECT="/Users/rounakchadha/Desktop/projects/deepfake"
VENV="$PROJECT/venv/bin/python3"

echo "Waiting for download to complete..."

# Wait for the zip file to exist and stop growing
PREV_SIZE=0
STABLE_COUNT=0
while true; do
    if [ ! -f "$ZIP" ]; then
        echo "  Waiting for zip file to appear..."
        sleep 10
        continue
    fi

    CURR_SIZE=$(stat -f%z "$ZIP" 2>/dev/null || echo 0)
    SIZE_MB=$((CURR_SIZE / 1048576))
    echo "  $(date '+%H:%M:%S') — Downloaded: ${SIZE_MB} MB"

    if [ "$CURR_SIZE" -eq "$PREV_SIZE" ] && [ "$CURR_SIZE" -gt 1000000000 ]; then
        STABLE_COUNT=$((STABLE_COUNT + 1))
        if [ "$STABLE_COUNT" -ge 3 ]; then
            echo "Download complete! (${SIZE_MB} MB)"
            break
        fi
    else
        STABLE_COUNT=0
    fi
    PREV_SIZE=$CURR_SIZE
    sleep 15
done

# Extract
echo ""
echo "Extracting zip to SSD..."
$VENV -c "
import zipfile, os
zip_path = '$ZIP'
extract_to = '$SSD'
print(f'Extracting {os.path.getsize(zip_path)/1e9:.1f} GB zip...')
with zipfile.ZipFile(zip_path, 'r') as z:
    members = z.namelist()
    print(f'Total files: {len(members):,}')
    z.extractall(extract_to)
print('Extraction complete.')
"

# Organize into REAL/ FAKE/
echo ""
echo "Organizing into REAL/ and FAKE/ folders..."
$VENV -c "
import os, shutil

ssd = '$SSD'
real_out = os.path.join(ssd, 'REAL')
fake_out = os.path.join(ssd, 'FAKE')
os.makedirs(real_out, exist_ok=True)
os.makedirs(fake_out, exist_ok=True)

for split in ['train', 'valid']:
    for cls, out in [('real', real_out), ('fake', fake_out)]:
        src = os.path.join(ssd, 'real_vs_fake', 'real-vs-fake', split, cls)
        if not os.path.exists(src):
            print(f'Missing: {src}')
            continue
        files = os.listdir(src)
        print(f'  Copying {split}/{cls}: {len(files):,} files...')
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(out, f))

real_count = len(os.listdir(real_out))
fake_count = len(os.listdir(fake_out))
print(f'Done. REAL={real_count:,} FAKE={fake_count:,}')
"

# Start training
echo ""
echo "Starting EfficientNet training..."
DATA_DIR="$SSD" \
CHECKPOINT_DIR="$PROJECT/checkpoints" \
BATCH_SIZE="16" \
NUM_WORKERS="4" \
    $VENV "$PROJECT/train.py" 2>&1 | tee "$SSD/training_output.log"

# After training: activate ensemble
echo "Activating EfficientNet ensemble..."
sed -i '' 's/EFFICIENTNET_WEIGHT = 0.0/EFFICIENTNET_WEIGHT = 0.3/' \
    "$PROJECT/backend/inference.py" 2>/dev/null

# Restart backend
pkill -f uvicorn 2>/dev/null; sleep 2
nohup "$PROJECT/venv/bin/python3" -m uvicorn backend.api:app \
    --host 0.0.0.0 --port 8000 > "$PROJECT/backend.log" 2>&1 &

echo ""
echo "ALL DONE! New model active. Run ./start.sh to restart everything cleanly."
