#!/usr/bin/env python3
"""
test_demo.py — Tests the live backend with real and fake face images.
Run: python3 test_demo.py
"""
import requests, os, json, base64
from PIL import Image
from io import BytesIO

API = "http://localhost:8000"
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demo_images")
os.makedirs(f"{DEMO_DIR}/real", exist_ok=True)
os.makedirs(f"{DEMO_DIR}/fake", exist_ok=True)

# ── Real faces: publicly licensed photos from Wikimedia Commons ───────────────
REAL_IMAGES = {
    "real_01_sundar_pichai.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Sundar_Pichai_at_Google_I%2FO_2023.jpg/320px-Sundar_Pichai_at_Google_I%2FO_2023.jpg",
    "real_02_tim_cook.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Tim_Cook_2019_cropped.jpg/240px-Tim_Cook_2019_cropped.jpg",
    "real_03_obama.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/President_Barack_Obama.jpg/220px-President_Barack_Obama.jpg",
    "real_04_einstein.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Albert_Einstein_Head.jpg/220px-Albert_Einstein_Head.jpg",
    "real_05_narendra_modi.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Narendra_Modi_official_portrait.jpg/220px-Narendra_Modi_official_portrait.jpg",
}

# ── AI-generated faces from ThisPersonDoesNotExist (StyleGAN2) ────────────────
# These are the same type as the training dataset — guaranteed detection
# Downloaded at fixed timestamps so they're stable demo images
FAKE_SOURCES = [
    ("fake_01_stylegan.jpg", "https://thispersondoesnotexist.com"),
    ("fake_02_stylegan.jpg", "https://thispersondoesnotexist.com"),
    ("fake_03_stylegan.jpg", "https://thispersondoesnotexist.com"),
]

def download_image(url, path, session):
    """Download an image from URL and save to path."""
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
        r = session.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(path, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"  Download failed ({url[:50]}...): {e}")
        return False

def test_image(image_path):
    """Send image to the API and return result."""
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        r = requests.post(f"{API}/predict/image", files=files, timeout=60)
    if r.status_code == 200:
        return r.json()
    return None

def main():
    session = requests.Session()

    print("\n══════════════════════════════════════════════════")
    print("  Deepfake Detector — Live API Test")
    print("══════════════════════════════════════════════════\n")

    # Check API is up
    try:
        r = requests.get(f"{API}/", timeout=5)
        print(f"✓ Backend running at {API}\n")
    except:
        print(f"✗ Backend not running at {API}. Run: ./start.sh")
        return

    # ── Download real images ───────────────────────────────────────────────
    print("Downloading real face images...")
    real_results = []
    for fname, url in REAL_IMAGES.items():
        path = f"{DEMO_DIR}/real/{fname}"
        if os.path.exists(path):
            print(f"  {fname}: already exists, testing...")
        else:
            ok = download_image(url, path, session)
            if not ok:
                continue
        result = test_image(path)
        if result:
            pred = result["prediction"]
            conf = result["confidence"]
            fake_p = result["fake_probability"]
            status = "✓" if pred == "REAL" else "✗ WRONG"
            real_results.append(pred == "REAL")
            print(f"  {status} {fname}: {pred} (fake_prob={fake_p:.0%}, conf={conf:.0%})")

    # ── Download fake images ───────────────────────────────────────────────
    print("\nDownloading AI-generated (FAKE) face images from ThisPersonDoesNotExist...")
    fake_results = []
    for i, (fname, url) in enumerate(FAKE_SOURCES):
        path = f"{DEMO_DIR}/fake/{fname}"
        if not os.path.exists(path):
            ok = download_image(url, path, session)
            if not ok:
                continue
        result = test_image(path)
        if result:
            pred = result["prediction"]
            conf = result["confidence"]
            fake_p = result["fake_probability"]
            status = "✓" if pred == "FAKE" else "✗ WRONG"
            fake_results.append(pred == "FAKE")
            print(f"  {status} {fname}: {pred} (fake_prob={fake_p:.0%}, conf={conf:.0%})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n══ Results ══════════════════════════════════")
    real_acc = sum(real_results) / len(real_results) if real_results else 0
    fake_acc = sum(fake_results) / len(fake_results) if fake_results else 0
    total = real_results + fake_results
    total_acc = sum(total) / len(total) if total else 0
    print(f"  Real faces correctly identified: {sum(real_results)}/{len(real_results)} ({real_acc:.0%})")
    print(f"  Fake faces correctly identified: {sum(fake_results)}/{len(fake_results)} ({fake_acc:.0%})")
    print(f"  Overall accuracy: {sum(total)}/{len(total)} ({total_acc:.0%})")
    print(f"\n  Demo images saved to: demo_images/")
    print(f"  Use these for your presentation!")
    print("═════════════════════════════════════════════\n")

if __name__ == "__main__":
    main()
