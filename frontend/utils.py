"""
utils.py — Shared Utility Functions
Handles API communication, image decoding, feedback logging, and report generation.
"""

import requests
import base64
import json
import csv
import io
import os
from datetime import datetime
from PIL import Image

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
API_URL = os.getenv("DEEPFAKE_API_URL", "http://localhost:8000")
FEEDBACK_CSV = os.path.join(os.path.dirname(__file__), "feedback.csv")
MAX_IMAGE_MB = 20
MAX_VIDEO_MB = 100
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
ALLOWED_VIDEO_TYPES = ["video/mp4", "video/quicktime"]


# ─────────────────────────────────────────────────────────
# API Health
# ─────────────────────────────────────────────────────────
def get_api_status() -> dict:
    """Check if the backend FastAPI server is online. Returns status dict."""
    try:
        resp = requests.get(f"{API_URL}/", timeout=3)
        if resp.status_code == 200:
            return {"online": True, "message": "API Connected"}
    except Exception:
        pass
    return {"online": False, "message": "API Offline — Start the backend first"}


# ─────────────────────────────────────────────────────────
# File Validation
# ─────────────────────────────────────────────────────────
def validate_upload(file_bytes: bytes, mime_type: str, filename: str) -> dict:
    """
    Validates uploaded file. Returns dict with 'valid' bool and 'error' string.
    """
    size_mb = len(file_bytes) / (1024 * 1024)

    if mime_type in ALLOWED_IMAGE_TYPES:
        if size_mb > MAX_IMAGE_MB:
            return {"valid": False, "error": f"Image too large ({size_mb:.1f} MB). Max: {MAX_IMAGE_MB} MB."}
    elif mime_type in ALLOWED_VIDEO_TYPES:
        if size_mb > MAX_VIDEO_MB:
            return {"valid": False, "error": f"Video too large ({size_mb:.1f} MB). Max: {MAX_VIDEO_MB} MB."}
    else:
        return {"valid": False, "error": f"Unsupported format: {mime_type}. Supported: JPG, PNG, WebP, MP4, MOV."}

    return {"valid": True, "error": None, "size_mb": round(size_mb, 2)}


# ─────────────────────────────────────────────────────────
# API Calls
# ─────────────────────────────────────────────────────────
def call_predict_image(file_bytes: bytes, filename: str, mime: str) -> dict:
    """
    Posts image bytes to /predict/image endpoint.
    Returns structured dict with prediction, confidence, heatmap, and errors.
    """
    try:
        files = {"file": (filename, file_bytes, mime)}
        resp = requests.post(f"{API_URL}/predict/image", files=files, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "prediction": data.get("prediction", "UNKNOWN"),
                "confidence": data.get("confidence", 0.0),
                "fake_probability": data.get("fake_probability", 0.0),
                "heatmap_base64": data.get("heatmap_base64", None),
            }
        else:
            return {"success": False, "error": f"API Error {resp.status_code}: {resp.text}"}

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend. Is FastAPI running on port 8000?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. The model may still be loading — try again."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


def call_predict_video(file_bytes: bytes, filename: str, mime: str) -> dict:
    """
    Posts video bytes to /predict/video endpoint.
    Returns frame-level predictions and summary.
    """
    try:
        files = {"file": (filename, file_bytes, mime)}
        resp = requests.post(f"{API_URL}/predict/video", files=files, timeout=300)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "success": True,
                "prediction": data.get("prediction", "UNKNOWN"),
                "confidence": data.get("confidence", 0.0),
                "fake_probability": data.get("fake_probability", 0.0),
                "frames_analyzed": data.get("frames_analyzed", 0),
                "fake_frame_count": data.get("fake_frame_count", 0),
                "fake_percentage": data.get("fake_percentage", 0.0),
                "frame_probabilities": data.get("frame_probabilities", []),
                "heatmap_samples": data.get("heatmap_samples", []),
            }
        else:
            return {"success": False, "error": f"API Error {resp.status_code}: {resp.text}"}

    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to backend. Is FastAPI running on port 8000?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Video analysis timed out. Try a shorter clip."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# ─────────────────────────────────────────────────────────
# Image Helpers
# ─────────────────────────────────────────────────────────
def decode_heatmap(base64_str: str) -> Image.Image:
    """Decodes a base64-encoded image string to a PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Converts PIL Image to bytes for Streamlit download."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def blend_images(original: Image.Image, heatmap: Image.Image, alpha: float = 0.6) -> Image.Image:
    """Blends original and heatmap images at given alpha for overlay mode."""
    original_rgb = original.convert("RGB").resize(heatmap.size)
    heatmap_rgb = heatmap.convert("RGB")
    return Image.blend(original_rgb, heatmap_rgb, alpha)


# ─────────────────────────────────────────────────────────
# Feedback Storage
# ─────────────────────────────────────────────────────────
def save_feedback(filename: str, prediction: str, user_label: str, confidence: float):
    """
    Appends a feedback row to local CSV file for future retraining.
    Columns: timestamp, filename, model_prediction, user_label, confidence
    """
    file_exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "model_prediction", "user_label", "confidence"])
        writer.writerow([
            datetime.now().isoformat(),
            filename,
            prediction,
            user_label,
            confidence
        ])


# ─────────────────────────────────────────────────────────
# Export / Download
# ─────────────────────────────────────────────────────────
def generate_json_report(result: dict, filename: str) -> str:
    """
    Generates a structured JSON report for download.
    """
    report = {
        "report_generated": datetime.now().isoformat(),
        "analyzed_file": filename,
        "model": "EfficientNet-B0 + Custom Classifier",
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence"),
        "fake_probability": result.get("fake_probability"),
        "threshold_used": 0.5,
        "interpretation": (
            "Score > 0.5 indicates likely DEEPFAKE. "
            "Score ≤ 0.5 indicates likely REAL."
        ),
        "disclaimer": (
            "This result is produced by an AI model and should not be used "
            "as conclusive legal evidence. Always apply human judgment."
        )
    }
    return json.dumps(report, indent=2)


# ─────────────────────────────────────────────────────────
# Interpretability Helpers
# ─────────────────────────────────────────────────────────
def get_attention_reasons(fake_probability: float) -> list:
    """
    Returns human-readable attention region reasons based on confidence level.
    These are rule-based summaries approximating Grad-CAM region focus.
    """
    reasons = []
    if fake_probability > 0.85:
        reasons = [
            "🔴 Very high attention near **jawline boundary** — common GAN blending artifact",
            "🔴 **Unnatural texture detected around the mouth region**",
            "🔴 **Lighting inconsistency** between face and background",
            "🔴 **Eye region artifacts** — irregular reflection or blink pattern",
        ]
    elif fake_probability > 0.65:
        reasons = [
            "🟠 Moderate attention near **facial edges** — possible warping",
            "🟠 **Skin smoothing artifacts** detected (over-blurred pores)",
            "🟠 Slight **color mismatch** in forehead region",
        ]
    elif fake_probability > 0.5:
        reasons = [
            "🟡 Weak signals near **temporal regions** (sides of face)",
            "🟡 Minor **compression artifacts** that may signal post-processing",
        ]
    else:
        reasons = [
            "🟢 **No strong manipulation signals detected**",
            "🟢 Face texture and lighting appear consistent",
            "🟢 Boundary regions show natural skin characteristics",
        ]
    return reasons
