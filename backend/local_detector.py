"""
local_detector.py — Local ViT Deepfake Detector
=================================================
Uses 'prithivMLmods/Deep-Fake-Detector-v2-Model': a ViT fine-tuned specifically
for deepfake face detection. Labels: {0: 'Realism', 1: 'Deepfake'}.

IMPORTANT: The model was trained on FACE-CROPPED images.
We extract the face first before feeding to ViT to match training distribution.
"""

import io
import logging
import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
_model_cache = {}


def _extract_face_pil(image_bytes: bytes) -> Image.Image:
    """
    Extract face from raw image bytes using OpenCV Haar cascade.
    Falls back to full image if no face found.
    Returns a PIL Image ready for ViT preprocessing.
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        # Can't decode — return blank
        return Image.new("RGB", (224, 224), (128, 128, 128))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    h_img, w_img = img_bgr.shape[:2]
    margin = 20

    if len(faces) > 0:
        # Use the largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)
        face_crop = img_bgr[y1:y2, x1:x2]
        logger.info(f"Face detected at ({x},{y},{w},{h}) — cropping for ViT")
    else:
        # Fallback: center crop
        size = min(w_img, h_img)
        cx, cy = w_img // 2, h_img // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(w_img, cx + size // 2)
        y2 = min(h_img, cy + size // 2)
        face_crop = img_bgr[y1:y2, x1:x2]
        logger.info("No face detected — using center crop for ViT")

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def _load_model(device=None):
    """Load ViT model once and cache in memory."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["processor"], _model_cache["device"]

    logger.info(f"Loading local deepfake detector: {MODEL_ID}")

    try:
        from transformers import ViTForImageClassification, ViTImageProcessor

        processor = ViTImageProcessor.from_pretrained(MODEL_ID)
        model = ViTForImageClassification.from_pretrained(MODEL_ID)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        model = model.to(device)
        model.eval()

        _model_cache["model"] = model
        _model_cache["processor"] = processor
        _model_cache["device"] = device

        logger.info(f"Deepfake ViT loaded on {device}. Labels: {model.config.id2label}")
        return model, processor, device

    except Exception as e:
        logger.error(f"Failed to load deepfake detector: {e}")
        return None, None, None


def predict_local(image_bytes: bytes):
    """
    Run local ViT inference on raw image bytes.
    Crops face first to match training distribution.
    Returns dict with fake_probability, or None on failure.
    """
    try:
        model, processor, device = _load_model()
        if model is None:
            return None

        # Extract face crop (model trained on face images, not full photos)
        face_image = _extract_face_pil(image_bytes)

        inputs = processor(images=face_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        id2label = model.config.id2label

        # Find fake class index dynamically
        fake_idx = None
        for idx, lbl in id2label.items():
            if any(w in lbl.lower() for w in ("fake", "deepfake", "artificial", "generated")):
                fake_idx = idx
                break
        if fake_idx is None:
            fake_idx = 1  # Default for prithivMLmods: 1=Deepfake

        fake_prob = probs[fake_idx].item()
        all_scores = {id2label[i]: round(probs[i].item(), 4) for i in range(len(probs))}
        logger.info(f"ViT scores (face-cropped): {all_scores}")

        return {
            "fake_probability": round(fake_prob, 4),
            "source": "local_vit",
            "model": MODEL_ID,
            "all_scores": all_scores,
        }

    except Exception as e:
        logger.error(f"Local ViT inference error: {e}")
        return None
