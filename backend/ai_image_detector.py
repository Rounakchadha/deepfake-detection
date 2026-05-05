"""
ai_image_detector.py — General AI-Generated Image Detector
===========================================================
Detects AI-generated images of ALL types:
  - DALL-E, Midjourney, Stable Diffusion generated scenes (Trump in Pope costume, etc.)
  - StyleGAN / AI portraits of non-existent people
  - Any image that looks AI-generated to a human

Model: haywoodsloan/ai-generated-image-classifier
  A ViT fine-tuned on real vs AI-generated images from diverse generators
  (SD, DALL-E, Midjourney, GAN, etc.)
"""

import logging
import torch
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

MODEL_ID = "haywoodsloan/ai-image-detector-deploy"
_cache = {}


def _load():
    if "model" in _cache:
        return _cache["model"], _cache["processor"], _cache["device"]

    try:
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        logger.info(f"Loading general AI image detector: {MODEL_ID}")

        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID, low_cpu_mem_usage=False)

        # Swin transformers have MPS compatibility issues — use CPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = model.to(device).eval()
        _cache.update({"model": model, "processor": processor, "device": device})
        logger.info(f"AI image detector ready on {device}. Labels: {model.config.id2label}")
        return model, processor, device
    except Exception as e:
        logger.error(f"Failed to load AI image detector: {e}", exc_info=True)
        return None, None, None


def warmup():
    model, _, _ = _load()
    if model is None:
        logger.error("AI image detector warmup failed — will return None during inference")
    else:
        logger.info("AI image detector warmup OK")


def predict_ai_generated(image_bytes: bytes):
    """
    Returns probability that image is AI-generated (any type: DALL-E, SD, GAN, etc.)
    Works on whole scenes, not just faces.
    """
    try:
        model, processor, device = _load()
        if model is None:
            return None

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
        id2label = model.config.id2label

        # Find the "AI/artificial" class (Organika model: {0: 'artificial', 1: 'human'})
        ai_idx = None
        for idx, lbl in id2label.items():
            if any(w in lbl.lower() for w in ("fake", "ai", "artificial", "generated", "synthetic")):
                ai_idx = idx
                break
        if ai_idx is None:
            ai_idx = 0  # Organika model: class 0 = artificial

        ai_prob = probs[ai_idx].item()
        all_scores = {id2label[i]: round(probs[i].item(), 4) for i in range(len(probs))}
        logger.info(f"AI image detector scores: {all_scores}")

        return {
            "ai_generated_probability": round(ai_prob, 4),
            "model": MODEL_ID,
            "all_scores": all_scores,
        }

    except Exception as e:
        logger.error(f"AI image detector error: {e}")
        return None
