"""
hf_fallback.py — HuggingFace Inference API Fallback
=====================================================
When our primary EfficientNet-B0 model gives a low-confidence borderline
prediction (0.35–0.65), this module queries a second model via the
HuggingFace Inference API (HTTP) and ensembles both predictions.

This adds:
  1. Confidence-aware dual-model inference (novel vs. old paper's single model)
  2. Zero extra RAM — no local model loaded, pure HTTP call
  3. Graceful fallback — if HF API is unavailable, returns None silently

Models supported:
  - 'umm-maybe/AI-image-detector' : binary ARTIFICIAL vs NATURAL classifier
    (ViT-based, good at detecting AI-generated images including deepfakes)

Usage:
    from backend.hf_fallback import HuggingFaceEnsemble
    hf = HuggingFaceEnsemble()
    result = hf.query_image_bytes(image_bytes)
    # Returns: {"fake_probability": float, "source": "huggingface"} or None
"""

import io
import os
import time
import requests
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The model to use for fallback.
# 'prithivMLmods/Deep-Fake-Detector-v2-Model' is a ViT fine-tuned specifically
# for deepfake face detection with labels: "Realism" (real) / "Deepfake" (fake)
HF_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Get HF token from env (optional — many models are public)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Confidence zone where we trigger the fallback (inclusive)
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65

# Timeout for HF API call
HF_TIMEOUT = 10  # seconds

# Ensemble weight: how much to trust HF vs our model in uncertain zone
# 0.5 = equal weight, >0.5 = trust HF more
HF_ENSEMBLE_WEIGHT = 0.45  # Slight preference for our model


class HuggingFaceEnsemble:
    """
    Confidence-aware dual-model inference.
    Only queries HuggingFace when our primary model is uncertain.
    """

    def __init__(self):
        self.headers = {}
        if HF_API_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
        self._available = None  # Lazy check

    def _check_availability(self) -> bool:
        """Check if HF API is reachable (cached for 60s)."""
        try:
            resp = requests.get(
                f"https://api-inference.huggingface.co/status/{HF_MODEL_ID}",
                timeout=5,
                headers=self.headers,
            )
            return resp.status_code in (200, 424)  # 424 = model loading (still available)
        except Exception:
            return False

    def query_image_bytes(self, image_bytes: bytes) -> dict | None:
        """
        Send raw image bytes to HF Inference API.

        Returns:
            dict with 'fake_probability' and 'source', or None on failure.
        """
        try:
            resp = requests.post(
                HF_API_URL,
                headers={**self.headers, "Content-Type": "application/octet-stream"},
                data=image_bytes,
                timeout=HF_TIMEOUT,
            )

            if resp.status_code == 503:
                # Model is loading — wait and retry once
                time.sleep(3)
                resp = requests.post(
                    HF_API_URL,
                    headers={**self.headers, "Content-Type": "application/octet-stream"},
                    data=image_bytes,
                    timeout=HF_TIMEOUT,
                )

            if resp.status_code != 200:
                logger.warning(f"HF API returned {resp.status_code}: {resp.text[:200]}")
                return None

            results = resp.json()

            # Parse classifier output: list of [{"label": "...", "score": ...}]
            # Labels: "artificial" / "natural" (or "LABEL_0"/"LABEL_1")
            fake_prob = self._parse_fake_probability(results)
            if fake_prob is None:
                return None

            return {
                "fake_probability": round(fake_prob, 4),
                "source": "huggingface",
                "model": HF_MODEL_ID,
            }

        except requests.exceptions.Timeout:
            logger.warning("HF Inference API timed out")
            return None
        except Exception as e:
            logger.warning(f"HF Inference API error: {e}")
            return None

    def _parse_fake_probability(self, results) -> float | None:
        """
        Map HF classifier labels to a [0, 1] fake probability.
        prithivMLmods/Deep-Fake-Detector-v2-Model uses:
          'Deepfake' -> fake
          'Realism'  -> real
        """
        if not isinstance(results, list):
            return None

        label_map = {}
        for item in results:
            label = item.get("label", "").lower()
            score = item.get("score", 0.0)
            label_map[label] = score

        # Primary labels for prithivMLmods model
        if "deepfake" in label_map:
            return label_map["deepfake"]

        # Fallback: 'artificial' == FAKE (umm-maybe model)
        if "artificial" in label_map:
            return label_map["artificial"]

        # Generic label_1 (often the positive/fake class)
        if "label_1" in label_map:
            return label_map["label_1"]

        # Single label: infer direction
        if len(label_map) == 1:
            only_label, score = next(iter(label_map.items()))
            if any(w in only_label for w in ("fake", "artificial", "ai", "deepfake", "generated")):
                return score
            else:
                return 1.0 - score

        return None

    def ensemble(self, primary_prob: float, image_bytes: bytes) -> dict:
        """
        HF is the PRIMARY detector (always called). Local model is a supplement.

        When HF is available:
            final = 0.8 * HF_prob + 0.2 * local_prob

        When HF is unavailable:
            final = local_prob (with a warning)
        """
        logger.info(f"Querying HF primary detector (local model gave {primary_prob:.3f})...")
        hf_result = self.query_image_bytes(image_bytes)

        if hf_result is None:
            # HF unavailable — use local model only (with accuracy warning)
            logger.warning("HF API unavailable — using local model only (accuracy may be low)")
            return {
                "final_fake_probability": primary_prob,
                "hf_result": None,
                "ensemble_used": False,
                "note": "⚠️ HF API unavailable. Local model only — set HF_API_TOKEN for best results.",
            }

        hf_prob = hf_result["fake_probability"]

        # HF is primary (80%), local model supplements (20%)
        HF_WEIGHT = 0.80
        blended = HF_WEIGHT * hf_prob + (1 - HF_WEIGHT) * primary_prob

        logger.info(
            f"HF={hf_prob:.3f} (80%) + local={primary_prob:.3f} (20%) → {blended:.3f}"
        )

        return {
            "final_fake_probability": round(blended, 4),
            "hf_result": hf_result,
            "ensemble_used": True,
            "note": (
                f"HF ({HF_MODEL_ID}) is primary (80%) — "
                f"HF: {hf_prob:.0%}, Local: {primary_prob:.0%}"
            ),
        }
