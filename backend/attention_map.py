"""
attention_map.py — ViT Attention Rollout Visualisation
=======================================================
Extracts multi-head self-attention from all transformer layers and computes
an "attention rollout" map showing which image patches the model focused on
when making its deepfake prediction.

Reference: Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers",
ACL 2020.
"""

import base64
import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def compute_attention_rollout(
    model,
    processor,
    device,
    face_image: Image.Image,
) -> Optional[str]:
    """
    Run a forward pass with output_attentions=True, then aggregate attention
    matrices across all heads and layers using rollout.
    Returns a base64-encoded JPEG heatmap overlaid on the face crop.
    """
    try:
        inputs = processor(images=face_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
        attentions = outputs.attentions  # each: (1, H, N, N)

        # ── Attention Rollout ──────────────────────────────────────────────
        # Start with identity matrix, then fold in each layer's averaged attention
        num_tokens = attentions[0].shape[-1]
        rollout = torch.eye(num_tokens, device="cpu")

        for attn in attentions:
            # Average over heads → (1, N, N)
            avg_heads = attn.mean(dim=1).squeeze(0).cpu()  # (N, N)
            # Add residual connection (0.5 * I + 0.5 * A)
            avg_heads = 0.5 * avg_heads + 0.5 * torch.eye(num_tokens)
            # Normalise rows
            avg_heads = avg_heads / avg_heads.sum(dim=-1, keepdim=True)
            rollout = torch.matmul(avg_heads, rollout)

        # Token 0 is [CLS] — its attention row tells us what each patch contributed
        cls_attention = rollout[0, 1:]  # drop CLS itself → (num_patches,)
        num_patches = cls_attention.shape[0]
        grid_size = int(num_patches ** 0.5)

        attn_map = cls_attention.reshape(grid_size, grid_size).numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # ── Overlay on face image ──────────────────────────────────────────
        face_np = np.array(face_image.resize((224, 224)))  # RGB
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

        attn_resized = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)
        attn_color = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(face_bgr, 0.45, attn_color, 0.55, 0)

        cv2.putText(overlay, "Attention Rollout", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf).decode('utf-8')

    except Exception as e:
        logger.warning(f"Attention rollout failed: {e}")
        return None
