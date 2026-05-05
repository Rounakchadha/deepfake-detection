"""
fft_analysis.py — Frequency Domain Forensics
=============================================
GANs leave characteristic periodic artifacts in the high-frequency spectrum
of generated images. This module detects those fingerprints.

Reference: Dzanic et al. (2020) "Fourier Spectrum Discrepancies in Deep Network
Generated Images", NeurIPS 2020.
"""

import base64
import cv2
import numpy as np
from typing import Optional


def compute_fft_analysis(image_bytes: bytes) -> Optional[dict]:
    """
    Compute 2D FFT on the image and return:
      - fft_heatmap_base64: colored log-magnitude spectrum
      - high_freq_energy_ratio: fraction of energy in high frequencies (>0.4 = suspicious)
      - spectral_peak_score: normalised count of anomalous periodic peaks (GAN grid artifact)
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None

    # Work on luma channel (most informative for GAN artifacts)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    luma = img_yuv[:, :, 0].astype(np.float32)
    luma = cv2.resize(luma, (256, 256))

    # 2D FFT with shift so DC is at centre
    fft = np.fft.fft2(luma)
    fshift = np.fft.fftshift(fft)
    magnitude = np.abs(fshift)

    # ── Visualisation ──────────────────────────────────────────────────────
    log_mag = np.log1p(magnitude)
    norm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    # Annotate with a circle marking the low/high-freq boundary
    h, w = colored.shape[:2]
    cy, cx = h // 2, w // 2
    boundary_r = min(h, w) // 6
    cv2.circle(colored, (cx, cy), boundary_r, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(colored, "FFT Spectrum", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
    fft_b64 = base64.b64encode(buf).decode('utf-8')

    # ── Metrics ────────────────────────────────────────────────────────────
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_from_centre = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    total_energy = float(np.sum(magnitude ** 2))
    low_mask = dist_from_centre <= boundary_r
    low_energy = float(np.sum(magnitude[low_mask] ** 2))
    high_freq_ratio = round(1.0 - low_energy / (total_energy + 1e-8), 4)

    # Spectral peak score: GANs produce a characteristic grid of periodic peaks.
    # We detect anomalously bright isolated spots in the high-freq ring.
    high_only = magnitude.copy()
    high_only[low_mask] = 0
    mean_hf = float(np.mean(high_only))
    std_hf = float(np.std(high_only))
    threshold = mean_hf + 4 * std_hf
    peak_count = int(np.sum(high_only > threshold))
    # Normalise: typical real images have ~0-5 peaks, GANs 20-80+
    spectral_peak_score = round(min(peak_count / 80.0, 1.0), 4)

    return {
        "fft_heatmap_base64": fft_b64,
        "high_freq_energy_ratio": high_freq_ratio,
        "spectral_peak_score": spectral_peak_score,
    }
