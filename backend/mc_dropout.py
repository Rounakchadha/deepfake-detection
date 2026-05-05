"""
mc_dropout.py — Monte Carlo Dropout Uncertainty Estimation
===========================================================
Instead of a single deterministic forward pass, we enable dropout at inference
time and run N stochastic passes. The mean gives a calibrated probability;
the standard deviation gives an epistemic uncertainty estimate.

Reference: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning", ICML 2016.
"""

import torch
import numpy as np


def _enable_dropout(model: torch.nn.Module):
    """Switch all Dropout layers to train mode (active) while keeping BN in eval."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_dropout_predict(model, input_tensor: torch.Tensor, n_passes: int = 20) -> dict:
    """
    Run N stochastic forward passes with dropout active.

    Returns:
        mean_prob   — averaged fake probability across passes
        std_prob    — standard deviation (epistemic uncertainty)
        ci_lower    — 5th percentile
        ci_upper    — 95th percentile
    """
    model.eval()
    _enable_dropout(model)

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            if hasattr(model, 'predict'):
                p = model.predict(input_tensor).item()
            else:
                logits = model(input_tensor)
                # ImageFolder: FAKE=0, REAL=1 → sigmoid = P(REAL), so P(FAKE) = 1 - sigmoid
                p = 1.0 - torch.sigmoid(logits).item()
            probs.append(p)

    model.eval()  # restore full eval (re-disables dropout)

    arr = np.array(probs)
    return {
        "mc_mean_prob": round(float(arr.mean()), 4),
        "mc_std_prob": round(float(arr.std()), 4),
        "mc_ci_lower": round(float(np.percentile(arr, 5)), 4),
        "mc_ci_upper": round(float(np.percentile(arr, 95)), 4),
        "mc_n_passes": n_passes,
    }
