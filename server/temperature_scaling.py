"""
server/temperature_scaling.py
Post-hoc calibration via temperature scaling (Phase 3 — M5).

After each round, the global model's classification confidence can be
miscalibrated (overconfident or underconfident). Temperature scaling fits
a single scalar T to the validation logits so that:

    p_calibrated = softmax(logits / T)

This is applied to BOTH the toxicity (multi-class) and binary heads.

References
----------
  Guo et al. "On Calibration of Modern Neural Networks." ICML 2017.

Public API
----------
    from temperature_scaling import TemperatureScaler

    scaler = TemperatureScaler()
    T_tox, T_bin = scaler.fit(model, val_loader, active_depth=4)
    proj_update = {"temperature_tox": T_tox, "temperature_bin": T_bin}

    # At inference (client):
    from temperature_scaling import apply_temperature
    calibrated_logits = apply_temperature(raw_logits, T)
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass


def apply_temperature(logits: "torch.Tensor", T: float) -> "torch.Tensor":
    """
    Apply temperature scaling to raw logits.

    Parameters
    ----------
    logits : torch.Tensor — raw model output (before softmax/sigmoid)
    T      : float        — temperature (T=1.0 = no scaling)

    Returns
    -------
    torch.Tensor — scaled logits (apply softmax/sigmoid afterwards)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for temperature scaling")
    T = max(T, 1e-6)   # guard against division by zero
    return logits / T


class TemperatureScaler:
    """
    Fits temperature T on a validation DataLoader using NLL minimisation.

    Fits separately for the toxicity head (multi-class) and binary head
    (binary cross-entropy) since they may need different calibrations.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 50) -> None:
        self.lr = lr
        self.max_iter = max_iter

    def fit(
        self,
        model,
        val_loader,
        active_depth: int = 4,
        device: Optional[str] = None,
    ) -> tuple[float, float]:
        """
        Fit temperature parameters T_tox and T_bin.

        Parameters
        ----------
        model       : Supernet instance with loaded global weights
        val_loader  : DataLoader yielding (X, y_reg, y_tox, y_bin)
        active_depth: active subnet depth to use for inference
        device      : optional torch device string

        Returns
        -------
        (T_tox, T_bin) — fitted temperatures (both > 0)
        """
        if not _TORCH_AVAILABLE:
            logger.warning("torch not available — returning T=1.0 for both heads")
            return 1.0, 1.0

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(dev).eval()

        all_tox_logits, all_tox_true = [], []
        all_bin_logits, all_bin_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                X, _, y_tox, y_bin = [t.to(dev) for t in batch]
                preds = model.forward_multi_head(X, active_depth)
                all_tox_logits.append(preds["toxicity"].cpu())
                all_tox_true.append(y_tox.cpu())
                all_bin_logits.append(preds["binary"].cpu())
                all_bin_true.append(y_bin.cpu())

        tox_logits = torch.cat(all_tox_logits)
        tox_true   = torch.cat(all_tox_true)
        bin_logits = torch.cat(all_bin_logits).squeeze(1)
        bin_true   = torch.cat(all_bin_true)

        T_tox = self._fit_temperature(
            tox_logits, tox_true, "ce", label="toxicity"
        )
        T_bin = self._fit_temperature(
            bin_logits, bin_true, "bce", label="binary"
        )
        return T_tox, T_bin

    def _fit_temperature(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        loss_type: str,
        label: str = "",
    ) -> float:
        """Optimise a single scalar T to minimise NLL."""
        T = nn.Parameter(torch.ones(1))
        optimiser = torch.optim.LBFGS([T], lr=self.lr, max_iter=self.max_iter)

        if loss_type == "ce":
            criterion = nn.CrossEntropyLoss()
            def closure():
                optimiser.zero_grad()
                scaled = logits / T.clamp(min=1e-6)
                loss = criterion(scaled, labels)
                loss.backward()
                return loss
        else:
            criterion = nn.BCEWithLogitsLoss()
            def closure():
                optimiser.zero_grad()
                scaled = logits / T.clamp(min=1e-6)
                loss = criterion(scaled, labels.float())
                loss.backward()
                return loss

        optimiser.step(closure)
        T_val = float(T.item())
        T_val = max(0.01, min(T_val, 10.0))   # clamp to reasonable range
        logger.info("Temperature scaling [%s]: T=%.4f", label, T_val)
        return T_val
