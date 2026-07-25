"""
client/focal_loss.py
Focal Loss for class-imbalanced multi-class classification (Phase 3 — M8).

Focal loss (Lin et al., 2017) down-weights easy examples so the model
focuses on hard, misclassified examples. This is especially useful for the
toxicity multi-class head where class imbalance is common in clinical data.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where p_t is the model's estimated probability of the correct class.

Public API
----------
    from focal_loss import FocalLoss

    criterion = FocalLoss(gamma=2.0, alpha=None, reduction="mean")
    loss = criterion(logits, targets)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class (softmax) classification.

    Parameters
    ----------
    gamma     : float  — focusing parameter (0 = standard CE, 2 = typical FL)
    alpha     : float or list[float] or None
                If float, same weight for all classes.
                If list, per-class weights (length = num_classes).
                If None, equal weights (default).
    reduction : str    — "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float | list] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}")

        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.tensor(float(alpha))
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, C) float tensor — raw (pre-softmax) class scores
        targets : (N,)   long tensor  — ground-truth class indices

        Returns
        -------
        torch.Tensor — scalar loss (or per-sample if reduction='none')
        """
        # Standard cross-entropy: -log(p_t)
        log_probs = F.log_softmax(logits, dim=1)                   # (N, C)
        probs = torch.exp(log_probs)                                # (N, C)

        # Gather probability of the true class: p_t
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)           # (N,)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - pt) ** self.gamma

        # Apply per-class alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            if alpha.dim() == 0:
                # Scalar alpha
                alpha_t = alpha.expand(targets.size(0))
            else:
                # Per-class alpha
                alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt   # (N,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification (sigmoid).

    Replaces BCEWithLogitsLoss in train_loop.py for the binary survival head.

    Parameters
    ----------
    gamma     : float  — focusing parameter (0 = standard BCE, 2 = typical FL)
    alpha     : float  — positive class weight in [0, 1]
    reduction : str    — "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N,) float tensor — raw sigmoid inputs
        targets : (N,) float tensor — binary labels {0.0, 1.0}

        Returns
        -------
        torch.Tensor — scalar loss
        """
        probs = torch.sigmoid(logits)
        pt    = torch.where(targets == 1.0, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1.0,
                              torch.tensor(self.alpha),
                              torch.tensor(1.0 - self.alpha))
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
