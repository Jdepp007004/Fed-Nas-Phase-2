"""
client/train_loop.py
M1/M2: Local FedProx training loop.
Owner: Praneeth Raj V (M1) / T Dheeraj Sai Skand (M2 FedProx integration)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
from typing import Iterator, NamedTuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from supernet import Supernet, compute_joint_loss, get_subnet_weights
from shared.model_schema import (
    DEFAULT_LR,
    DEFAULT_EPOCHS,
    DEFAULT_FEDPROX_MU,
    DEFAULT_CLIP_NORM,
    DEFAULT_TASK_WEIGHTS,
    DEFAULT_ACTIVE_DEPTH,
)


# ─── Config ──────────────────────────────────────────────────────────────────

class TrainConfig(NamedTuple):
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    active_depth: int = DEFAULT_ACTIVE_DEPTH
    fedprox_mu: float = DEFAULT_FEDPROX_MU
    task_weights: dict = None       # will default to DEFAULT_TASK_WEIGHTS
    clip_norm: float = DEFAULT_CLIP_NORM


# ─── FedProx Penalty ─────────────────────────────────────────────────────────

def apply_fedprox_penalty(
    local_params: Iterator,
    global_params: Iterator,
    mu: float,
) -> torch.Tensor:
    """
    Compute the FedProx proximal regularisation term:
        (mu/2) * sum(||w_local - w_global||^2)

    Parameters
    ----------
    local_params  : model.parameters() of the currently-training local model
    global_params : frozen parameter tensors from the global model snapshot
    mu            : FedProx regularisation coefficient

    Returns
    -------
    torch.Tensor — scalar penalty (to be added to total_loss before .backward())
    """
    penalty = torch.tensor(0.0, requires_grad=False)
    for w_local, w_global in zip(local_params, global_params):
        penalty = penalty + torch.norm(w_local - w_global.detach()) ** 2
    return (mu / 2.0) * penalty


# ─── Main Training Loop ──────────────────────────────────────────────────────

def run_local_training(
    model: Supernet,
    dataloader: DataLoader,
    config: TrainConfig = None,
    axes: dict = None,          # optional: visualizer axes for live loss plot
) -> dict:
    """
    Execute local federated training for a configured number of epochs.

    Parameters
    ----------
    model      : Supernet pre-loaded with global weights
    dataloader : (train_loader, val_loader) tuple — as returned by create_federated_dataloader()
    config     : TrainConfig named tuple
    axes       : optional Matplotlib axes dict for live loss updates

    Returns
    -------
    dict: {
        'weights':     dict (from get_subnet_weights()),
        'num_samples': int,
        'metrics':     {'loss': float, 'val_rmse': float, 'val_acc_tox': float, 'val_auc': float}
    }
    """
    if config is None:
        config = TrainConfig()

    task_weights = config.task_weights if config.task_weights else DEFAULT_TASK_WEIGHTS

    # Unpack dataloader tuple
    if isinstance(dataloader, (tuple, list)):
        train_loader, val_loader = dataloader
    else:
        train_loader = dataloader
        val_loader = None

    # Freeze a copy of the global model for FedProx penalty
    global_model_snapshot = copy.deepcopy(model)
    for p in global_model_snapshot.parameters():
        p.requires_grad_(False)
    global_params = list(global_model_snapshot.parameters())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    global_model_snapshot = global_model_snapshot.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train()
    epoch_losses = []
    total_samples = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # batch = (X, y_reg, y_tox, y_bin)
            X, y_reg, y_tox, y_bin = [t.to(device) for t in batch]

            optimizer.zero_grad()

            predictions = model.forward_multi_head(X, config.active_depth)
            targets = {
                "regression": y_reg,
                "toxicity": y_tox,
                "binary": y_bin,
            }

            joint_loss, breakdown = compute_joint_loss(predictions, targets, task_weights)

            # FedProx proximal penalty — pass a fresh iterator each batch
            fedprox_penalty = apply_fedprox_penalty(
                model.parameters(), iter(list(global_params)), config.fedprox_mu
            )
            total_loss = joint_loss + fedprox_penalty

            # Gradient clipping
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1
            if epoch == 0:
                total_samples += X.size(0)

        mean_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_losses.append(mean_epoch_loss)

        # Live loss update via visualizer if axes provided
        if axes is not None:
            try:
                from visualizer import update_local_loss
                update_local_loss(axes, epoch_losses)
            except Exception:
                pass

    # ── Validation ──────────────────────────────────────────────────────────
    val_rmse, val_acc_tox, val_auc = 0.0, 0.0, 0.0

    if val_loader is not None:
        model.eval()
        val_rmse, val_acc_tox, val_auc = _run_validation(
            model, val_loader, config.active_depth, task_weights, device
        )

    # ── Package results ─────────────────────────────────────────────────────
    subnet_weights = get_subnet_weights(model, config.active_depth)

    return {
        "weights": subnet_weights,
        "num_samples": total_samples,
        "metrics": {
            "loss": epoch_losses[-1] if epoch_losses else 0.0,
            "val_rmse": val_rmse,
            "val_acc_tox": val_acc_tox,
            "val_auc": val_auc,
        },
    }


def _run_validation(model, val_loader, active_depth, task_weights, device):
    """Internal helper: compute validation RMSE, toxicity accuracy, and AUC-ROC."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    all_reg_pred, all_reg_true = [], []
    all_tox_pred, all_tox_true = [], []
    all_bin_pred, all_bin_true = [], []

    with torch.no_grad():
        for batch in val_loader:
            X, y_reg, y_tox, y_bin = [t.to(device) for t in batch]
            preds = model.forward_multi_head(X, active_depth)

            all_reg_pred.append(preds["regression"].squeeze(1).cpu().numpy())
            all_reg_true.append(y_reg.cpu().numpy())
            all_tox_pred.append(preds["toxicity"].argmax(dim=1).cpu().numpy())
            all_tox_true.append(y_tox.cpu().numpy())
            all_bin_pred.append(torch.sigmoid(preds["binary"]).squeeze(1).cpu().numpy())
            all_bin_true.append(y_bin.cpu().numpy())

    reg_pred = np.concatenate(all_reg_pred)
    reg_true = np.concatenate(all_reg_true)
    tox_pred = np.concatenate(all_tox_pred)
    tox_true = np.concatenate(all_tox_true)
    bin_pred = np.concatenate(all_bin_pred)
    bin_true = np.concatenate(all_bin_true)

    val_rmse = float(np.sqrt(np.mean((reg_pred - reg_true) ** 2)))
    val_acc_tox = float(np.mean(tox_pred == tox_true))
    try:
        val_auc = float(roc_auc_score(bin_true.astype(int), bin_pred))
    except ValueError:
        val_auc = 0.0

    return val_rmse, val_acc_tox, val_auc
