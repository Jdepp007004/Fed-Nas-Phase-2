"""
server/aggregation.py
M2: Federated Optimization Engine — FedAvg, Momentum, Validation.
Owner: T Dheeraj Sai Skand
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import datetime  # noqa: E402
import numpy as np  # noqa: E402


# ─── Custom Exceptions ────────────────────────────────────────────────────────

class EmptyRoundError(Exception):
    """Raised when aggregate_fedavg() receives an empty update list."""


# ─── FedAvg ──────────────────────────────────────────────────────────────────

def aggregate_fedavg(client_updates: list, sample_counts: list) -> dict:
    """
    Canonical FedAvg with sample-proportional weighting.

    Parameters
    ----------
    client_updates : list[dict]  — weight dicts from each client
    sample_counts  : list[int]   — number of local training samples per client

    Returns
    -------
    dict — aggregated global weight dict

    Raises
    ------
    EmptyRoundError  if client_updates is empty
    """
    if not client_updates:
        raise EmptyRoundError("No client updates to aggregate.")

    if len(client_updates) != len(sample_counts):
        raise ValueError("client_updates and sample_counts must have the same length.")

    total_samples = sum(sample_counts)
    if total_samples == 0:
        raise EmptyRoundError("All clients report 0 samples.")

    # Union of all parameter keys present in any update
    all_keys = set()
    for upd in client_updates:
        all_keys.update(upd.keys())

    aggregated = {}
    for key in all_keys:
        # Collect (weight, array) for clients that include this key
        contributors = [
            (sample_counts[i], np.array(client_updates[i][key], dtype=np.float32))
            for i in range(len(client_updates))
            if key in client_updates[i]
        ]
        if not contributors:
            continue

        contrib_weights = np.array([w for w, _ in contributors], dtype=np.float64)
        contrib_arrays = [arr for _, arr in contributors]
        total_contrib = contrib_weights.sum()

        # Weighted average
        weighted_sum = np.zeros_like(contrib_arrays[0], dtype=np.float64)
        for w, arr in zip(contrib_weights, contrib_arrays):
            weighted_sum += (w / total_contrib) * arr.astype(np.float64)

        aggregated[key] = weighted_sum.astype(np.float32)

    return aggregated


# ─── Momentum Smoothing ───────────────────────────────────────────────────────

def update_with_momentum(
    current_global: dict,
    fedavg_aggregate: dict,
    momentum: float,
    velocity: dict,
) -> tuple:
    """
    Apply Nesterov-style server momentum to smooth the global update.

    v_t = beta * v_{t-1} + (1 - beta) * (fedavg_aggregate - current_global)
    new_global = current_global + v_t

    Parameters
    ----------
    current_global   : dict  — current global model weights
    fedavg_aggregate : dict  — raw FedAvg result
    momentum         : float — beta coefficient (default 0.9)
    velocity         : dict  — running velocity (zeros at round 0)

    Returns
    -------
    tuple: (new_global_weights: dict, updated_velocity: dict)
    """
    new_global = {}
    new_velocity = {}

    all_keys = set(current_global) | set(fedavg_aggregate)

    for key in all_keys:
        cur = np.array(current_global.get(key, np.zeros(1)), dtype=np.float64)
        agg = np.array(fedavg_aggregate.get(key, cur), dtype=np.float64)
        vel = np.array(velocity.get(key, np.zeros_like(cur)), dtype=np.float64)

        # Ensure shape compatibility
        if cur.shape != agg.shape:
            new_global[key] = agg.astype(np.float32)
            new_velocity[key] = np.zeros_like(agg, dtype=np.float32)
            continue

        delta = agg - cur
        new_vel = momentum * vel + (1.0 - momentum) * delta
        new_w = cur + new_vel

        new_global[key] = new_w.astype(np.float32)
        new_velocity[key] = new_vel.astype(np.float32)

    return new_global, new_velocity


# ─── Global Model Validation ──────────────────────────────────────────────────

def validate_global_model(
    global_weights: dict,
    val_dataloader,
    config: dict,
) -> dict:
    """
    Compute global validation metrics after aggregation.

    Parameters
    ----------
    global_weights  : dict        — freshly aggregated weights
    val_dataloader  : DataLoader  — server-side held-out validation set
    config          : dict        — model config (input_dim, max_depth, etc.)

    Returns
    -------
    dict: {round, global_val_rmse, global_tox_accuracy, global_auc, timestamp}
    """
    import torch
    from sklearn.metrics import roc_auc_score

    # Lazy import to avoid circular deps
    client_path = os.path.join(os.path.dirname(__file__), '..', 'client')
    if client_path not in sys.path:
        sys.path.insert(0, client_path)
    from supernet import Supernet, load_global_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Supernet(
        input_dim=config.get("input_dim", 512),
        max_depth=config.get("max_depth", 6),
        hidden_dim=config.get("hidden_dim", 256),
        num_toxicity_classes=config.get("num_toxicity_classes", 4),
    ).to(device)
    load_global_weights(model, global_weights, strict=False)
    model.eval()

    active_depth = config.get("max_depth", 6)
    all_reg_pred, all_reg_true = [], []
    all_tox_pred, all_tox_true = [], []
    all_bin_pred, all_bin_true = [], []

    with torch.no_grad():
        for batch in val_dataloader:
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

    rmse = float(np.sqrt(np.mean((reg_pred - reg_true) ** 2)))
    tox_acc = float(np.mean(tox_pred == tox_true))
    try:
        auc = float(roc_auc_score(bin_true.astype(int), bin_pred))
    except ValueError:
        auc = 0.0

    return {
        "global_val_rmse": rmse,
        "global_tox_accuracy": tox_acc,
        "global_auc": auc,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
