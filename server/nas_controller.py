"""
server/nas_controller.py
M2: NAS depth search and subnet selection logic.
Owner: T Dheeraj Sai Skand
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.model_schema import MAX_DEPTH, DEFAULT_ACTIVE_DEPTH


# ─── Depth Lookup Table ───────────────────────────────────────────────────────
# Maps (ram_gb_bucket, has_gpu) → recommended active_depth
# Expandable to a learned policy in future iterations.

_DEPTH_LOOKUP = [
    # (min_ram_gb, has_gpu, min_data_size, max_data_size, depth)
    (32, True,  0,       1_000_000, 6),
    (16, True,  0,       1_000_000, 5),
    (8,  True,  0,       1_000_000, 4),
    (16, False, 5_000,   1_000_000, 5),
    (8,  False, 2_000,   1_000_000, 4),
    (4,  False, 500,     1_000_000, 3),
    (0,  False, 0,       1_000_000, 2),
]

# Per-client cache: client_id → assigned depth
_depth_cache: dict = {}


# ─── Public Functions ─────────────────────────────────────────────────────────

def recommend_subnet_depth(client_id: str, client_profile: dict) -> int:
    """
    Map a client's hardware profile to an optimal subnet depth.

    Parameters
    ----------
    client_id      : str  — unique client identifier (for caching and logging)
    client_profile : dict — {'ram_gb': float, 'cpu_cores': int,
                             'gpu_available': bool, 'local_data_size': int}

    Returns
    -------
    int — recommended active_depth in range [2, MAX_DEPTH]
    """
    ram_gb    = float(client_profile.get("ram_gb", 4))
    has_gpu   = bool(client_profile.get("gpu_available", False))
    data_size = int(client_profile.get("local_data_size", 0))

    depth = 2  # conservative fallback
    for min_ram, needs_gpu, min_data, max_data, d in _DEPTH_LOOKUP:
        gpu_ok   = (not needs_gpu) or has_gpu
        ram_ok   = ram_gb >= min_ram
        data_ok  = min_data <= data_size <= max_data
        if ram_ok and gpu_ok and data_ok:
            depth = d
            break

    # Clamp to [2, MAX_DEPTH]
    depth = max(2, min(depth, MAX_DEPTH))

    _depth_cache[client_id] = depth
    return depth


def evaluate_architecture_candidates(
    updates_by_depth: dict,
    global_weights: dict,
) -> int:
    """
    Compare validation loss contribution from client groups using different
    depth values, then select the depth with the best average improvement
    per additional compute cost.

    Parameters
    ----------
    updates_by_depth : dict[int, list]  — {depth: [weight_dicts]}
    global_weights   : dict             — current global weights

    Returns
    -------
    int — globally recommended depth for the next round
    """
    if not updates_by_depth:
        return DEFAULT_ACTIVE_DEPTH

    # Import aggregation locally to avoid circular imports
    from aggregation import aggregate_fedavg

    best_depth = DEFAULT_ACTIVE_DEPTH
    best_score = float("inf")

    for depth, updates in updates_by_depth.items():
        if not updates:
            continue

        # Mini-aggregate for this depth group
        sample_counts = [u.get("num_samples", 100) for u in updates]
        weight_dicts  = [u.get("weights", {}) if isinstance(u, dict) else u for u in updates]

        try:
            mini_agg = aggregate_fedavg(weight_dicts, sample_counts)
        except Exception:
            continue

        # Compare aggregate vs current global using L2 norm of delta
        # (proxy for improvement magnitude per unit of depth cost)
        delta_norm = 0.0
        shared_keys = set(mini_agg) & set(global_weights)
        for key in shared_keys:
            import numpy as np
            delta = np.array(mini_agg[key]) - np.array(global_weights.get(key, mini_agg[key]))
            delta_norm += float(np.linalg.norm(delta))

        # Cost proxy: higher depth costs more compute
        depth_cost = float(depth)
        score = depth_cost / max(delta_norm, 1e-8)  # lower is better

        if score < best_score:
            best_score = best_depth
            best_depth = int(depth)

    return max(2, min(best_depth, MAX_DEPTH))
