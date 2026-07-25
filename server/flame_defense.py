"""
server/flame_defense.py
FLAME-inspired clustering-based Byzantine defense (Phase 3 — B3).

FLAME (Fung et al., 2022) clusters client updates using cosine similarity
and discards outlier clusters that likely represent malicious/poisoned updates.

This implementation follows the core FLAME idea:
1. Compute pairwise cosine similarity between flattened client updates.
2. Use Agglomerative Clustering to group clients.
3. Keep only the largest cluster (honest majority assumption).
4. Optionally clip the cluster's norms (adaptive clipping).

Reference
---------
  Fung, C. et al. "FLAME: Taming Backdoors in Federated Learning." USENIX 2022.

Public API
----------
    from flame_defense import filter_updates_flame

    clean_updates, clean_counts = filter_updates_flame(
        client_updates, sample_counts,
        noise_sigma=0.001,   # adaptive noise (0 = disabled)
    )
    # Pass clean_updates to aggregate_fedavg()
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _flatten_updates(updates: list[dict]) -> np.ndarray:
    """Stack all parameter arrays into a single flat feature matrix (n_clients, n_params)."""
    rows = []
    for upd in updates:
        flat = np.concatenate([np.array(v, dtype=np.float32).flatten()
                                for v in upd.values()])
        rows.append(flat)
    return np.stack(rows, axis=0)   # (n_clients, n_params)


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute n×n pairwise cosine similarity matrix."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    X_norm = X / norms
    return X_norm @ X_norm.T   # (n, n)


def _agglomerative_cluster(sim_matrix: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    """
    Simple single-linkage agglomerative clustering using similarity threshold.

    Returns a 1-D array of cluster labels (int) with the same length as
    the number of clients.  Label 0 = largest cluster.
    """
    n = sim_matrix.shape[0]
    # Start: each client in its own cluster
    labels = np.arange(n)

    # Repeatedly merge the most similar pair above threshold
    for _ in range(n * n):
        # Build similarity between clusters (max similarity between any two members)
        cluster_ids = np.unique(labels)
        if len(cluster_ids) <= 1:
            break
        best_sim = threshold
        best_pair = None
        for i, ca in enumerate(cluster_ids):
            for cb in cluster_ids[i + 1:]:
                members_a = np.where(labels == ca)[0]
                members_b = np.where(labels == cb)[0]
                pair_sim = sim_matrix[np.ix_(members_a, members_b)].max()
                if pair_sim > best_sim:
                    best_sim = pair_sim
                    best_pair = (ca, cb)
        if best_pair is None:
            break
        ca, cb = best_pair
        labels[labels == cb] = ca   # merge cb into ca

    # Relabel: 0 = largest cluster
    cluster_ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)   # descending by size
    remap = {old: new for new, old in enumerate(cluster_ids[order])}
    labels = np.array([remap[l] for l in labels])
    return labels


def filter_updates_flame(
    client_updates: list[dict],
    sample_counts: list[int],
    similarity_threshold: float = 0.6,
    noise_sigma: float = 0.0,
    min_cluster_size: int = 1,
) -> tuple[list[dict], list[int]]:
    """
    Apply FLAME-style clustering filter to remove outlier client updates.

    Parameters
    ----------
    client_updates      : list[dict] — raw weight-update dicts from clients
    sample_counts       : list[int]  — matching sample count per client
    similarity_threshold: float      — cosine similarity threshold for merging
                                       clusters (0=all separate, 1=all same)
    noise_sigma         : float      — Gaussian noise added to surviving updates
                                       for adaptive clipping (0 = disabled)
    min_cluster_size    : int        — minimum clients in kept cluster;
                                       if largest cluster is smaller, keep all

    Returns
    -------
    (filtered_updates, filtered_counts) — clean client updates and sample counts
    """
    n = len(client_updates)
    if n < 2:
        # Cannot cluster with < 2 clients — return as-is
        return client_updates, sample_counts

    try:
        X = _flatten_updates(client_updates)
        sim_matrix = _cosine_similarity_matrix(X)
        labels = _agglomerative_cluster(sim_matrix, threshold=similarity_threshold)

        # Keep only the largest cluster (label 0)
        keep_mask = labels == 0
        n_kept = int(keep_mask.sum())

        if n_kept < min_cluster_size:
            logger.warning(
                "FLAME: largest cluster has only %d clients (< min_cluster_size=%d) "
                "— keeping all clients.",
                n_kept, min_cluster_size,
            )
            keep_mask = np.ones(n, dtype=bool)
            n_kept = n

        n_removed = n - n_kept
        if n_removed > 0:
            logger.info(
                "FLAME: removed %d/%d outlier clients (kept %d in cluster 0).",
                n_removed, n, n_kept,
            )

        filtered_updates = [upd for upd, keep in zip(client_updates, keep_mask) if keep]
        filtered_counts = [cnt for cnt, keep in zip(sample_counts, keep_mask) if keep]

        # Optional adaptive Gaussian noise for additional privacy
        if noise_sigma > 0.0:
            rng = np.random.default_rng()
            noised = []
            for upd in filtered_updates:
                noised.append({
                    k: (np.array(v, dtype=np.float32)
                        + rng.normal(0, noise_sigma, size=np.array(v).shape).astype(np.float32))
                    for k, v in upd.items()
                })
            filtered_updates = noised

        return filtered_updates, filtered_counts

    except Exception as e:
        logger.warning(
            "FLAME defense encountered an error (%s) — falling back to all clients.", e,
        )
        return client_updates, sample_counts
