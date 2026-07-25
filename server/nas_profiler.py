"""
server/nas_profiler.py
FLOP-aware NAS with zero-cost proxies (Phase 3 — M6).

Zero-cost proxies estimate network quality without training:
  - FLOP count: compute budget (operations per forward pass)
  - Synflow: gradient-magnitude proxy — sum of |param * grad| at init
  - Grad-norm: L2 norm of gradients at random input

These let the NAS controller rank subnet depths instantly, without
any data or training, enabling rapid architecture search.

Public API
----------
    from nas_profiler import profile_depth_candidates

    scores = profile_depth_candidates(
        model_class=Supernet,
        model_kwargs={"input_dim": 32, "max_depth": 4, ...},
        input_shape=(1, 32),
        depths=[1, 2, 3, 4],
    )
    # scores: dict[depth → {"flops": int, "synflow": float, "grad_norm": float, "score": float}]
    best_depth = max(scores, key=lambda d: scores[d]["score"])
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


# ---------------------------------------------------------------------------
# FLOP counter (analytical)
# ---------------------------------------------------------------------------

def count_flops_linear(in_features: int, out_features: int, bias: bool = True) -> int:
    """FLOPs for a single Linear layer: 2 * in * out (multiply-add)."""
    return 2 * in_features * out_features + (out_features if bias else 0)


def count_model_flops(model: "nn.Module", input_shape: tuple) -> int:
    """
    Estimate total FLOPs for a forward pass through `model`.

    Uses a hook-based approach: registers forward hooks on Linear and
    Conv layers and accumulates operations.

    Parameters
    ----------
    model       : nn.Module
    input_shape : tuple — e.g. (1, 32) for batch=1, dim=32

    Returns
    -------
    int — estimated number of multiply-add operations
    """
    if not _TORCH_AVAILABLE:
        return 0

    total_flops = [0]
    hooks = []

    def _linear_hook(module, inp, out):
        x = inp[0]
        total_flops[0] += x.shape[-1] * out.shape[-1] * 2

    def _conv_hook(module, inp, out):
        x = inp[0]
        k = module.kernel_size if hasattr(module, "kernel_size") else (1,)
        k_size = k[0] * k[1] if len(k) > 1 else k[0]
        total_flops[0] += (out.numel() * module.in_channels * k_size * 2)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            hooks.append(m.register_forward_hook(_conv_hook))

    with torch.no_grad():
        dummy = torch.zeros(*input_shape)
        try:
            model(dummy)
        except Exception:
            # Some models (multi-head) need different call signature
            try:
                model.forward_multi_head(dummy, active_depth=1)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    return total_flops[0]


# ---------------------------------------------------------------------------
# Zero-cost proxies
# ---------------------------------------------------------------------------

def synflow_score(model: "nn.Module", input_shape: tuple) -> float:
    """
    Synflow proxy (Tanaka et al., 2020): sum of |param * grad| at init.

    Correlates with trainability without any data. Higher = better.
    """
    if not _TORCH_AVAILABLE:
        return 0.0

    model = model.train()
    # All-ones input + all-ones loss → data-free
    inp = torch.ones(*input_shape)
    try:
        out = model(inp)
        if isinstance(out, dict):
            out = list(out.values())[0]
        loss = out.sum()
    except Exception:
        try:
            preds = model.forward_multi_head(inp, active_depth=1)
            out = list(preds.values())[0]
            loss = out.sum()
        except Exception:
            return 0.0

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss.backward()

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += float((p.data.abs() * p.grad.abs()).sum().item())

    # Clear gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    return score


def grad_norm_score(model: "nn.Module", input_shape: tuple) -> float:
    """
    Gradient norm proxy: L2 norm of all gradients at a random input.

    Higher = model is more sensitive / expressive at init.
    """
    if not _TORCH_AVAILABLE:
        return 0.0

    model = model.train()
    inp = torch.randn(*input_shape)
    try:
        out = model(inp)
        if isinstance(out, dict):
            out = list(out.values())[0]
        loss = out.sum()
    except Exception:
        try:
            preds = model.forward_multi_head(inp, active_depth=1)
            out = list(preds.values())[0]
            loss = out.sum()
        except Exception:
            return 0.0

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += float(p.grad.data.norm(2).item() ** 2)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    return math.sqrt(total_norm)


# ---------------------------------------------------------------------------
# Profile depth candidates
# ---------------------------------------------------------------------------

def profile_depth_candidates(
    model_class,
    model_kwargs: dict,
    input_shape: tuple,
    depths: Optional[list] = None,
    flop_weight: float = 0.3,
    synflow_weight: float = 0.5,
    grad_norm_weight: float = 0.2,
) -> dict:
    """
    Profile multiple subnet depths using zero-cost proxies.

    Parameters
    ----------
    model_class     : class — e.g. Supernet
    model_kwargs    : dict  — passed to model_class(**model_kwargs)
    input_shape     : tuple — e.g. (1, 32)
    depths          : list[int] — subnet depths to profile (default 1..max_depth)
    flop_weight     : float — weight for FLOP efficiency in composite score
    synflow_weight  : float — weight for synflow proxy in composite score
    grad_norm_weight: float — weight for grad-norm proxy in composite score

    Returns
    -------
    dict[depth → {"flops": int, "synflow": float, "grad_norm": float, "score": float}]
    """
    if not _TORCH_AVAILABLE:
        logger.warning("torch not available — returning empty NAS profile")
        return {}

    max_depth = model_kwargs.get("max_depth", 4)
    depths = depths or list(range(1, max_depth + 1))

    results = {}
    raw_flops, raw_syn, raw_gn = {}, {}, {}

    for depth in depths:
        try:
            kw = dict(model_kwargs)
            kw["max_depth"] = depth
            model = model_class(**kw)
            model.eval()

            flops = count_model_flops(model, input_shape)
            syn   = synflow_score(model, input_shape)
            gn    = grad_norm_score(model, input_shape)

            raw_flops[depth] = flops
            raw_syn[depth]   = syn
            raw_gn[depth]    = gn

            results[depth] = {"flops": flops, "synflow": syn, "grad_norm": gn, "score": 0.0}
        except Exception as e:
            logger.warning("NAS profiler failed for depth=%d: %s", depth, e)

    if not results:
        return results

    # Normalise and compute composite score
    max_flops  = max(raw_flops.values()) or 1
    max_syn    = max(raw_syn.values())   or 1
    max_gn     = max(raw_gn.values())   or 1

    for depth in results:
        # Lower FLOPs → better efficiency
        flop_eff = 1.0 - (raw_flops[depth] / max_flops)
        syn_norm = raw_syn[depth]   / max_syn
        gn_norm  = raw_gn[depth]    / max_gn

        results[depth]["score"] = (
            flop_weight * flop_eff
            + synflow_weight * syn_norm
            + grad_norm_weight * gn_norm
        )

    logger.info(
        "NAS profiler results: %s",
        {d: round(v["score"], 4) for d, v in results.items()},
    )
    return results
