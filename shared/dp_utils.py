"""
shared/dp_utils.py
Differential Privacy utilities for the FL Platform.

Implements:
  - Gaussian mechanism for gradient-level DP-SGD noise addition
  - PrivacyAccountant for tracking cumulative (epsilon, delta) privacy budget

Usage (client side, inside run_local_training):
    from shared.dp_utils import apply_dp, PrivacyAccountant

    accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)
    clipped_grads = clip_gradients(grads, max_norm=clip_norm)
    noisy_grads   = apply_dp(clipped_grads, sensitivity=clip_norm,
                              epsilon=epsilon_per_step,
                              delta=delta_per_step)
    accountant.step()

Theory
------
Gaussian mechanism: add noise ~ N(0, sigma^2) where
    sigma = sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon

Privacy amplification by sampling: when using mini-batches of fraction q,
the effective epsilon scales as O(q * epsilon) — tracked by moments accountant.

References
----------
  Abadi et al. "Deep Learning with Differential Privacy." CCS 2016.
  Mironov, I. "Rényi Differential Privacy." CSF 2017.
"""
from __future__ import annotations

import math
import warnings
import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Gaussian noise calibration
# ---------------------------------------------------------------------------

def _gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """
    Compute the Gaussian mechanism noise multiplier sigma.

    sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon

    Parameters
    ----------
    sensitivity : float — L2-norm bound on the function (= clip_norm for gradients)
    epsilon     : float — per-step privacy budget
    delta       : float — per-step failure probability

    Returns
    -------
    float — standard deviation of Gaussian noise to add
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")

    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon
    return sigma


def apply_dp(
    weights: Dict[str, np.ndarray],
    sensitivity: float,
    epsilon: float,
    delta: float,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply the Gaussian mechanism to a weight-update dict.

    Each value in `weights` is independently noised. The noise is calibrated
    to (epsilon, delta)-DP under the assumption that gradients have already
    been clipped to L2-norm `sensitivity`.

    Parameters
    ----------
    weights     : dict[str, np.ndarray] — weight-update dict from get_subnet_weights()
    sensitivity : float — L2 clip norm (must match clip_norm used in training)
    epsilon     : float — per-step privacy budget (e.g. 1.0 / num_steps)
    delta       : float — per-step failure probability (e.g. 1e-5 / num_steps)
    rng         : optional numpy Generator for reproducible noise (tests only)

    Returns
    -------
    dict[str, np.ndarray] — noised weight-update dict (same keys, same shapes)
    """
    sigma = _gaussian_sigma(sensitivity, epsilon, delta)
    if rng is None:
        rng = np.random.default_rng()

    noised = {}
    for key, arr in weights.items():
        arr = np.array(arr, dtype=np.float32)
        noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
        noised[key] = arr + noise

    return noised


def clip_weights(
    weights: Dict[str, np.ndarray],
    max_norm: float,
) -> Dict[str, np.ndarray]:
    """
    Clip the entire weight-update dict so its global L2 norm <= max_norm.

    This is the sensitivity-bounding step that must precede apply_dp().

    Parameters
    ----------
    weights  : dict[str, np.ndarray]
    max_norm : float — L2 norm bound (= clip_norm in TrainConfig)

    Returns
    -------
    dict[str, np.ndarray] — clipped weight dict (same structure)
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be positive, got {max_norm}")

    # Compute global L2 norm across all parameter tensors
    flat = np.concatenate([arr.flatten() for arr in weights.values()])
    global_norm = float(np.linalg.norm(flat))

    if global_norm <= max_norm:
        return {k: np.array(v, dtype=np.float32) for k, v in weights.items()}

    scale = max_norm / (global_norm + 1e-12)
    return {k: (np.array(v, dtype=np.float32) * scale) for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Privacy Accountant (Moments / Rényi DP)
# ---------------------------------------------------------------------------

class PrivacyAccountant:
    """
    Tracks cumulative privacy budget consumption across training steps.

    Uses the simplified strong composition theorem as a conservative upper
    bound. For tighter accounting in production, integrate with google/
    dp-accounting or autodp. This implementation is intentionally
    self-contained and dependency-free.

    The accountant signals a warning (not an error) when the budget is
    exceeded so that training can continue for research purposes — operators
    should decide whether to abort at that point.

    Parameters
    ----------
    target_epsilon : float — total privacy budget across all rounds/steps
    target_delta   : float — global failure probability
    noise_multiplier: float — sigma / sensitivity ratio (for Rényi accounting)
    sampling_rate  : float — batch size / dataset size (privacy amplification)
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        sampling_rate: float = 0.01,
    ) -> None:
        if target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if not 0 < target_delta < 1:
            raise ValueError("target_delta must be in (0, 1)")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if not 0 < sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0, 1]")

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.sampling_rate = sampling_rate

        self._steps: int = 0
        self._spent_epsilon: float = 0.0
        self._budget_exceeded: bool = False

    # ------------------------------------------------------------------
    # Budget tracking
    # ------------------------------------------------------------------

    def step(self, num_steps: int = 1) -> None:
        """
        Record that `num_steps` DP-SGD steps have been taken.
        Updates the accumulated privacy budget estimate.
        Emits a RuntimeWarning if the target budget is exceeded.
        """
        self._steps += num_steps
        self._spent_epsilon = self._compute_epsilon()

        if self._spent_epsilon > self.target_epsilon and not self._budget_exceeded:
            self._budget_exceeded = True
            warnings.warn(
                f"PrivacyAccountant: privacy budget exceeded! "
                f"spent_epsilon={self._spent_epsilon:.4f} > "
                f"target_epsilon={self.target_epsilon:.4f} "
                f"after {self._steps} steps.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _compute_epsilon(self) -> float:
        """
        Conservative epsilon estimate using strong composition theorem.

        eps_total ≤ sqrt(2 * k * ln(1/delta')) * eps_step + k * eps_step * (e^eps_step - 1)

        where eps_step = per-step budget and k = number of steps.
        We use the simplified first-order approximation for small eps_step:
            eps_total ≈ eps_step * sqrt(2 * k * ln(1/delta))
        """
        if self._steps == 0:
            return 0.0

        # Per-step epsilon from noise multiplier (Gaussian mechanism inversion)
        # sigma = noise_multiplier, sensitivity = 1 (normalised), delta = target_delta
        # epsilon_step ≈ sensitivity / sigma * sqrt(2 * ln(1.25/delta))
        eps_step = (
            math.sqrt(2.0 * math.log(1.25 / self.target_delta))
            / self.noise_multiplier
        )

        # Privacy amplification by subsampling: eps_step → q * eps_step (approx)
        eps_step_amplified = self.sampling_rate * eps_step

        # Strong composition: eps ≈ eps_step * sqrt(2k * ln(1/delta))
        k = self._steps
        eps_total = eps_step_amplified * math.sqrt(
            2.0 * k * math.log(1.0 / self.target_delta)
        )
        return float(eps_total)

    @property
    def spent_epsilon(self) -> float:
        """Current cumulative privacy expenditure estimate."""
        return self._spent_epsilon

    @property
    def steps(self) -> int:
        """Total number of DP-SGD steps recorded."""
        return self._steps

    @property
    def budget_remaining(self) -> float:
        """Remaining epsilon budget (may be negative if exceeded)."""
        return self.target_epsilon - self._spent_epsilon

    @property
    def budget_exceeded(self) -> bool:
        """True if the privacy budget has been exceeded."""
        return self._budget_exceeded

    def get_privacy_spent(self) -> dict:
        """
        Return a serialisable dict summarising current privacy expenditure.
        Suitable for including in round metrics.
        """
        return {
            "steps":            self._steps,
            "spent_epsilon":    round(self._spent_epsilon, 6),
            "target_epsilon":   self.target_epsilon,
            "target_delta":     self.target_delta,
            "budget_remaining": round(self.budget_remaining, 6),
            "budget_exceeded":  self._budget_exceeded,
            "noise_multiplier": self.noise_multiplier,
            "sampling_rate":    self.sampling_rate,
        }

    def reset(self) -> None:
        """Reset the accountant for a new training session."""
        self._steps = 0
        self._spent_epsilon = 0.0
        self._budget_exceeded = False

    def __repr__(self) -> str:
        return (
            f"PrivacyAccountant("
            f"steps={self._steps}, "
            f"spent_eps={self._spent_epsilon:.4f}, "
            f"target_eps={self.target_epsilon}, "
            f"target_delta={self.target_delta})"
        )
