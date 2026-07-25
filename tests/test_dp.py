"""
tests/test_dp.py
Tests for shared/dp_utils.py — Gaussian DP mechanism + PrivacyAccountant.

Covers:
  - Noise magnitude calibration
  - Gradient clipping (clip_weights)
  - apply_dp end-to-end
  - PrivacyAccountant accumulation and budget-exceeded warning
  - TrainConfig DP integration in run_local_training
  - Trimmed-mean Byzantine defense in aggregate_fedavg
"""
import warnings

import numpy as np
import pytest


# =============================================================================
# A. shared/dp_utils.py — clip_weights
# =============================================================================
class TestClipWeights:
    def _make_weights(self, val: float, shape=(8, 8)) -> dict:
        return {"w": np.full(shape, val, dtype=np.float32)}

    def test_no_clip_when_norm_below_bound(self):
        from shared.dp_utils import clip_weights
        w = {"a": np.ones((4,), dtype=np.float32)}
        # norm = 2.0; max_norm = 5.0 → no clipping
        out = clip_weights(w, max_norm=5.0)
        np.testing.assert_allclose(out["a"], w["a"])

    def test_clips_when_norm_exceeds_bound(self):
        from shared.dp_utils import clip_weights
        w = {"a": np.full((4,), 10.0, dtype=np.float32)}
        out = clip_weights(w, max_norm=1.0)
        flat = np.concatenate([v.flatten() for v in out.values()])
        norm = float(np.linalg.norm(flat))
        assert norm <= 1.0 + 1e-5

    def test_multi_key_global_norm_clipped(self):
        from shared.dp_utils import clip_weights
        w = {
            "a": np.full((4,), 3.0, dtype=np.float32),
            "b": np.full((4,), 4.0, dtype=np.float32),
        }
        # Global L2 norm = sqrt(9*4 + 16*4) = sqrt(100) = 10
        out = clip_weights(w, max_norm=1.0)
        flat = np.concatenate([v.flatten() for v in out.values()])
        norm = float(np.linalg.norm(flat))
        assert norm <= 1.0 + 1e-5

    def test_zero_dict_unchanged(self):
        from shared.dp_utils import clip_weights
        w = {"z": np.zeros((4,), dtype=np.float32)}
        out = clip_weights(w, max_norm=1.0)
        np.testing.assert_array_equal(out["z"], w["z"])

    def test_invalid_max_norm_raises(self):
        from shared.dp_utils import clip_weights
        with pytest.raises(ValueError):
            clip_weights({"w": np.ones((4,))}, max_norm=0.0)

    def test_output_dtype_float32(self):
        from shared.dp_utils import clip_weights
        w = {"w": np.ones((4,), dtype=np.float64)}
        out = clip_weights(w, max_norm=1.0)
        assert out["w"].dtype == np.float32


# =============================================================================
# B. shared/dp_utils.py — apply_dp (Gaussian mechanism)
# =============================================================================
class TestApplyDP:
    def _zeros(self, shape=(4, 4)) -> dict:
        return {"w": np.zeros(shape, dtype=np.float32)}

    def test_output_keys_preserved(self):
        from shared.dp_utils import apply_dp
        w = {"layer_a": np.zeros((8,), np.float32), "layer_b": np.zeros((4, 4), np.float32)}
        out = apply_dp(w, sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert set(out.keys()) == set(w.keys())

    def test_output_shape_preserved(self):
        from shared.dp_utils import apply_dp
        w = {"w": np.zeros((8, 8), np.float32)}
        out = apply_dp(w, sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert out["w"].shape == (8, 8)

    def test_noise_is_nonzero(self):
        from shared.dp_utils import apply_dp
        w = self._zeros()
        rng = np.random.default_rng(0)
        out = apply_dp(w, sensitivity=1.0, epsilon=1.0, delta=1e-5, rng=rng)
        assert np.any(out["w"] != 0.0)

    def test_smaller_epsilon_gives_more_noise(self):
        """Smaller privacy budget → larger sigma → larger noise variance."""
        from shared.dp_utils import apply_dp
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        shape = (100, 100)
        w = {"w": np.zeros(shape, dtype=np.float32)}
        out_loose = apply_dp(dict(w), sensitivity=1.0, epsilon=10.0, delta=1e-5, rng=rng1)
        out_tight  = apply_dp(dict(w), sensitivity=1.0, epsilon=0.1, delta=1e-5, rng=rng2)
        var_loose = float(np.var(out_loose["w"]))
        var_tight  = float(np.var(out_tight["w"]))
        assert var_tight > var_loose

    def test_larger_sensitivity_gives_more_noise(self):
        from shared.dp_utils import apply_dp
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        shape = (100, 100)
        w = {"w": np.zeros(shape, dtype=np.float32)}
        out_low  = apply_dp(dict(w), sensitivity=0.1, epsilon=1.0, delta=1e-5, rng=rng1)
        out_high = apply_dp(dict(w), sensitivity=10.0, epsilon=1.0, delta=1e-5, rng=rng2)
        assert float(np.var(out_high["w"])) > float(np.var(out_low["w"]))

    def test_invalid_epsilon_raises(self):
        from shared.dp_utils import apply_dp
        with pytest.raises(ValueError):
            apply_dp(self._zeros(), sensitivity=1.0, epsilon=0.0, delta=1e-5)

    def test_invalid_delta_raises(self):
        from shared.dp_utils import apply_dp
        with pytest.raises(ValueError):
            apply_dp(self._zeros(), sensitivity=1.0, epsilon=1.0, delta=0.0)

    def test_output_dtype_float32(self):
        from shared.dp_utils import apply_dp
        out = apply_dp(self._zeros(), sensitivity=1.0, epsilon=1.0, delta=1e-5)
        assert out["w"].dtype == np.float32


# =============================================================================
# C. shared/dp_utils.py — PrivacyAccountant
# =============================================================================
class TestPrivacyAccountant:
    def _make_accountant(self, target_epsilon=10.0, target_delta=1e-5,
                         noise_multiplier=5.0, sampling_rate=0.01):
        from shared.dp_utils import PrivacyAccountant
        return PrivacyAccountant(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            noise_multiplier=noise_multiplier,
            sampling_rate=sampling_rate,
        )

    def test_initial_state(self):
        acc = self._make_accountant()
        assert acc.steps == 0
        assert acc.spent_epsilon == 0.0
        assert not acc.budget_exceeded

    def test_step_increments(self):
        acc = self._make_accountant()
        acc.step(5)
        assert acc.steps == 5

    def test_budget_remaining_decreases(self):
        acc = self._make_accountant()
        r0 = acc.budget_remaining
        acc.step(1)
        assert acc.budget_remaining < r0

    def test_budget_exceeded_triggers_warning(self):
        """Very small target_epsilon with many steps → budget exceeded warning."""
        from shared.dp_utils import PrivacyAccountant
        acc = PrivacyAccountant(
            target_epsilon=0.001,
            target_delta=1e-5,
            noise_multiplier=0.1,   # very noisy → high eps per step
            sampling_rate=0.5,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acc.step(1000)
        budget_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)
                           and "budget" in str(w.message).lower()]
        assert len(budget_warnings) >= 1

    def test_budget_exceeded_only_warns_once(self):
        from shared.dp_utils import PrivacyAccountant
        acc = PrivacyAccountant(
            target_epsilon=0.001, target_delta=1e-5,
            noise_multiplier=0.1, sampling_rate=0.5,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            acc.step(1000)
            acc.step(1000)  # second call after budget exceeded — no extra warning
        budget_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)
                           and "budget" in str(w.message).lower()]
        assert len(budget_warnings) == 1

    def test_get_privacy_spent_keys(self):
        acc = self._make_accountant()
        acc.step(10)
        spent = acc.get_privacy_spent()
        for key in ("steps", "spent_epsilon", "target_epsilon", "target_delta",
                    "budget_remaining", "budget_exceeded", "noise_multiplier",
                    "sampling_rate"):
            assert key in spent

    def test_reset_clears_state(self):
        acc = self._make_accountant()
        acc.step(100)
        assert acc.steps == 100
        acc.reset()
        assert acc.steps == 0
        assert acc.spent_epsilon == 0.0
        assert not acc.budget_exceeded

    def test_repr_does_not_crash(self):
        acc = self._make_accountant()
        assert "PrivacyAccountant" in repr(acc)

    def test_invalid_target_epsilon_raises(self):
        from shared.dp_utils import PrivacyAccountant
        with pytest.raises(ValueError):
            PrivacyAccountant(target_epsilon=0.0)

    def test_invalid_sampling_rate_raises(self):
        from shared.dp_utils import PrivacyAccountant
        with pytest.raises(ValueError):
            PrivacyAccountant(sampling_rate=0.0)


# =============================================================================
# D. TrainConfig DP fields + run_local_training integration
# =============================================================================
class TestTrainLoopDP:
    """
    Tests that DP-enabled run_local_training returns privacy_spent in metrics.
    """
    _INPUT_DIM = 32
    _MAX_DEPTH = 2
    _HIDDEN_DIM = 16

    def _make_loaders(self, n=64, batch_size=16):
        import torch
        from torch.utils.data import TensorDataset, DataLoader, Subset
        torch.manual_seed(1)
        X     = torch.randn(n, self._INPUT_DIM)
        y_reg = torch.randn(n)
        y_tox = torch.randint(0, 4, (n,))
        y_bin = torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)
        ds = TensorDataset(X, y_reg, y_tox, y_bin)
        n_train = int(n * 0.75)
        train = DataLoader(Subset(ds, list(range(n_train))), batch_size=batch_size, drop_last=True)
        val   = DataLoader(Subset(ds, list(range(n_train, n))), batch_size=batch_size)
        return train, val

    def _make_model(self):
        from supernet import Supernet
        return Supernet(input_dim=self._INPUT_DIM, max_depth=self._MAX_DEPTH,
                        hidden_dim=self._HIDDEN_DIM, num_toxicity_classes=4)

    def test_dp_disabled_by_default_no_privacy_key(self):
        from train_loop import run_local_training, TrainConfig
        cfg = TrainConfig(epochs=1, active_depth=1, dp_epsilon=0.0)
        result = run_local_training(self._make_model(), self._make_loaders(), cfg)
        # DP disabled → no privacy_spent key
        assert "privacy_spent" not in result["metrics"]

    def test_dp_enabled_returns_privacy_spent(self):
        from train_loop import run_local_training, TrainConfig
        cfg = TrainConfig(epochs=1, active_depth=1, dp_epsilon=2.0, dp_delta=1e-5,
                          dp_max_grad_norm=1.0)
        result = run_local_training(self._make_model(), self._make_loaders(), cfg)
        assert "privacy_spent" in result["metrics"]
        spent = result["metrics"]["privacy_spent"]
        for key in ("steps", "spent_epsilon", "target_epsilon", "budget_exceeded"):
            assert key in spent

    def test_dp_weights_have_noise(self):
        """Weights with DP enabled should differ from weights without."""
        import torch
        from train_loop import run_local_training, TrainConfig
        from supernet import Supernet
        torch.manual_seed(99)
        cfg_no_dp = TrainConfig(epochs=1, active_depth=1, dp_epsilon=0.0)
        cfg_dp    = TrainConfig(epochs=1, active_depth=1, dp_epsilon=0.1,
                                dp_delta=1e-5, dp_max_grad_norm=0.1)
        m1 = self._make_model()
        m2 = self._make_model()
        # Give both models identical initial weights
        m2.load_state_dict(m1.state_dict())
        loaders = self._make_loaders()
        r1 = run_local_training(m1, loaders, cfg_no_dp)
        r2 = run_local_training(m2, loaders, cfg_dp)
        # Weights must differ due to noise
        key = list(r1["weights"].keys())[0]
        assert not np.allclose(r1["weights"][key], r2["weights"][key])

    def test_dp_config_fields_accessible(self):
        from train_loop import TrainConfig
        cfg = TrainConfig(dp_epsilon=1.0, dp_delta=1e-6, dp_max_grad_norm=0.5)
        assert cfg.dp_epsilon == 1.0
        assert cfg.dp_delta == 1e-6
        assert cfg.dp_max_grad_norm == 0.5


# =============================================================================
# E. Trimmed-mean Byzantine defense in aggregate_fedavg
# =============================================================================
class TestTrimmedMean:
    def _make_update(self, val, shape=(8,)):
        return {"w": np.full(shape, val, dtype=np.float32)}

    def test_trimming_ratio_zero_matches_standard_fedavg(self):
        from aggregation import aggregate_fedavg
        updates = [self._make_update(0.0), self._make_update(2.0),
                   self._make_update(1.0), self._make_update(1.0)]
        counts = [100, 100, 100, 100]
        std_result = aggregate_fedavg(updates, counts, trimming_ratio=0.0)
        np.testing.assert_allclose(std_result["w"], 1.0, atol=1e-4)

    def test_trimmed_mean_removes_outlier(self):
        """With trimming, an extreme outlier client should not dominate."""
        from aggregation import aggregate_fedavg
        # 6 clients: 5 honest (val=1.0) + 1 Byzantine (val=100.0)
        honest = [self._make_update(1.0) for _ in range(5)]
        byzantine = [self._make_update(100.0)]
        updates = honest + byzantine
        counts = [100] * 6
        # Without trimming, mean ≈ 17.5
        no_trim = aggregate_fedavg(updates, counts, trimming_ratio=0.0)
        # With trimming (16% each side → trim 1), Byzantine removed
        trimmed = aggregate_fedavg(updates, counts, trimming_ratio=0.16)
        # Trimmed result should be much closer to 1.0
        assert abs(float(trimmed["w"][0]) - 1.0) < abs(float(no_trim["w"][0]) - 1.0)

    def test_invalid_trimming_ratio_raises(self):
        from aggregation import aggregate_fedavg
        updates = [self._make_update(1.0)] * 4
        with pytest.raises(ValueError):
            aggregate_fedavg(updates, [100] * 4, trimming_ratio=0.5)
        with pytest.raises(ValueError):
            aggregate_fedavg(updates, [100] * 4, trimming_ratio=-0.1)

    def test_trimming_with_too_few_clients_fallback(self):
        """Fewer than 4 clients → falls back to weighted average (no crash)."""
        from aggregation import aggregate_fedavg
        updates = [self._make_update(1.0), self._make_update(3.0)]
        result = aggregate_fedavg(updates, [100, 100], trimming_ratio=0.2)
        np.testing.assert_allclose(result["w"], 2.0, atol=1e-4)

    def test_trimming_bounds_result_between_min_max(self):
        from aggregation import aggregate_fedavg
        vals = [0.0, 1.0, 2.0, 3.0, 100.0, 200.0]
        updates = [self._make_update(v) for v in vals]
        result = aggregate_fedavg(updates, [100] * 6, trimming_ratio=0.2)
        # Extremes removed: result should be in range [0, 100]
        assert float(result["w"][0]) < 100.0
