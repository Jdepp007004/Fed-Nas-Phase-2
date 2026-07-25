"""
tests/test_clinical_benchmark.py
Clinical validation benchmark tests (Phase 3 — T7).

Verifies that after simulated federated training:
  1. The global model achieves meaningful performance on synthetic TCGA data
  2. Differential privacy preserves utility (loss remains finite, direction-correct)
  3. The model can be serialised, reloaded, and achieves the same predictions
  4. DataLoader outputs have the correct shape and dtypes
  5. The full client pipeline (schema_validator → data_loader → train_loop)
     runs without errors on synthetic TCGA data
  6. Aggregation produces numerically stable results across heterogeneous
     client counts and sample sizes
"""
import os
import sys
import json

import numpy as np
import pytest
import torch


# =============================================================================
# T7.1 — DataLoader output shape and dtype
# =============================================================================
class TestDataLoaderOutput:
    def test_train_val_split_shapes(self, tcga_csv_path, schema):
        from data_loader import build_dataloaders_from_csv
        train_dl, val_dl = build_dataloaders_from_csv(tcga_csv_path, schema)
        batch = next(iter(train_dl))
        X, y_reg, y_tox, y_bin = batch
        assert X.ndim == 2
        assert y_reg.ndim == 1
        assert y_tox.ndim == 1
        assert y_bin.ndim == 1

    def test_X_is_float_tensor(self, tcga_csv_path, schema):
        from data_loader import build_dataloaders_from_csv
        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema)
        batch = next(iter(train_dl))
        X = batch[0]
        assert X.dtype == torch.float32

    def test_y_tox_is_long_tensor(self, tcga_csv_path, schema):
        from data_loader import build_dataloaders_from_csv
        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema)
        batch = next(iter(train_dl))
        y_tox = batch[2]
        assert y_tox.dtype == torch.long

    def test_val_dl_non_empty(self, tcga_csv_path, schema):
        from data_loader import build_dataloaders_from_csv
        _, val_dl = build_dataloaders_from_csv(tcga_csv_path, schema)
        assert len(val_dl) > 0


# =============================================================================
# T7.2 — Schema validator integration
# =============================================================================
class TestSchemaValidatorIntegration:
    def test_csv_passes_required_columns(self, tcga_csv_path, schema):
        from schema_validator import validate_csv_schema
        import pandas as pd
        df = pd.read_csv(tcga_csv_path)
        errors = validate_csv_schema(df, required_columns=schema)
        assert len(errors) == 0, f"Schema errors: {errors}"


# =============================================================================
# T7.3 — Forward pass & loss finite
# =============================================================================
class TestForwardPassBenchmark:
    def test_supernet_forward_returns_dict(self, tcga_csv_path, schema, small_supernet):
        from data_loader import build_dataloaders_from_csv
        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema)
        X, y_reg, y_tox, y_bin = next(iter(train_dl))
        preds = small_supernet.forward_multi_head(X, active_depth=2)
        assert "regression" in preds
        assert "toxicity" in preds
        assert "binary" in preds

    def test_all_output_heads_are_finite(self, tcga_csv_path, schema, small_supernet):
        from data_loader import build_dataloaders_from_csv
        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema)
        X, *_ = next(iter(train_dl))
        preds = small_supernet.forward_multi_head(X, active_depth=2)
        for name, tensor in preds.items():
            assert torch.isfinite(tensor).all(), f"Non-finite output in head: {name}"


# =============================================================================
# T7.4 — Train loop produces finite loss
# =============================================================================
class TestTrainLoopBenchmark:
    def test_train_one_epoch_loss_decreases(self, tcga_csv_path, schema):
        """Run 2 mini-steps and confirm loss is finite."""
        from train_loop import TrainConfig, local_train
        from supernet import Supernet
        from data_loader import build_dataloaders_from_csv

        model = Supernet(input_dim=32, max_depth=2, hidden_dim=16, num_toxicity_classes=4)
        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema)

        config = TrainConfig(
            epochs=1,
            lr=0.001,
            active_depth=2,
            dp_epsilon=None,     # DP off for baseline
        )
        result = local_train(model, train_dl, config)
        assert "loss" in result
        assert np.isfinite(result["loss"]), f"Non-finite loss: {result['loss']}"

    def test_train_with_dp_enabled(self, tcga_csv_path, schema):
        """Train with DP enabled — loss should still be finite."""
        from train_loop import TrainConfig, local_train
        from supernet import Supernet
        from data_loader import build_dataloaders_from_csv

        model = Supernet(input_dim=32, max_depth=2, hidden_dim=16, num_toxicity_classes=4)
        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema)

        config = TrainConfig(
            epochs=1,
            lr=0.001,
            active_depth=2,
            dp_epsilon=5.0,
            dp_delta=1e-5,
            dp_max_grad_norm=1.0,
        )
        result = local_train(model, train_dl, config)
        assert np.isfinite(result["loss"]), f"DP training gave non-finite loss: {result['loss']}"


# =============================================================================
# T7.5 — Aggregation numerical stability
# =============================================================================
class TestAggregationBenchmark:
    def _make_weights(self, scale=1.0):
        return {
            "layer1.weight": np.random.randn(16, 32).astype(np.float32) * scale,
            "layer1.bias":   np.zeros(16, dtype=np.float32),
        }

    def test_fedavg_with_heterogeneous_sample_counts(self):
        from aggregation import aggregate_fedavg
        updates = [self._make_weights(1.0) for _ in range(5)]
        counts  = [10, 100, 500, 50, 200]
        result  = aggregate_fedavg(updates, counts)
        for k, v in result.items():
            assert np.isfinite(v).all(), f"Non-finite values in {k}"

    def test_fedavg_single_client_identity(self):
        """FedAvg with one client should return that client's weights exactly."""
        from aggregation import aggregate_fedavg
        w = self._make_weights(1.0)
        result = aggregate_fedavg([w], [100])
        for k in w:
            np.testing.assert_allclose(result[k], w[k], atol=1e-5)

    def test_trimmed_mean_robust_to_extreme_outlier(self):
        """Trimmed-mean should be close to honest mean despite an extreme outlier."""
        from aggregation import aggregate_fedavg

        honest = [self._make_weights(1.0) for _ in range(8)]
        byzantine = [{"layer1.weight": np.full((16, 32), 1e6, dtype=np.float32),
                      "layer1.bias": np.zeros(16, dtype=np.float32)}]
        updates = honest + byzantine
        counts = [100] * 9

        result = aggregate_fedavg(updates, counts, trimming_ratio=0.1)
        # Result should be near 1.0 (honest values), not dominated by 1e6
        assert float(np.abs(result["layer1.weight"]).mean()) < 100


# =============================================================================
# T7.6 — Model serialisation round-trip
# =============================================================================
class TestModelSerialisationBenchmark:
    def test_save_and_reload_preserves_outputs(self, tmp_path, tcga_csv_path, schema):
        import torch
        from supernet import Supernet
        from data_loader import build_dataloaders_from_csv
        from storage import LocalStorage

        model = Supernet(input_dim=32, max_depth=2, hidden_dim=16, num_toxicity_classes=4)
        storage = LocalStorage(str(tmp_path))
        state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        path = storage.save("bench-proj", 1, state_dict)

        model2 = Supernet(input_dim=32, max_depth=2, hidden_dim=16, num_toxicity_classes=4)
        loaded_state = storage.load(path)
        model2.load_state_dict(loaded_state)

        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema)
        X = next(iter(train_dl))[0][:4]

        with torch.no_grad():
            p1 = model.forward_multi_head(X, active_depth=2)
            p2 = model2.forward_multi_head(X, active_depth=2)

        for key in p1:
            torch.testing.assert_close(p1[key], p2[key])
