"""
tests/test_unit.py
End-to-end unit tests for the FL Platform.
Run with: pytest tests/ -v --cov=. --cov-report=term-missing
"""

import os
import sys  # noqa: F401
import json  # noqa: F401
import copy  # noqa: F401
import tempfile  # noqa: F401
import threading
import importlib  # noqa: F401

import numpy as np
import pytest
import torch

# ─── Path helpers (conftest already adds these, but be explicit) ─────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# 1. shared/model_schema.py
# =============================================================================
class TestModelSchema:
    def test_constants_sane(self):
        from shared.model_schema import (
            INPUT_DIM, MAX_DEPTH, HIDDEN_DIM, NUM_TOXICITY_CLASSES,
            DEFAULT_FEDPROX_MU, DEFAULT_MOMENTUM_BETA,
        )
        assert INPUT_DIM > 0
        assert 2 <= MAX_DEPTH <= 12
        assert HIDDEN_DIM > 0
        assert NUM_TOXICITY_CLASSES >= 2
        assert 0 < DEFAULT_FEDPROX_MU < 1
        assert 0 < DEFAULT_MOMENTUM_BETA < 1

    def test_required_columns_non_empty(self):
        from shared.model_schema import REQUIRED_COLUMNS, TARGET_COLUMNS
        assert len(REQUIRED_COLUMNS) > 5
        assert "regression" in TARGET_COLUMNS
        assert "toxicity" in TARGET_COLUMNS
        assert "binary" in TARGET_COLUMNS

    def test_server_schema_structure(self):
        from shared.model_schema import SERVER_SCHEMA
        for key in ("required_columns", "target_columns", "feature_ranges",
                    "categorical_values", "min_samples", "schema_version"):
            assert key in SERVER_SCHEMA, f"Missing key: {key}"

    def test_model_config_structure(self):
        from shared.model_schema import MODEL_CONFIG
        for key in ("input_dim", "max_depth", "hidden_dim", "num_toxicity_classes"):
            assert key in MODEL_CONFIG


# =============================================================================
# 2. shared/encryption.py
# =============================================================================
class TestEncryption:
    def _sample_weights(self):
        return {
            "backbone.0.0.weight": np.random.randn(16, 32).astype(np.float32),
            "backbone.0.0.bias":   np.random.randn(16).astype(np.float32),
            "head_regression.weight": np.random.randn(1, 16).astype(np.float32),
        }

    def test_roundtrip(self):
        from shared.encryption import encrypt_weights, decrypt_weights
        original = self._sample_weights()
        enc = encrypt_weights(original)
        dec = decrypt_weights(enc)
        for k in original:
            np.testing.assert_allclose(dec[k], original[k], rtol=1e-5)

    def test_encrypted_payload_has_required_keys(self):
        from shared.encryption import encrypt_weights
        enc = encrypt_weights({"w": np.zeros((2, 2), dtype=np.float32)})
        assert "ciphertext" in enc
        assert "nonce" in enc

    def test_generate_key_b64_length(self):
        from shared.encryption import generate_key_b64
        import base64
        key = generate_key_b64()
        raw = base64.b64decode(key)
        assert len(raw) == 32

    def test_tampered_ciphertext_raises(self):
        from shared.encryption import encrypt_weights, decrypt_weights
        enc = encrypt_weights({"w": np.zeros((3,), dtype=np.float32)})
        enc["ciphertext"] = enc["ciphertext"][:-4] + "XXXX"  # corrupt
        with pytest.raises(Exception):
            decrypt_weights(enc)


# =============================================================================
# 3. client/supernet.py  (M1)
# =============================================================================
class TestSupernet:
    def test_forward_shapes(self, small_supernet):
        net = small_supernet
        x = torch.randn(8, 32)
        for depth in range(1, net.max_depth + 1):
            out = net.forward_multi_head(x, depth)
            assert out["regression"].shape == (8, 1)
            assert out["toxicity"].shape == (8, net.num_toxicity_classes)
            assert out["binary"].shape == (8, 1)

    def test_invalid_depth_raises(self, small_supernet):
        x = torch.randn(4, 32)
        with pytest.raises(ValueError):
            small_supernet.forward_multi_head(x, 0)
        with pytest.raises(ValueError):
            small_supernet.forward_multi_head(x, small_supernet.max_depth + 1)

    def test_get_subnet_weights(self, small_supernet):
        from supernet import get_subnet_weights
        weights = get_subnet_weights(small_supernet, active_depth=2)
        assert isinstance(weights, dict)
        # Should include backbone layers 0 and 1, plus all heads
        assert any("backbone.0" in k for k in weights)
        assert any("backbone.1" in k for k in weights)
        assert not any("backbone.2" in k for k in weights)  # depth=2, so layer 2 excluded
        for k, v in weights.items():
            assert isinstance(v, np.ndarray)

    def test_load_global_weights_roundtrip(self, small_supernet):
        from supernet import get_subnet_weights, load_global_weights, Supernet
        net1 = small_supernet
        net2 = Supernet(input_dim=32, max_depth=3, hidden_dim=16, num_toxicity_classes=4)
        # Make net1 and net2 have different weights
        with torch.no_grad():
            for p in net1.parameters():
                p.fill_(1.0)
        weights = get_subnet_weights(net1, active_depth=2)
        load_global_weights(net2, weights, strict=False)
        # Loaded backbone layers should now match
        for k, v in weights.items():
            parts = k.split(".")
            param = net2
            for part in parts:
                param = getattr(param, part, None) if not part.isdigit() else param[int(part)]
            if param is not None and hasattr(param, 'data'):
                np.testing.assert_allclose(
                    param.detach().cpu().numpy(), v, rtol=1e-4
                )


# =============================================================================
# 4. client/supernet.py — compute_joint_loss
# =============================================================================
class TestJointLoss:
    def test_loss_is_finite(self, small_supernet):
        from supernet import compute_joint_loss
        net = small_supernet
        x = torch.randn(8, 32)
        preds = net.forward_multi_head(x, 2)
        targets = {
            "regression": torch.randn(8),
            "toxicity":   torch.randint(0, 4, (8,)),
            "binary":     torch.randint(0, 2, (8,)).float(),
        }
        weights = {"regression": 1.0, "toxicity": 0.8, "binary": 0.6}
        total, breakdown = compute_joint_loss(preds, targets, weights)
        assert torch.isfinite(total)
        assert "loss_reg" in breakdown
        assert "loss_tox" in breakdown
        assert "loss_bin" in breakdown

    def test_loss_requires_grad(self, small_supernet):
        from supernet import compute_joint_loss
        x = torch.randn(8, 32)
        preds = small_supernet.forward_multi_head(x, 1)
        targets = {
            "regression": torch.randn(8),
            "toxicity":   torch.randint(0, 4, (8,)),
            "binary":     torch.zeros(8).float(),
        }
        loss, _ = compute_joint_loss(preds, targets, {"regression": 1.0, "toxicity": 1.0, "binary": 1.0})
        assert loss.requires_grad


# =============================================================================
# 5. client/train_loop.py — apply_fedprox_penalty
# =============================================================================
class TestFedProx:
    def test_penalty_zero_when_models_equal(self, small_supernet):
        from train_loop import apply_fedprox_penalty
        net = small_supernet
        local_params = list(net.parameters())
        global_params = [p.clone().detach() for p in local_params]
        penalty = apply_fedprox_penalty(iter(local_params), iter(global_params), mu=0.01)
        assert penalty.item() < 1e-6

    def test_penalty_positive_when_models_differ(self, small_supernet):
        from train_loop import apply_fedprox_penalty, Supernet
        local = Supernet(input_dim=32, max_depth=3, hidden_dim=16, num_toxicity_classes=4)
        with torch.no_grad():
            for p in local.parameters():
                p.fill_(5.0)
        global_params = [torch.zeros_like(p) for p in local.parameters()]
        penalty = apply_fedprox_penalty(local.parameters(), iter(global_params), mu=0.01)
        assert penalty.item() > 0

    def test_penalty_scales_with_mu(self, small_supernet):
        from train_loop import apply_fedprox_penalty, Supernet
        local = Supernet(input_dim=32, max_depth=3, hidden_dim=16, num_toxicity_classes=4)
        with torch.no_grad():
            for p in local.parameters():
                p.fill_(2.0)
        global_params_1 = [torch.zeros_like(p) for p in local.parameters()]
        global_params_2 = [torch.zeros_like(p) for p in local.parameters()]
        p1 = apply_fedprox_penalty(local.parameters(), iter(global_params_1), mu=0.01)
        p2 = apply_fedprox_penalty(local.parameters(), iter(global_params_2), mu=0.10)
        assert abs(p2.item() / p1.item() - 10) < 0.5  # roughly 10x


# =============================================================================
# 6. server/aggregation.py — aggregate_fedavg
# =============================================================================
class TestFedAvg:
    def _make_update(self, val, shape=(4, 4)):
        return {"w": np.full(shape, val, dtype=np.float32)}

    def test_single_client_returns_same_weights(self):
        from aggregation import aggregate_fedavg
        upd = self._make_update(3.0)
        result = aggregate_fedavg([upd], [100])
        np.testing.assert_allclose(result["w"], 3.0)

    def test_equal_clients_equal_average(self):
        from aggregation import aggregate_fedavg
        updates = [self._make_update(1.0), self._make_update(3.0)]
        result = aggregate_fedavg(updates, [50, 50])
        np.testing.assert_allclose(result["w"], 2.0, atol=1e-5)

    def test_weighted_average(self):
        from aggregation import aggregate_fedavg
        updates = [self._make_update(0.0), self._make_update(10.0)]
        # 100 samples at 0.0 and 100 samples at 10.0 with 3:1 ratio → 2.5
        result = aggregate_fedavg(updates, [300, 100])
        np.testing.assert_allclose(result["w"], 2.5, atol=1e-4)

    def test_empty_raises(self):
        from aggregation import aggregate_fedavg, EmptyRoundError
        with pytest.raises(EmptyRoundError):
            aggregate_fedavg([], [])

    def test_missing_keys_handled(self):
        """Clients with different subnets (missing keys) should still aggregate."""
        from aggregation import aggregate_fedavg
        u1 = {"backbone.0.weight": np.ones((4,), dtype=np.float32),
               "head.weight":      np.ones((4,), dtype=np.float32)}  # noqa: E127
        u2 = {"head.weight":       np.full((4,), 3.0, dtype=np.float32)}  # shallower client
        result = aggregate_fedavg([u1, u2], [100, 100])
        # head.weight should be average of 1.0 and 3.0 = 2.0
        np.testing.assert_allclose(result["head.weight"], 2.0, atol=1e-4)
        # backbone.0.weight contributed by only u1
        np.testing.assert_allclose(result["backbone.0.weight"], 1.0, atol=1e-4)


# =============================================================================
# 7. server/aggregation.py — update_with_momentum
# =============================================================================
class TestMomentum:
    def test_converges_toward_aggregate(self):
        from aggregation import update_with_momentum
        current = {"w": np.zeros((4,), dtype=np.float32)}
        agg = {"w": np.full((4,), 10.0, dtype=np.float32)}
        velocity = {}
        new_g, vel = update_with_momentum(current, agg, momentum=0.9, velocity=velocity)
        # New global should be closer to 10 than 0
        assert new_g["w"].mean() > current["w"].mean()

    def test_velocity_persists_across_rounds(self):
        from aggregation import update_with_momentum
        current = {"w": np.zeros((4,), dtype=np.float32)}
        agg = {"w": np.full((4,), 10.0, dtype=np.float32)}
        velocity = {}
        g1, vel1 = update_with_momentum(current, agg, 0.9, velocity)
        g2, vel2 = update_with_momentum(g1,      agg, 0.9, vel1)
        # Second update moves further toward target
        assert g2["w"].mean() > g1["w"].mean()


# =============================================================================
# 8. server/nas_controller.py
# =============================================================================
class TestNASController:
    def test_gpu_high_ram_gets_high_depth(self):
        from nas_controller import recommend_subnet_depth
        d = recommend_subnet_depth("c1", {"ram_gb": 64, "cpu_cores": 16, "gpu_available": True, "local_data_size": 5000})  # noqa: E501
        assert d >= 5

    def test_low_ram_no_gpu_gets_low_depth(self):
        from nas_controller import recommend_subnet_depth
        d = recommend_subnet_depth("c2", {"ram_gb": 2, "cpu_cores": 2, "gpu_available": False, "local_data_size": 100})
        assert d <= 3

    def test_depth_within_bounds(self):
        from nas_controller import recommend_subnet_depth
        from shared.model_schema import MAX_DEPTH
        for ram in [2, 4, 8, 16, 32]:
            d = recommend_subnet_depth("cx", {"ram_gb": ram, "cpu_cores": 4, "gpu_available": False, "local_data_size": 1000})  # noqa: E501
            assert 2 <= d <= MAX_DEPTH

    def test_different_clients_cached_independently(self):
        from nas_controller import recommend_subnet_depth, _depth_cache
        recommend_subnet_depth("ca", {"ram_gb": 32, "gpu_available": True, "cpu_cores": 8, "local_data_size": 2000})
        recommend_subnet_depth("cb", {"ram_gb":  2, "gpu_available": False, "cpu_cores": 1, "local_data_size": 50})
        assert _depth_cache["ca"] != _depth_cache["cb"]


# =============================================================================
# 9. server/db_handler.py
# =============================================================================
class TestDBHandler:
    def test_read_empty_db_returns_structure(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        assert "users" in db
        assert "projects" in db
        assert "rounds_history" in db

    def test_write_and_read_roundtrip(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        data = {"users": [{"id": "u1"}], "projects": [], "rounds_history": []}
        db_handler.write_db(data)
        recovered = db_handler.read_db()
        assert recovered["users"][0]["id"] == "u1"

    def test_get_project_found(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        proj = {"proj_id": "p123", "name": "Test Project", "connected_clients": []}
        db = db_handler.read_db()
        db["projects"].append(proj)
        db_handler.write_db(db)
        found = db_handler.get_project("p123")
        assert found is not None
        assert found["name"] == "Test Project"

    def test_get_project_not_found_returns_none(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        result = db_handler.get_project("nonexistent_id")
        assert result is None

    def test_update_project_merges(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        proj = {"proj_id": "pA", "current_round": 0, "connected_clients": []}
        db = db_handler.read_db()
        db["projects"].append(proj)
        db_handler.write_db(db)
        db_handler.update_project("pA", {"current_round": 5})
        updated = db_handler.get_project("pA")
        assert updated["current_round"] == 5

    def test_update_missing_project_raises(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        with pytest.raises(KeyError):
            db_handler.update_project("no-such-project", {"x": 1})

    def test_thread_safety(self, tmp_db_path, monkeypatch):
        """Concurrent writes should not corrupt the database."""
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        errors = []

        def write_something(i):
            try:
                db = db_handler.read_db()
                db.setdefault("extra", []).append(i)
                db_handler.write_db(db)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_something, args=(i,)) for i in range(10)]
        for t in threads: t.start()  # noqa: E701
        for t in threads: t.join()  # noqa: E701
        assert not errors, f"Thread safety errors: {errors}"


# =============================================================================
# 10. client/schema_validator.py
# =============================================================================
class TestSchemaValidator:
    def test_valid_dataframe_passes(self, schema):
        import pandas as pd
        from schema_validator import validate_schema
        from shared.model_schema import REQUIRED_COLUMNS

        rng = np.random.default_rng(0)
        data = {}
        for col in REQUIRED_COLUMNS:
            data[col] = ["a"] * 150
        data["overall_survival"] = rng.uniform(0, 5000, 150)
        df = pd.DataFrame(data)
        result = validate_schema(df, schema)
        assert result.passed

    def test_missing_columns_fails(self, schema):
        import pandas as pd
        from schema_validator import validate_schema
        df = pd.DataFrame({"col_x": [1, 2, 3]})
        result = validate_schema(df, schema)
        assert not result.passed
        assert any("Missing" in e for e in result.errors)

    def test_too_few_rows_fails(self, schema):
        import pandas as pd
        from schema_validator import validate_schema
        from shared.model_schema import REQUIRED_COLUMNS
        data = {col: ["a"] * 10 for col in REQUIRED_COLUMNS}
        df = pd.DataFrame(data)
        result = validate_schema(df, schema)
        assert not result.passed
        assert any("rows" in e.lower() for e in result.errors)

    def test_out_of_range_produces_warning(self, schema):
        import pandas as pd
        from schema_validator import validate_schema
        from shared.model_schema import REQUIRED_COLUMNS
        data = {col: ["a"] * 200 for col in REQUIRED_COLUMNS}
        data["age_at_diagnosis"] = [9999.0] * 200   # way out of [0, 120]
        data["overall_survival"] = [100.0] * 200
        df = pd.DataFrame(data)
        result = validate_schema(df, schema)
        # May still pass but should have warnings
        assert len(result.warnings) > 0


# =============================================================================
# 11. client/data_loader.py
# =============================================================================
class TestDataLoader:
    def test_load_tcga_dataset(self, tcga_csv_path, schema):
        from data_loader import load_tcga_dataset
        df = load_tcga_dataset(tcga_csv_path, schema)
        assert len(df) > 0
        assert not df.isnull().any().any()

    def test_preprocess_features_output_shape(self, tcga_csv_path, schema):
        from data_loader import load_tcga_dataset, preprocess_features
        from shared.model_schema import INPUT_DIM
        df = load_tcga_dataset(tcga_csv_path, schema)
        X, y = preprocess_features(df, schema)
        assert X.shape[1] == INPUT_DIM
        assert X.dtype == np.float32
        assert "regression" in y and "toxicity" in y and "binary" in y

    def test_dataloaders_created(self, tcga_csv_path, schema):
        from data_loader import build_dataloaders_from_csv
        train_dl, val_dl = build_dataloaders_from_csv(tcga_csv_path, schema, split=0.2, batch_size=16)
        assert train_dl is not None
        assert val_dl is not None
        # Check we can iterate
        batch = next(iter(train_dl))
        assert len(batch) == 4  # X, y_reg, y_tox, y_bin

    def test_file_not_found_raises(self, schema):
        from data_loader import load_tcga_dataset
        with pytest.raises(FileNotFoundError):
            load_tcga_dataset("/nonexistent/path/data.csv", schema)

    def test_batch_shapes(self, tcga_csv_path, schema):
        from data_loader import build_dataloaders_from_csv
        from shared.model_schema import INPUT_DIM
        train_dl, _ = build_dataloaders_from_csv(tcga_csv_path, schema, split=0.2, batch_size=16)
        X, y_reg, y_tox, y_bin = next(iter(train_dl))
        assert X.shape[1] == INPUT_DIM
        assert y_reg.shape[0] == X.shape[0]
        assert y_tox.shape[0] == X.shape[0]
        assert y_bin.shape[0] == X.shape[0]


# =============================================================================
# 12. server/auth_router.py — JWT helpers
# =============================================================================
class TestJWT:
    def test_create_and_verify_jwt(self):
        from auth_router import create_jwt, verify_jwt
        token = create_jwt("user-abc")
        payload = verify_jwt(token)
        assert payload["sub"] == "user-abc"

    def test_tampered_token_raises(self):
        from auth_router import create_jwt, verify_jwt
        token = create_jwt("user-xyz")
        # Corrupt the signature
        bad_token = token[:-5] + "XXXXX"
        with pytest.raises(Exception):
            verify_jwt(bad_token)
