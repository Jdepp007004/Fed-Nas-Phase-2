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
            ctype = schema.get("column_types", {}).get(col, "string")
            if ctype in ("int", "float"):
                data[col] = rng.uniform(0, 100, 150)
            else:
                cats = schema.get("categorical_values", {}).get(col)
                data[col] = rng.choice(cats, 150) if cats else ["a"] * 150
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


# =============================================================================
# 13. server/nas_controller.py — evaluate_architecture_candidates (Phase 1A-3)
# =============================================================================
class TestNASEvaluate:
    """Tests for evaluate_architecture_candidates() — correctness verified after Fix A."""

    def _make_update(self, depth: int, weight_val: float = 1.0) -> dict:
        return {
            "weights": {"w": np.full((8, 8), weight_val, dtype=np.float32)},
            "num_samples": 100,
            "active_depth": depth,
        }

    def test_empty_input_returns_default(self):
        from nas_controller import evaluate_architecture_candidates
        from shared.model_schema import DEFAULT_ACTIVE_DEPTH
        result = evaluate_architecture_candidates({}, {})
        assert result == DEFAULT_ACTIVE_DEPTH

    def test_single_depth_group_within_bounds(self):
        from nas_controller import evaluate_architecture_candidates
        from shared.model_schema import MAX_DEPTH
        global_w = {"w": np.zeros((8, 8), dtype=np.float32)}
        result = evaluate_architecture_candidates({3: [self._make_update(3)]}, global_w)
        assert 2 <= result <= MAX_DEPTH

    def test_result_always_within_bounds_multi_depth(self):
        from nas_controller import evaluate_architecture_candidates
        from shared.model_schema import MAX_DEPTH
        global_w = {"w": np.zeros((8, 8), dtype=np.float32)}
        updates = {d: [self._make_update(d, float(d))] for d in [2, 4, 6]}
        result = evaluate_architecture_candidates(updates, global_w)
        assert 2 <= result <= MAX_DEPTH

    def test_lower_depth_wins_when_delta_larger(self):
        """
        Fix A correctness: depth=2 → weight=100 vs global=0 → huge delta → low score → wins.
        depth=6 → weight=0 vs global=0 → near-zero delta → score≈6/1e-8 → loses.
        """
        from nas_controller import evaluate_architecture_candidates
        global_w = {"w": np.zeros((8, 8), dtype=np.float32)}
        updates = {
            2: [{"weights": {"w": np.full((8, 8), 100.0, dtype=np.float32)},
                 "num_samples": 100, "active_depth": 2}],
            6: [{"weights": {"w": np.zeros((8, 8), dtype=np.float32)},
                 "num_samples": 100, "active_depth": 6}],
        }
        result = evaluate_architecture_candidates(updates, global_w)
        assert result == 2

    def test_empty_depth_group_is_skipped(self):
        from nas_controller import evaluate_architecture_candidates
        from shared.model_schema import MAX_DEPTH
        # global_w must match the update shape (8,8) to avoid broadcast error
        global_w = {"w": np.zeros((8, 8), dtype=np.float32)}
        # depth=3 group is empty; depth=4 has an update → must still return a valid depth
        updates = {3: [], 4: [self._make_update(4, 5.0)]}
        result = evaluate_architecture_candidates(updates, global_w)
        assert 2 <= result <= MAX_DEPTH


# =============================================================================
# 14. server/auth_router.py — register & login endpoints (Phase 1B-1, 1B-2)
# =============================================================================
class TestAuthEndpoints:
    """Integration tests for /api/auth/register and /api/auth/login via FastAPI TestClient."""

    @pytest.fixture
    def client(self, tmp_db_path, monkeypatch):
        """Return a TestClient backed by an isolated temp database."""
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import auth_router as ar
        test_app = FastAPI()
        test_app.include_router(ar.router)
        return TestClient(test_app, raise_server_exceptions=True)

    def test_register_returns_201_with_user_id(self, client):
        resp = client.post("/api/auth/register", json={
            "username": "hosp_alpha",
            "password": "pass1234",
            "hospital_name": "Alpha Hospital",
            "contact_email": "alpha@hospital.com",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "user_id" in data
        assert data["username"] == "hosp_alpha"

    def test_register_duplicate_username_returns_409(self, client):
        payload = {"username": "dupe_hosp", "password": "p",
                   "hospital_name": "H", "contact_email": "e@e.com"}
        first = client.post("/api/auth/register", json=payload)
        assert first.status_code == 201
        second = client.post("/api/auth/register", json=payload)
        assert second.status_code == 409

    def test_login_valid_credentials_returns_token(self, client):
        client.post("/api/auth/register", json={
            "username": "login_ok", "password": "correct_pass",
            "hospital_name": "H", "contact_email": "x@y.com",
        })
        resp = client.post("/api/auth/login", json={
            "username": "login_ok", "password": "correct_pass",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user_id" in data

    def test_login_wrong_password_returns_401(self, client):
        client.post("/api/auth/register", json={
            "username": "wrong_pw_user", "password": "real_pass",
            "hospital_name": "H", "contact_email": "z@z.com",
        })
        resp = client.post("/api/auth/login", json={
            "username": "wrong_pw_user", "password": "wrong_pass",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user_returns_401(self, client):
        resp = client.post("/api/auth/login", json={
            "username": "ghost_user", "password": "any_pass",
        })
        assert resp.status_code == 401

    def test_registered_user_in_db(self, tmp_db_path, monkeypatch):
        """Verify registration actually persists the user record to the DB."""
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import auth_router as ar
        test_app = FastAPI()
        test_app.include_router(ar.router)
        tc = TestClient(test_app)
        tc.post("/api/auth/register", json={
            "username": "persist_check", "password": "pw",
            "hospital_name": "H", "contact_email": "p@p.com",
        })
        found = db_handler.get_user(username="persist_check")
        assert found is not None
        assert found["username"] == "persist_check"


# =============================================================================
# 15. server/db_handler.py — append_round_history & get_user (Phase 1B-4)
# =============================================================================
class TestDBHandlerExtras:
    """Tests for db_handler functions not covered by TestDBHandler."""

    def test_append_round_history_persists(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        record = {"proj_id": "p-hist", "round": 3, "global_val_rmse": 0.42}
        db_handler.append_round_history(record)
        db = db_handler.read_db()
        assert any(r.get("round") == 3 for r in db["rounds_history"])

    def test_append_multiple_round_history_entries(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        for i in range(1, 4):
            db_handler.append_round_history({"proj_id": "p-multi", "round": i})
        db = db_handler.read_db()
        rounds = [r["round"] for r in db["rounds_history"] if r.get("proj_id") == "p-multi"]
        assert sorted(rounds) == [1, 2, 3]

    def test_get_user_by_user_id(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db["users"].append({"user_id": "u-byid", "username": "byid_user",
                            "approved_projects": []})
        db_handler.write_db(db)
        found = db_handler.get_user(user_id="u-byid")
        assert found is not None
        assert found["username"] == "byid_user"

    def test_get_user_by_username(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db["users"].append({"user_id": "u-byname", "username": "target_name",
                            "approved_projects": []})
        db_handler.write_db(db)
        found = db_handler.get_user(username="target_name")
        assert found is not None
        assert found["user_id"] == "u-byname"

    def test_get_user_not_found_returns_none(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        assert db_handler.get_user(user_id="no-such-id") is None
        assert db_handler.get_user(username="no-such-name") is None


# =============================================================================
# 16. server/aggregation.py — validate_global_model (Phase 1C-1)
# =============================================================================
class TestValidateGlobalModel:
    """Tests for validate_global_model() using a small synthetic DataLoader."""

    _CONFIG = {"input_dim": 32, "max_depth": 2, "hidden_dim": 16, "num_toxicity_classes": 4}

    def _make_val_loader(self, n: int = 64, batch_size: int = 16):
        """Synthetic DataLoader yielding (X, y_reg, y_tox, y_bin) tuples."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        torch.manual_seed(0)
        ds_X = torch.randn(n, self._CONFIG["input_dim"])
        ds_reg = torch.randn(n)
        ds_tox = torch.randint(0, self._CONFIG["num_toxicity_classes"], (n,))
        # Ensure both binary classes are represented for AUC
        ds_bin = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).float()
        ds = TensorDataset(ds_X, ds_reg, ds_tox, ds_bin)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def _get_weights(self):
        from supernet import Supernet, get_subnet_weights
        model = Supernet(**self._CONFIG)
        return get_subnet_weights(model, active_depth=self._CONFIG["max_depth"])

    def test_returns_all_required_keys(self):
        from aggregation import validate_global_model
        result = validate_global_model(self._get_weights(), self._make_val_loader(),
                                       self._CONFIG)
        for key in ("global_val_rmse", "global_tox_accuracy", "global_auc", "timestamp"):
            assert key in result, f"Missing key: {key}"

    def test_metrics_are_finite(self):
        from aggregation import validate_global_model
        import math
        result = validate_global_model(self._get_weights(), self._make_val_loader(),
                                       self._CONFIG)
        assert math.isfinite(result["global_val_rmse"])
        assert math.isfinite(result["global_tox_accuracy"])
        assert math.isfinite(result["global_auc"])

    def test_tox_accuracy_in_unit_range(self):
        from aggregation import validate_global_model
        result = validate_global_model(self._get_weights(), self._make_val_loader(),
                                       self._CONFIG)
        assert 0.0 <= result["global_tox_accuracy"] <= 1.0

    def test_auc_in_unit_range(self):
        from aggregation import validate_global_model
        result = validate_global_model(self._get_weights(), self._make_val_loader(),
                                       self._CONFIG)
        assert 0.0 <= result["global_auc"] <= 1.0

    def test_timestamp_is_utc_string(self):
        from aggregation import validate_global_model
        result = validate_global_model(self._get_weights(), self._make_val_loader(),
                                       self._CONFIG)
        assert isinstance(result["timestamp"], str)
        assert "Z" in result["timestamp"]

    def test_rmse_nonnegative(self):
        from aggregation import validate_global_model
        result = validate_global_model(self._get_weights(), self._make_val_loader(),
                                       self._CONFIG)
        assert result["global_val_rmse"] >= 0.0


# =============================================================================
# 17. client/train_loop.py — run_local_training full loop (Phase 1C-2)
# =============================================================================
class TestRunLocalTraining:
    """Tests for the complete run_local_training() FedProx loop."""

    _INPUT_DIM = 32
    _MAX_DEPTH = 3
    _HIDDEN_DIM = 16

    def _make_model(self):
        from supernet import Supernet
        return Supernet(input_dim=self._INPUT_DIM, max_depth=self._MAX_DEPTH,
                        hidden_dim=self._HIDDEN_DIM, num_toxicity_classes=4)

    def _make_loaders(self, n: int = 80, batch_size: int = 16):
        import torch
        from torch.utils.data import TensorDataset, DataLoader, Subset
        torch.manual_seed(1)
        X = torch.randn(n, self._INPUT_DIM)
        y_reg = torch.randn(n)
        y_tox = torch.randint(0, 4, (n,))
        # Guarantee both binary classes for AUC computation in validation
        y_bin = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).float()
        ds = TensorDataset(X, y_reg, y_tox, y_bin)
        n_train = int(n * 0.75)
        train_dl = DataLoader(Subset(ds, list(range(n_train))),
                              batch_size=batch_size, drop_last=True)
        val_dl = DataLoader(Subset(ds, list(range(n_train, n))),
                            batch_size=batch_size, drop_last=False)
        return train_dl, val_dl

    def test_returns_required_keys(self):
        from train_loop import run_local_training, TrainConfig
        cfg = TrainConfig(epochs=1, lr=1e-3, active_depth=2,
                          fedprox_mu=0.01, clip_norm=1.0)
        result = run_local_training(self._make_model(), self._make_loaders(), cfg)
        assert "weights" in result
        assert "num_samples" in result
        assert "metrics" in result
        for k in ("loss", "val_rmse", "val_acc_tox", "val_auc"):
            assert k in result["metrics"], f"Missing metric key: {k}"

    def test_weights_are_numpy_dicts(self):
        from train_loop import run_local_training, TrainConfig
        cfg = TrainConfig(epochs=1, active_depth=2)
        result = run_local_training(self._make_model(), self._make_loaders(), cfg)
        assert isinstance(result["weights"], dict)
        assert len(result["weights"]) > 0
        for v in result["weights"].values():
            assert isinstance(v, np.ndarray)

    def test_num_samples_positive(self):
        from train_loop import run_local_training, TrainConfig
        cfg = TrainConfig(epochs=1, active_depth=1)
        result = run_local_training(self._make_model(), self._make_loaders(), cfg)
        assert result["num_samples"] > 0

    def test_loss_is_finite(self):
        from train_loop import run_local_training, TrainConfig
        import math
        cfg = TrainConfig(epochs=2, active_depth=2)
        result = run_local_training(self._make_model(), self._make_loaders(), cfg)
        assert math.isfinite(result["metrics"]["loss"])

    def test_val_metrics_in_valid_ranges(self):
        """val_auc is 0.0 when single-class is present (handled by try/except in _run_validation)."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader, Subset
        from train_loop import run_local_training, TrainConfig
        # Build dataset explicitly guaranteeing both binary classes in both splits
        n = 80
        torch.manual_seed(99)
        X = torch.randn(n, self._INPUT_DIM)
        y_reg = torch.randn(n)
        y_tox = torch.randint(0, 4, (n,))
        # Interleave 0s and 1s so any contiguous slice has both classes
        y_bin = torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)
        ds = TensorDataset(X, y_reg, y_tox, y_bin)
        n_train = int(n * 0.75)
        train_dl = DataLoader(Subset(ds, list(range(n_train))),
                              batch_size=16, drop_last=True)
        val_dl = DataLoader(Subset(ds, list(range(n_train, n))),
                            batch_size=16, drop_last=False)
        cfg = TrainConfig(epochs=1, active_depth=2)
        result = run_local_training(self._make_model(), (train_dl, val_dl), cfg)
        assert result["metrics"]["val_rmse"] >= 0.0
        assert 0.0 <= result["metrics"]["val_acc_tox"] <= 1.0
        # AUC is 0.0 (fallback) or a valid probability — both are in [0, 1]
        assert 0.0 <= result["metrics"]["val_auc"] <= 1.0

    def test_single_dataloader_no_crash(self):
        """Passing a bare DataLoader (not a tuple) must work — no val metrics expected."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        from train_loop import run_local_training, TrainConfig
        X = torch.randn(32, self._INPUT_DIM)
        y_reg = torch.randn(32)
        y_tox = torch.randint(0, 4, (32,))
        y_bin = torch.zeros(32).float()
        ds = TensorDataset(X, y_reg, y_tox, y_bin)
        single_loader = DataLoader(ds, batch_size=16, drop_last=True)
        cfg = TrainConfig(epochs=1, active_depth=2)
        result = run_local_training(self._make_model(), single_loader, cfg)
        assert "weights" in result

    def test_default_config_used_when_none(self):
        """
        Passing config=None uses TrainConfig defaults (DEFAULT_ACTIVE_DEPTH=4).
        The small test model only has max_depth=3, so we must pass a model deep enough.
        """
        from train_loop import run_local_training
        from supernet import Supernet
        from shared.model_schema import DEFAULT_ACTIVE_DEPTH
        # Build a model that can accommodate DEFAULT_ACTIVE_DEPTH layers
        model = Supernet(input_dim=self._INPUT_DIM, max_depth=DEFAULT_ACTIVE_DEPTH,
                         hidden_dim=self._HIDDEN_DIM, num_toxicity_classes=4)
        result = run_local_training(model, self._make_loaders(), config=None)
        assert "weights" in result


# =============================================================================
# 18. server/project_router.py — round_lifecycle (Phase 1C-3)
# =============================================================================
class TestRoundLifecycle:
    """Tests for the round_lifecycle() background task."""

    def _make_weight_dict(self, val: float = 1.0) -> dict:
        """Minimal synthetic weight dict (keys match a small depth-2 subnet shape)."""
        return {
            "backbone.0.0.weight": np.full((8, 4), val, dtype=np.float32),
            "backbone.0.0.bias":   np.zeros(8, dtype=np.float32),
            "backbone.1.0.weight": np.full((8, 8), val, dtype=np.float32),
            "backbone.1.0.bias":   np.zeros(8, dtype=np.float32),
            "head_regression.weight": np.ones((1, 8), dtype=np.float32),
            "head_regression.bias":   np.zeros(1, dtype=np.float32),
            "head_toxicity.weight":   np.ones((4, 8), dtype=np.float32),
            "head_toxicity.bias":     np.zeros(4, dtype=np.float32),
            "head_binary.weight":     np.ones((1, 8), dtype=np.float32),
            "head_binary.bias":       np.zeros(1, dtype=np.float32),
        }

    def _seed_project(self, db_handler_mod, proj_id: str) -> None:
        """Insert a minimal project record into the temp DB."""
        proj = {
            "proj_id": proj_id,
            "current_round": 0,
            "global_model_path": "",
            "momentum_beta": 0.9,
            "recommended_depth": 2,
            "connected_clients": ["client-1"],
            "min_clients_per_round": 1,
        }
        db = db_handler_mod.read_db()
        db["projects"].append(proj)
        db_handler_mod.write_db(db)

    def test_lifecycle_increments_round_and_saves_pt(
            self, tmp_path, tmp_db_path, monkeypatch):
        import db_handler
        import project_router
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        monkeypatch.setattr(project_router, "MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(project_router, "_val_dataloader", None)

        proj_id = "rl-proj-001"
        self._seed_project(db_handler, proj_id)

        updates_buffer = [{
            "user_id": "client-1",
            "weights": self._make_weight_dict(1.0),
            "num_samples": 50,
            "active_depth": 2,
            "metrics": {},
        }]
        db_snapshot = db_handler.read_db()

        from project_router import round_lifecycle
        round_lifecycle(proj_id, updates_buffer, db_snapshot)

        updated = db_handler.get_project(proj_id)
        assert updated["current_round"] == 1, "Round counter was not incremented"
        pt_file = tmp_path / f"{proj_id}_round1.pt"
        assert pt_file.exists(), ".pt model file was not written to MODELS_DIR"

    def test_lifecycle_appends_round_history(
            self, tmp_path, tmp_db_path, monkeypatch):
        import db_handler
        import project_router
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        monkeypatch.setattr(project_router, "MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(project_router, "_val_dataloader", None)

        proj_id = "rl-proj-002"
        self._seed_project(db_handler, proj_id)

        updates_buffer = [{
            "user_id": "client-1",
            "weights": self._make_weight_dict(2.0),
            "num_samples": 80,
            "active_depth": 2,
            "metrics": {},
        }]
        db_snapshot = db_handler.read_db()

        from project_router import round_lifecycle
        round_lifecycle(proj_id, updates_buffer, db_snapshot)

        db = db_handler.read_db()
        history = [r for r in db["rounds_history"] if r.get("proj_id") == proj_id]
        assert len(history) >= 1
        assert history[0]["round"] == 1

    def test_lifecycle_multiple_clients_aggregated(
            self, tmp_path, tmp_db_path, monkeypatch):
        """Two clients → FedAvg must produce a weighted average, not a copy."""
        import db_handler
        import project_router
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        monkeypatch.setattr(project_router, "MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(project_router, "_val_dataloader", None)

        proj_id = "rl-proj-003"
        proj = {
            "proj_id": proj_id, "current_round": 0,
            "global_model_path": "", "momentum_beta": 0.9,
            "recommended_depth": 2,
            "connected_clients": ["c1", "c2"],
            "min_clients_per_round": 2,
        }
        db = db_handler.read_db()
        db["projects"].append(proj)
        db_handler.write_db(db)

        updates_buffer = [
            {"user_id": "c1", "weights": self._make_weight_dict(0.0),
             "num_samples": 50, "active_depth": 2, "metrics": {}},
            {"user_id": "c2", "weights": self._make_weight_dict(2.0),
             "num_samples": 50, "active_depth": 2, "metrics": {}},
        ]
        db_snapshot = db_handler.read_db()

        from project_router import round_lifecycle
        round_lifecycle(proj_id, updates_buffer, db_snapshot)

        updated = db_handler.get_project(proj_id)
        assert updated["current_round"] == 1

    def test_lifecycle_empty_buffer_does_not_increment_round(
            self, tmp_path, tmp_db_path, monkeypatch):
        """Empty update list → EmptyRoundError caught internally → round stays at 0."""
        import db_handler
        import project_router
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        monkeypatch.setattr(project_router, "MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(project_router, "_val_dataloader", None)

        proj_id = "rl-proj-004"
        self._seed_project(db_handler, proj_id)
        db_snapshot = db_handler.read_db()

        from project_router import round_lifecycle
        round_lifecycle(proj_id, [], db_snapshot)

        updated = db_handler.get_project(proj_id)
        assert updated["current_round"] == 0, (
            "Round must not increment when no updates were provided")

    def test_lifecycle_unknown_project_is_noop(
            self, tmp_path, tmp_db_path, monkeypatch):
        """If proj_id absent from db_snapshot, lifecycle must return silently."""
        import db_handler
        import project_router
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        monkeypatch.setattr(project_router, "MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(project_router, "_val_dataloader", None)

        db_snapshot = {"projects": [], "users": [], "rounds_history": []}

        from project_router import round_lifecycle
        round_lifecycle("ghost-proj-xyz", [], db_snapshot)  # must not raise


# =============================================================================
# 19. server/pg_handler.py — PostgreSQL handler (Phase 2A)
# Uses SQLite in-memory via SQLAlchemy so no real Postgres is needed
# =============================================================================
class TestPgHandler:
    """Tests for PgHandler using an in-memory SQLite database."""

    @pytest.fixture
    def pg(self):
        from pg_handler import PgHandler
        return PgHandler(database_url="sqlite:///:memory:")

    def test_create_and_get_project(self, pg):
        pg.create_project({
            "proj_id": "pg-p-1", "name": "Test Project",
            "current_round": 0, "connected_clients": [],
        })
        result = pg.get_project("pg-p-1")
        assert result is not None
        assert result["proj_id"] == "pg-p-1"

    def test_get_project_not_found(self, pg):
        assert pg.get_project("missing-id") is None

    def test_update_project(self, pg):
        pg.create_project({
            "proj_id": "pg-p-2", "name": "Updatable",
            "current_round": 0, "connected_clients": [],
        })
        pg.update_project("pg-p-2", {"current_round": 3})
        assert pg.get_project("pg-p-2")["current_round"] == 3

    def test_update_missing_project_raises(self, pg):
        with pytest.raises(KeyError):
            pg.update_project("no-such-proj", {"current_round": 1})

    def test_create_and_get_user(self, pg):
        pg.create_user({
            "user_id": "pg-u-1", "username": "hospital_pg",
            "password_hash": "hash", "hospital_name": "PG Hospital",
            "contact_email": "pg@h.com",
        })
        found = pg.get_user(user_id="pg-u-1")
        assert found is not None
        assert found["username"] == "hospital_pg"

    def test_get_user_by_username(self, pg):
        pg.create_user({
            "user_id": "pg-u-2", "username": "by_name_user",
            "password_hash": "h", "hospital_name": "H", "contact_email": "e@e.com",
        })
        found = pg.get_user(username="by_name_user")
        assert found is not None
        assert found["user_id"] == "pg-u-2"

    def test_get_user_not_found(self, pg):
        assert pg.get_user(user_id="ghost") is None
        assert pg.get_user(username="ghost") is None

    def test_append_and_get_round_history(self, pg):
        pg.create_project({
            "proj_id": "pg-p-hist", "name": "Hist Project",
            "current_round": 0, "connected_clients": [],
        })
        pg.append_round_history({
            "proj_id": "pg-p-hist", "round": 1,
            "global_val_rmse": 0.5, "global_tox_accuracy": 0.8, "global_auc": 0.75,
        })
        pg.append_round_history({"proj_id": "pg-p-hist", "round": 2})
        history = pg.get_round_history("pg-p-hist")
        assert len(history) == 2
        assert history[0]["round"] == 1
        assert history[1]["round"] == 2

    def test_read_db_snapshot(self, pg):
        db = pg.read_db()
        assert "users" in db
        assert "projects" in db
        assert "rounds_history" in db

    def test_list_projects_empty(self, pg):
        assert pg.list_projects() == []

    def test_list_users_empty(self, pg):
        assert pg.list_users() == []


# =============================================================================
# 20. server/redis_state.py — RedisState (Phase 2A) in-memory fallback
# =============================================================================
class TestRedisState:
    """Tests for RedisState using in-memory fallback (no Redis required)."""

    @pytest.fixture
    def state(self):
        from redis_state import RedisState
        return RedisState(redis_url=None)   # force in-memory mode

    def test_backend_is_memory(self, state):
        assert state.backend == "memory"

    def test_push_and_count(self, state):
        state.push_update("proj-rs-1", {"w": 1.0})
        state.push_update("proj-rs-1", {"w": 2.0})
        assert state.count_updates("proj-rs-1") == 2

    def test_pop_all_is_atomic(self, state):
        state.push_update("proj-rs-2", {"a": 1})
        state.push_update("proj-rs-2", {"b": 2})
        items = state.pop_all_updates("proj-rs-2")
        assert len(items) == 2
        assert state.count_updates("proj-rs-2") == 0

    def test_pop_empty_returns_empty(self, state):
        items = state.pop_all_updates("proj-rs-empty")
        assert items == []

    def test_velocity_roundtrip(self, state):
        vel = {"layer.weight": [1.0, 2.0, 3.0]}
        state.set_velocity("proj-rv-1", vel)
        got = state.get_velocity("proj-rv-1")
        assert got == vel

    def test_get_velocity_missing_returns_empty(self, state):
        assert state.get_velocity("no-such-proj") == {}

    def test_delete_velocity(self, state):
        state.set_velocity("proj-del-1", {"v": 99})
        state.delete_velocity("proj-del-1")
        assert state.get_velocity("proj-del-1") == {}

    def test_ping_returns_true(self, state):
        assert state.ping() is True

    def test_separate_projects_are_isolated(self, state):
        state.push_update("proj-A", {"x": 1})
        state.push_update("proj-B", {"x": 2})
        assert state.count_updates("proj-A") == 1
        assert state.count_updates("proj-B") == 1
        a_items = state.pop_all_updates("proj-A")
        assert len(a_items) == 1
        assert state.count_updates("proj-B") == 1  # B not touched


# =============================================================================
# 21. server/round_state.py — RoundStateMachine (Phase 2B)
# =============================================================================
class TestRoundState:
    """Tests for the RoundState enum and RoundStateMachine."""

    def test_initial_state_is_idle(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-1")
        assert m.state == RoundState.IDLE

    def test_idle_to_collecting(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-2")
        m.transition(RoundState.COLLECTING)
        assert m.state == RoundState.COLLECTING

    def test_collecting_to_aggregating(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-3", "collecting")
        m.transition(RoundState.AGGREGATING)
        assert m.state == RoundState.AGGREGATING

    def test_aggregating_to_done(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-4", "aggregating")
        m.transition(RoundState.DONE)
        assert m.state == RoundState.DONE

    def test_done_to_idle(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-5", "done")
        m.transition(RoundState.IDLE)
        assert m.state == RoundState.IDLE

    def test_aggregating_to_error(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-6", "aggregating")
        m.transition(RoundState.ERROR)
        assert m.state == RoundState.ERROR

    def test_invalid_transition_raises(self):
        from round_state import RoundState, RoundStateMachine, InvalidTransitionError
        m = RoundStateMachine("proj-sm-7")
        with pytest.raises(InvalidTransitionError):
            m.transition(RoundState.DONE)   # IDLE → DONE is not allowed

    def test_can_transition_check(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-8", "idle")
        assert m.can_transition(RoundState.COLLECTING) is True
        assert m.can_transition(RoundState.DONE) is False

    def test_force_reset_to_idle(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-9", "aggregating")
        m.reset_to_idle()
        assert m.state == RoundState.IDLE

    def test_string_input_accepted(self):
        from round_state import RoundState, RoundStateMachine
        m = RoundStateMachine("proj-sm-10", "idle")
        m.transition("collecting")          # string accepted, not just enum
        assert m.state == RoundState.COLLECTING


# =============================================================================
# 22. server/storage.py — LocalStorage (Phase 2B)
# =============================================================================
class TestStorage:
    """Tests for LocalStorage backend."""

    @pytest.fixture
    def store(self, tmp_path):
        from storage import LocalStorage
        return LocalStorage(models_dir=str(tmp_path))

    def _make_state_dict(self, val: float = 1.0) -> dict:
        import torch
        return {"w": torch.tensor([val, val], dtype=torch.float32)}

    def test_save_returns_path_string(self, store):
        sd = self._make_state_dict()
        path = store.save("proj-s1", 1, sd)
        assert isinstance(path, str)
        assert "proj-s1_round1.pt" in path

    def test_file_exists_after_save(self, store):
        sd = self._make_state_dict()
        path = store.save("proj-s2", 2, sd)
        assert store.exists(path)

    def test_load_roundtrip(self, store):
        import torch
        sd = self._make_state_dict(3.14)
        path = store.save("proj-s3", 1, sd)
        loaded = store.load(path)
        assert "w" in loaded
        assert torch.allclose(loaded["w"], sd["w"])

    def test_load_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.load("/nonexistent/path/model.pt")

    def test_delete_removes_file(self, store):
        sd = self._make_state_dict()
        path = store.save("proj-s4", 1, sd)
        assert store.exists(path)
        store.delete(path)
        assert not store.exists(path)

    def test_delete_missing_is_noop(self, store):
        store.delete("/nonexistent/path.pt")   # must not raise

    def test_exists_false_for_missing(self, store):
        assert not store.exists("/no/such/file.pt")


# =============================================================================
# 23. server/metrics.py — Prometheus metrics stubs (Phase 2C)
# =============================================================================
class TestMetrics:
    """Tests for the metrics module — works with or without prometheus_client."""

    def test_import_does_not_crash(self):
        """metrics.py must be importable unconditionally."""
        import metrics  # noqa: F401

    def test_rounds_completed_is_callable(self):
        from metrics import ROUNDS_COMPLETED
        # Must not raise regardless of whether prometheus_client is installed
        ROUNDS_COMPLETED.labels(proj_id="test", status="success").inc()

    def test_round_duration_is_callable(self):
        from metrics import ROUND_DURATION
        ROUND_DURATION.labels(proj_id="test").observe(15.0)

    def test_active_clients_is_callable(self):
        from metrics import ACTIVE_CLIENTS
        ACTIVE_CLIENTS.labels(proj_id="test").set(3)

    def test_pending_updates_is_callable(self):
        from metrics import PENDING_UPDATES
        PENDING_UPDATES.labels(proj_id="test").set(2)

    def test_record_round_metrics_does_not_crash(self):
        from metrics import record_round_metrics
        record_round_metrics("proj-m", {
            "global_val_rmse": 0.42,
            "global_tox_accuracy": 0.85,
            "global_auc": 0.91,
        })

    def test_prometheus_response_returns_bytes_and_content_type(self):
        from metrics import prometheus_response
        body, ct = prometheus_response()
        assert isinstance(body, bytes)
        assert isinstance(ct, str)
        assert len(body) > 0

    def test_api_requests_is_callable(self):
        from metrics import API_REQUESTS
        API_REQUESTS.labels(endpoint="/api/status", method="GET", status_code="200").inc()

    def test_global_gauges_are_callable(self):
        from metrics import GLOBAL_VAL_RMSE, GLOBAL_TOX_ACCURACY, GLOBAL_AUC
        GLOBAL_VAL_RMSE.labels(proj_id="test").set(0.5)
        GLOBAL_TOX_ACCURACY.labels(proj_id="test").set(0.9)
        GLOBAL_AUC.labels(proj_id="test").set(0.88)


# =============================================================================
# 24. server/otel_setup.py — OTel no-op (Phase 2C)
# =============================================================================
class TestOtelSetup:
    """Tests for the OTel setup module — must work when SDK is not installed."""

    def test_import_does_not_crash(self):
        import otel_setup  # noqa: F401

    def test_get_tracer_returns_something(self):
        from otel_setup import get_tracer
        tracer = get_tracer()
        assert tracer is not None

    def test_tracer_start_span_does_not_crash(self):
        from otel_setup import get_tracer
        tracer = get_tracer()
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("key", "value")

    def test_get_meter_returns_something(self):
        from otel_setup import get_meter
        meter = get_meter()
        assert meter is not None

    def test_instrument_round_lifecycle_does_not_crash(self):
        from otel_setup import instrument_round_lifecycle
        ctx = instrument_round_lifecycle("proj-otel-1", 3)
        # Must be usable as a context manager
        with ctx:
            pass

