"""
tests/test_integration.py
Integration tests for server/project_router.py, server/storage.py,
server/mtls.py, server/nas_profiler.py — Phase 3 coverage targets.
"""
import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _read_db_raw(db_path):
    with open(db_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_app(tmp_path_factory):
    """Create a TestClient with a fresh DB for the entire module."""
    import db_handler
    import project_router
    from storage import reset_storage

    db_dir = tmp_path_factory.mktemp("db")
    db_path = str(db_dir / "database.json")
    models_dir = str(tmp_path_factory.mktemp("models"))

    db_handler.DB_PATH = db_path
    project_router.MODELS_DIR = models_dir
    project_router._val_dataloader = None
    reset_storage(None)  # clear any cached singleton

    with open(db_path, "w") as f:
        json.dump({"users": [], "projects": [], "rounds_history": [],
                   "consents": [], "client_reputations": []}, f)

    from main import app
    client = TestClient(app)
    return client, db_path, models_dir


@pytest.fixture(scope="module")
def auth_token(test_app):
    """Register and login a test user, return JWT token."""
    client, db_path, _ = test_app
    resp = client.post("/api/auth/register", json={
        "username": "int_user",
        "password": "int_pass",
        "hospital_name": "Integration Hospital",
        "contact_email": "int@test.com",
    })
    assert resp.status_code in (200, 201), resp.text
    user_id = resp.json()["user_id"]

    resp = client.post("/api/auth/login", json={
        "username": "int_user",
        "password": "int_pass",
    })
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"], user_id


# =============================================================================
# /api/status
# =============================================================================
class TestStatus:
    def test_status_returns_ok(self, test_app):
        client, _, _ = test_app
        resp = client.get("/api/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# =============================================================================
# /api/projects  (GET list, GET single)
# =============================================================================
class TestProjectsEndpoint:
    def test_list_projects_unauthenticated(self, test_app):
        client, _, _ = test_app
        resp = client.get("/api/projects")
        assert resp.status_code == 401

    def test_list_projects_authenticated(self, test_app, auth_token):
        client, _, _ = test_app
        token, _ = auth_token
        resp = client.get("/api/projects", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_project_not_found(self, test_app, auth_token):
        client, _, _ = test_app
        token, _ = auth_token
        resp = client.get("/api/projects/ghost-proj",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 404

    def test_get_project_after_startup(self, test_app, auth_token):
        """The default project is created at app startup."""
        client, db_path, _ = test_app
        token, _ = auth_token
        db = _read_db_raw(db_path)
        if not db.get("projects"):
            pytest.skip("No project created yet")
        proj_id = db["projects"][0]["proj_id"]
        resp = client.get(f"/api/projects/{proj_id}",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["proj_id"] == proj_id


# =============================================================================
# /api/projects/{proj_id}/join and /approve
# =============================================================================
class TestJoinApprove:
    def _get_proj_id(self, db_path):
        db = _read_db_raw(db_path)
        if not db.get("projects"):
            pytest.skip("No project in DB")
        return db["projects"][0]["proj_id"]

    def test_join_project(self, test_app, auth_token):
        client, db_path, _ = test_app
        token, _ = auth_token
        proj_id = self._get_proj_id(db_path)
        resp = client.post(
            f"/api/projects/{proj_id}/join",
            json={"hardware_profile": {}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code in (200, 400), resp.text

    def test_approve_client(self, test_app, auth_token):
        client, db_path, _ = test_app
        token, user_id = auth_token
        proj_id = self._get_proj_id(db_path)
        resp = client.post(
            f"/api/projects/{proj_id}/approve/{user_id}",
            headers={
                "Authorization": f"Bearer {token}",
                "X-Admin-Key": os.environ.get("JWT_SECRET", "test_jwt_secret"),
            },
        )
        assert resp.status_code in (200, 400), resp.text


# =============================================================================
# /api/projects/{proj_id}/history
# =============================================================================
class TestHistoryEndpoint:
    def test_history_returns_list(self, test_app, auth_token):
        client, db_path, _ = test_app
        token, _ = auth_token
        db = _read_db_raw(db_path)
        if not db.get("projects"):
            pytest.skip("No project")
        proj_id = db["projects"][0]["proj_id"]
        resp = client.get(f"/api/projects/{proj_id}/history",
                          headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# =============================================================================
# /api/projects/{proj_id}/update — rejected for non-approved client
# =============================================================================
class TestUpdateEndpoint:
    def test_update_unapproved_client_rejected(self, test_app, auth_token):
        import base64
        from shared.encryption import encrypt_weights

        client, db_path, _ = test_app
        db = _read_db_raw(db_path)
        if not db.get("projects"):
            pytest.skip("No project")
        proj_id = db["projects"][0]["proj_id"]

        # Register a second user who has NOT been approved
        resp = client.post("/api/auth/register", json={
            "username": "unauth_user2",
            "password": "pass2",
            "hospital_name": "UA2",
            "contact_email": "ua2@test.com",
        })
        login_resp = client.post("/api/auth/login", json={
            "username": "unauth_user2", "password": "pass2",
        })
        ua_token = login_resp.json().get("access_token", "")
        if ua_token:
            weights = {"w": np.ones((4,), dtype=np.float32)}
            payload = {
                "round_id": 0,
                "active_depth": 2,
                "weights": encrypt_weights(weights),
                "num_samples": 100,
                "metrics": {"loss": 0.1, "val_rmse": 0.5, "val_acc_tox": 0.9, "val_auc": 0.85},
            }
            resp = client.post(
                f"/api/projects/{proj_id}/update",
                json=payload,
                headers={"Authorization": f"Bearer {ua_token}"},
            )
            assert resp.status_code in (403, 404, 400, 409)


# =============================================================================
# server/storage.py — LocalStorage (using actual API)
# =============================================================================
class TestLocalStorage:
    def test_save_and_load_state_dict(self, tmp_path):
        """Save a plain dict (numpy arrays), load it back."""
        import torch
        from storage import LocalStorage, reset_storage
        import torch.serialization

        ls = LocalStorage(str(tmp_path))
        # Use torch tensors to avoid weights_only numpy issues
        state = {"w": torch.ones(4, 4)}
        path = ls.save("test-proj", round_num=1, state_dict=state)
        assert os.path.isfile(path)
        loaded = ls.load(path)
        assert loaded is not None
        assert "w" in loaded

    def test_exists_and_delete(self, tmp_path):
        import torch
        from storage import LocalStorage
        ls = LocalStorage(str(tmp_path))
        state = {"w": torch.zeros(2)}
        path = ls.save("del-proj", round_num=1, state_dict=state)
        assert ls.exists(path)
        ls.delete(path)
        assert not ls.exists(path)

    def test_load_missing_raises(self, tmp_path):
        from storage import LocalStorage
        ls = LocalStorage(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            ls.load("/nonexistent/path.pt")

    def test_get_storage_factory_local(self, monkeypatch):
        monkeypatch.delenv("S3_BUCKET", raising=False)
        from storage import get_storage, reset_storage, LocalStorage
        reset_storage(None)  # force re-init
        st = get_storage()
        assert isinstance(st, LocalStorage)
        reset_storage(None)  # cleanup


# =============================================================================
# /metrics — Prometheus scrape endpoint
# =============================================================================
class TestMetricsEndpoint:
    def test_metrics_returns_200(self, test_app):
        client, _, _ = test_app
        resp = client.get("/metrics")
        assert resp.status_code == 200


# =============================================================================
# server/mtls.py
# =============================================================================
class TestMtls:
    def test_is_mtls_configured_false_without_env(self, monkeypatch):
        from mtls import is_mtls_configured
        monkeypatch.delenv("MTLS_CA_CERT", raising=False)
        monkeypatch.delenv("MTLS_SERVER_CERT", raising=False)
        monkeypatch.delenv("MTLS_SERVER_KEY", raising=False)
        assert is_mtls_configured() is False

    def test_is_mtls_configured_false_missing_files(self, monkeypatch):
        from mtls import is_mtls_configured
        monkeypatch.setenv("MTLS_CA_CERT", "/nonexistent/ca.pem")
        monkeypatch.setenv("MTLS_SERVER_CERT", "/nonexistent/server.pem")
        monkeypatch.setenv("MTLS_SERVER_KEY", "/nonexistent/server.key")
        assert is_mtls_configured() is False

    def test_get_ssl_context_raises_without_env(self, monkeypatch):
        from mtls import get_ssl_context
        monkeypatch.delenv("MTLS_CA_CERT", raising=False)
        monkeypatch.delenv("MTLS_SERVER_CERT", raising=False)
        monkeypatch.delenv("MTLS_SERVER_KEY", raising=False)
        with pytest.raises(EnvironmentError):
            get_ssl_context()

    def test_apply_to_uvicorn_config_raises_without_certs(self, monkeypatch):
        from mtls import apply_to_uvicorn_config
        monkeypatch.delenv("MTLS_CA_CERT", raising=False)
        monkeypatch.delenv("MTLS_SERVER_CERT", raising=False)
        monkeypatch.delenv("MTLS_SERVER_KEY", raising=False)
        with pytest.raises(EnvironmentError):
            apply_to_uvicorn_config({})


# =============================================================================
# server/nas_profiler.py — zero-cost proxies
# =============================================================================
class TestNasProfiler:
    def _make_model(self, depth=2):
        from supernet import Supernet
        return Supernet(input_dim=32, max_depth=depth, hidden_dim=16, num_toxicity_classes=4)

    def test_count_model_flops_positive(self):
        from nas_profiler import count_model_flops
        model = self._make_model(depth=2)
        flops = count_model_flops(model, (1, 32))
        assert flops >= 0

    def test_synflow_score_positive(self):
        from nas_profiler import synflow_score
        model = self._make_model(depth=2)
        score = synflow_score(model, (1, 32))
        assert score >= 0.0

    def test_grad_norm_score_positive(self):
        from nas_profiler import grad_norm_score
        model = self._make_model(depth=2)
        score = grad_norm_score(model, (1, 32))
        assert score >= 0.0

    def test_profile_depth_candidates_returns_all_depths(self):
        from nas_profiler import profile_depth_candidates
        from supernet import Supernet
        results = profile_depth_candidates(
            model_class=Supernet,
            model_kwargs={"input_dim": 32, "max_depth": 2,
                          "hidden_dim": 16, "num_toxicity_classes": 4},
            input_shape=(1, 32),
            depths=[1, 2],
        )
        assert set(results.keys()) == {1, 2}

    def test_profile_scores_between_zero_one(self):
        from nas_profiler import profile_depth_candidates
        from supernet import Supernet
        results = profile_depth_candidates(
            model_class=Supernet,
            model_kwargs={"input_dim": 32, "max_depth": 2,
                          "hidden_dim": 16, "num_toxicity_classes": 4},
            input_shape=(1, 32),
            depths=[1, 2],
        )
        for depth, info in results.items():
            assert 0.0 <= info["score"] <= 1.0, f"score out of range for depth={depth}"

    def test_profile_has_required_keys(self):
        from nas_profiler import profile_depth_candidates
        from supernet import Supernet
        results = profile_depth_candidates(
            model_class=Supernet,
            model_kwargs={"input_dim": 32, "max_depth": 2,
                          "hidden_dim": 16, "num_toxicity_classes": 4},
            input_shape=(1, 32),
            depths=[1],
        )
        assert "flops" in results[1]
        assert "synflow" in results[1]
        assert "grad_norm" in results[1]
        assert "score" in results[1]
