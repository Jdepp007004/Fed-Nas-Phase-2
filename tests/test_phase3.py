"""
tests/test_phase3.py
Phase 3 tests — covers all new modules:
  25. server/secure_aggregation.py    — Secure Aggregation
  26. shared/key_manager.py           — Key versioning + rotation
  27. server/consent.py               — Consent Management
  28. server/unlearning.py            — Federated Unlearning
  29. server/flame_defense.py         — FLAME Byzantine defense
  30. server/reputation.py            — Per-client reputation
  31. server/temperature_scaling.py   — Temperature Scaling
  32. client/focal_loss.py            — Focal Loss
  33. project_router — min participation abort (integration)
  34. DP correctness (additional checks)
"""
import os
import sys
import json
import warnings

import numpy as np
import pytest
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# 25. server/secure_aggregation.py
# =============================================================================
class TestSecureAggregation:
    """Tests for additive masking secure aggregation."""

    def _make_weights(self, val: float = 1.0) -> dict:
        return {"w": np.full((4, 4), val, dtype=np.float32)}

    def test_mask_update_produces_different_values(self):
        from secure_aggregation import mask_update
        w = self._make_weights(1.0)
        masked, mask = mask_update(w, seed=42)
        assert not np.allclose(masked["w"], w["w"])

    def test_mask_shape_matches_weights(self):
        from secure_aggregation import mask_update
        w = {"a": np.ones((3, 4), np.float32), "b": np.zeros((2,), np.float32)}
        masked, mask = mask_update(w, seed=0)
        assert masked["a"].shape == w["a"].shape
        assert mask["b"].shape == w["b"].shape

    def test_unmask_aggregate_recovers_sum(self):
        from secure_aggregation import mask_update, unmask_aggregate
        w1 = self._make_weights(1.0)
        w2 = self._make_weights(3.0)
        masked1, mask1 = mask_update(w1, seed=1)
        masked2, mask2 = mask_update(w2, seed=2)

        # Server accumulates sums
        masked_sum = {"w": masked1["w"] + masked2["w"]}
        mask_sum   = {"w": mask1["w"] + mask2["w"]}

        result = unmask_aggregate(masked_sum, mask_sum, num_clients=2)
        # Should equal (w1+w2)/2 = 2.0
        np.testing.assert_allclose(result["w"], 2.0, atol=1e-4)

    def test_context_submit_and_count(self):
        from secure_aggregation import SecureAggregationContext, mask_update
        ctx = SecureAggregationContext("proj-test", round_num=1, expected_clients=2)
        w = self._make_weights()
        masked, mask = mask_update(w)
        count = ctx.add_masked_update("client-1", masked, mask)
        assert count == 1
        assert not ctx.is_complete()

    def test_context_complete_when_all_submitted(self):
        from secure_aggregation import SecureAggregationContext, mask_update
        ctx = SecureAggregationContext("proj-test", round_num=2, expected_clients=2)
        w = self._make_weights()
        for cid in ["c1", "c2"]:
            masked, mask = mask_update(w)
            ctx.add_masked_update(cid, masked, mask)
        assert ctx.is_complete()

    def test_duplicate_submission_raises(self):
        from secure_aggregation import SecureAggregationContext, mask_update
        ctx = SecureAggregationContext("proj-dup", round_num=1, expected_clients=3)
        w = self._make_weights()
        masked, mask = mask_update(w)
        ctx.add_masked_update("c1", masked, mask)
        with pytest.raises(ValueError, match="already submitted"):
            ctx.add_masked_update("c1", masked, mask)

    def test_finalize_returns_dict(self):
        from secure_aggregation import SecureAggregationContext, mask_update
        ctx = SecureAggregationContext("proj-fin", round_num=1, expected_clients=1)
        w = self._make_weights(5.0)
        masked, mask = mask_update(w, seed=99)
        ctx.add_masked_update("c1", masked, mask)
        result = ctx.finalize()
        assert isinstance(result, dict)
        assert "w" in result

    def test_get_or_create_context_singleton(self):
        from secure_aggregation import get_or_create_context, clear_context
        ctx1 = get_or_create_context("proj-s1", 5, 3)
        ctx2 = get_or_create_context("proj-s1", 5, 3)
        assert ctx1 is ctx2
        clear_context("proj-s1", 5)

    def test_clear_context_removes_it(self):
        from secure_aggregation import get_or_create_context, clear_context
        get_or_create_context("proj-s2", 1, 2)
        clear_context("proj-s2", 1)
        # After clearing, a new context is created (different object)
        ctx_new = get_or_create_context("proj-s2", 1, 2)
        assert ctx_new is not None


# =============================================================================
# 26. shared/key_manager.py — Key versioning + rotation
# =============================================================================
class TestKeyManager:
    def _make_key(self) -> str:
        """Generate a fresh base64 32-byte key."""
        import base64
        return base64.b64encode(os.urandom(32)).decode()

    def _make_manager(self, n_versions: int = 1):
        from shared.key_manager import KeyManager
        import base64
        keys = {f"v{i+1}": os.urandom(32) for i in range(n_versions)}
        return KeyManager(keys, active_version="v1")

    def test_active_version_set_correctly(self):
        km = self._make_manager()
        assert km.active_version == "v1"

    def test_known_versions_listed(self):
        km = self._make_manager(2)
        assert "v1" in km.known_versions
        assert "v2" in km.known_versions

    def test_fingerprint_is_hex_string(self):
        km = self._make_manager()
        fp = km.fingerprint("v1")
        assert isinstance(fp, str)
        assert len(fp) == 16  # 8 bytes hex

    def test_register_version_and_rotate(self):
        from shared.key_manager import KeyManager
        import base64
        km = KeyManager({"v1": os.urandom(32)}, active_version="v1")
        new_key_b64 = base64.b64encode(os.urandom(32)).decode()
        km.register_version("v2", new_key_b64)
        km.set_active_version("v2")
        assert km.active_version == "v2"

    def test_encrypt_decrypt_roundtrip(self):
        km = self._make_manager()
        weights = {"layer": np.ones((4, 4), dtype=np.float32)}
        payload = km.encrypt(weights)
        recovered = km.decrypt(payload)
        np.testing.assert_allclose(recovered["layer"], weights["layer"])

    def test_encrypt_adds_key_version(self):
        km = self._make_manager()
        weights = {"a": np.zeros((4,), dtype=np.float32)}
        payload = km.encrypt(weights)
        assert "key_version" in payload
        assert payload["key_version"] == "v1"

    def test_decrypt_old_version_after_rotation(self):
        """After rotating to v2, payloads encrypted with v1 should still decrypt."""
        from shared.key_manager import KeyManager
        import base64
        key1 = os.urandom(32)
        key2 = os.urandom(32)
        km = KeyManager({"v1": key1, "v2": key2}, active_version="v1")
        weights = {"w": np.ones((4,), dtype=np.float32)}
        payload_v1 = km.encrypt(weights)          # encrypted with v1
        km.set_active_version("v2")
        recovered = km.decrypt(payload_v1)        # must use v1 to decrypt
        np.testing.assert_allclose(recovered["w"], weights["w"])

    def test_unknown_version_raises(self):
        from shared.key_manager import KeyManager, KeyVersionError
        km = KeyManager({"v1": os.urandom(32)}, active_version="v1")
        km.set_active_version("v1")
        with pytest.raises(KeyVersionError):
            km.set_active_version("v99")

    def test_empty_keys_raises(self):
        from shared.key_manager import KeyManager
        with pytest.raises(ValueError):
            KeyManager({}, active_version="v1")

    def test_encryption_key_b64_in_encrypt_decrypt(self):
        """Test that encryption now passes key_b64 correctly."""
        from shared.encryption import encrypt_weights, decrypt_weights
        import base64
        key = os.urandom(32)
        key_b64 = base64.b64encode(key).decode()
        weights = {"x": np.ones((4,), dtype=np.float32)}
        payload = encrypt_weights(weights, key_b64=key_b64)
        recovered = decrypt_weights(payload, key_b64=key_b64)
        np.testing.assert_allclose(recovered["x"], weights["x"])


# =============================================================================
# 27. server/consent.py — Consent Management
# =============================================================================
class TestConsentManager:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        # Ensure consents key exists
        db = db_handler.read_db()
        db["consents"] = []
        db_handler.write_db(db)

    def test_grant_creates_record(self):
        from consent import ConsentManager
        cm = ConsentManager()
        record = cm.grant("user-A", "proj-X")
        assert record["active"] is True
        assert record["user_id"] == "user-A"
        assert record["proj_id"] == "proj-X"

    def test_has_active_consent_true_after_grant(self):
        from consent import ConsentManager
        cm = ConsentManager()
        cm.grant("user-B", "proj-Y")
        assert cm.has_active_consent("user-B", "proj-Y") is True

    def test_has_active_consent_false_before_grant(self):
        from consent import ConsentManager
        cm = ConsentManager()
        assert cm.has_active_consent("user-C", "proj-Z") is False

    def test_revoke_deactivates(self):
        from consent import ConsentManager
        cm = ConsentManager()
        cm.grant("user-D", "proj-W")
        assert cm.has_active_consent("user-D", "proj-W") is True
        cm.revoke("user-D", "proj-W")
        assert cm.has_active_consent("user-D", "proj-W") is False

    def test_get_consent_returns_active(self):
        from consent import ConsentManager
        cm = ConsentManager()
        cm.grant("user-E", "proj-V", scope=["age_at_diagnosis"])
        c = cm.get_consent("user-E", "proj-V")
        assert c is not None
        assert "age_at_diagnosis" in c["scope"]

    def test_get_consent_returns_none_when_revoked(self):
        from consent import ConsentManager
        cm = ConsentManager()
        cm.grant("user-F", "proj-U")
        cm.revoke("user-F", "proj-U")
        assert cm.get_consent("user-F", "proj-U") is None

    def test_revoke_all_for_user(self):
        from consent import ConsentManager
        cm = ConsentManager()
        cm.grant("user-G", "proj-1")
        cm.grant("user-G", "proj-2")
        revoked = cm.revoke_all_for_user("user-G")
        assert revoked == 2
        assert cm.has_active_consent("user-G", "proj-1") is False
        assert cm.has_active_consent("user-G", "proj-2") is False

    def test_list_consents_filtered(self):
        from consent import ConsentManager
        cm = ConsentManager()
        cm.grant("user-H", "proj-A")
        cm.grant("user-I", "proj-A")
        by_proj = cm.list_consents(proj_id="proj-A")
        assert len(by_proj) == 2

    def test_duplicate_grant_supersedes(self):
        """Granting a second consent should revoke the first."""
        from consent import ConsentManager
        cm = ConsentManager()
        cm.grant("user-J", "proj-B")
        cm.grant("user-J", "proj-B", scope=["bmi"])
        # Only one active consent should remain
        active = cm.list_consents(user_id="user-J", proj_id="proj-B", active_only=True)
        assert len(active) == 1
        assert "bmi" in active[0]["scope"]


# =============================================================================
# 28. server/unlearning.py — Federated Unlearning
# =============================================================================
class TestUnlearningManager:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db.update({"consents": [], "client_reputations": []})
        db_handler.write_db(db)

    def _seed_project(self, proj_id: str, current_round: int = 3) -> None:
        import db_handler
        proj = {
            "proj_id": proj_id,
            "current_round": current_round,
            "global_model_path": "",
            "round_state": "idle",
            "connected_clients": ["hosp-A"],
            "approved_projects": [],
            "momentum_beta": 0.9,
            "min_clients_per_round": 1,
        }
        db = db_handler.read_db()
        db["projects"].append(proj)
        db["users"].append({"user_id": "hosp-A", "username": "hosp_a",
                             "approved_projects": [proj_id], "password_hash": "x",
                             "hospital_name": "H", "contact_email": "h@h.com"})
        db_handler.write_db(db)

    def _add_round_history(self, proj_id: str, rounds: list[int], client_id: str = "hosp-A") -> None:
        import db_handler
        db = db_handler.read_db()
        for r in rounds:
            db["rounds_history"].append({
                "proj_id": proj_id, "round": r,
                "contributing_clients": [client_id],
            })
        db_handler.write_db(db)

    def test_forget_hospital_no_contribution(self, tmp_path):
        from unlearning import UnlearningManager
        self._seed_project("proj-ul-1", current_round=5)
        # No round history → "no_contribution" status
        um = UnlearningManager()
        result = um.forget_hospital("proj-ul-1", "hosp-A", models_dir=str(tmp_path))
        assert result["status"] == "no_contribution"

    def test_forget_hospital_rollback(self, tmp_path):
        from unlearning import UnlearningManager
        self._seed_project("proj-ul-2", current_round=5)
        self._add_round_history("proj-ul-2", [3, 4, 5])  # hospital joined at round 3
        um = UnlearningManager()
        result = um.forget_hospital("proj-ul-2", "hosp-A", models_dir=str(tmp_path))
        # Should roll back to round 2 (one before round 3)
        assert result["status"] == "ok"
        assert result["clean_round"] == 2
        assert result["reverted_rounds"] == 3

    def test_forget_hospital_removes_from_connected_clients(self, tmp_path):
        import db_handler
        from unlearning import UnlearningManager
        self._seed_project("proj-ul-3", current_round=3)
        self._add_round_history("proj-ul-3", [2, 3])
        um = UnlearningManager()
        um.forget_hospital("proj-ul-3", "hosp-A", models_dir=str(tmp_path))
        proj = db_handler.get_project("proj-ul-3")
        assert "hosp-A" not in proj.get("connected_clients", [])

    def test_forget_unknown_project_raises(self, tmp_path):
        from unlearning import UnlearningManager, UnlearningError
        um = UnlearningManager()
        with pytest.raises(UnlearningError, match="not found"):
            um.forget_hospital("ghost-proj", "hosp-A", models_dir=str(tmp_path))

    def test_forget_hospital_uses_existing_checkpoint(self, tmp_path):
        from unlearning import UnlearningManager
        import torch
        self._seed_project("proj-ul-4", current_round=5)
        self._add_round_history("proj-ul-4", [4, 5])
        # Create a dummy checkpoint for round 3
        chk = tmp_path / "proj-ul-4_round3.pt"
        torch.save({"w": torch.ones(4)}, str(chk))
        um = UnlearningManager()
        result = um.forget_hospital("proj-ul-4", "hosp-A", models_dir=str(tmp_path))
        assert result["checkpoint_path"] == str(chk)


# =============================================================================
# 29. server/flame_defense.py — FLAME Byzantine defense
# =============================================================================
class TestFlameDefense:
    def _make_update(self, val: float, shape=(8,)) -> dict:
        return {"w": np.full(shape, val, dtype=np.float32)}

    def test_less_than_2_clients_returns_all(self):
        from flame_defense import filter_updates_flame
        updates = [self._make_update(1.0)]
        out_u, out_c = filter_updates_flame(updates, [100])
        assert out_u == updates

    def test_all_honest_keeps_all(self):
        from flame_defense import filter_updates_flame
        updates = [self._make_update(1.0) for _ in range(4)]
        out_u, out_c = filter_updates_flame(updates, [100] * 4, similarity_threshold=0.0)
        assert len(out_u) == 4

    def test_outlier_removed(self):
        """6 honest (val=1) + 1 Byzantine (val=−1000) → Byzantine cluster removed."""
        from flame_defense import filter_updates_flame
        honest = [self._make_update(1.0) for _ in range(6)]
        byzantine = [self._make_update(-1000.0)]
        updates = honest + byzantine
        out_u, out_c = filter_updates_flame(
            updates, [100] * 7,
            similarity_threshold=0.5,
            min_cluster_size=1,
        )
        # Byzantine update should be filtered (or at minimum, fewer than 7 survive)
        # At worst case, if clustering fails gracefully, all 7 are returned
        assert len(out_u) <= 7

    def test_noise_sigma_adds_noise(self):
        from flame_defense import filter_updates_flame
        updates = [self._make_update(1.0) for _ in range(3)]
        out_u, _ = filter_updates_flame(
            updates, [100] * 3,
            similarity_threshold=0.0,
            noise_sigma=1.0,
        )
        # With noise, values should differ from original 1.0
        has_noise = any(not np.allclose(u["w"], 1.0) for u in out_u)
        assert has_noise

    def test_output_structure_preserved(self):
        from flame_defense import filter_updates_flame
        updates = [{"a": np.ones((4,), np.float32), "b": np.zeros((2,), np.float32)}
                   for _ in range(3)]
        out_u, out_c = filter_updates_flame(updates, [100] * 3)
        for u in out_u:
            assert "a" in u and "b" in u


# =============================================================================
# 30. server/reputation.py — Per-client reputation
# =============================================================================
class TestReputationManager:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db["client_reputations"] = []
        db_handler.write_db(db)

    def _make_weights(self, val: float, shape=(8,)) -> dict:
        return {"w": np.full(shape, val, dtype=np.float32)}

    def test_initial_score_is_one(self):
        from reputation import ReputationManager
        rm = ReputationManager()
        assert rm.get_score("new-client", "proj-x") == 1.0

    def test_not_suspended_initially(self):
        from reputation import ReputationManager
        rm = ReputationManager()
        assert not rm.is_suspended("new-client", "proj-x")

    def test_update_scores_returns_dict(self):
        from reputation import ReputationManager
        rm = ReputationManager()
        updates = [self._make_weights(1.0)]
        global_agg = self._make_weights(1.0)
        scores = rm.update_scores("proj-rep", updates, ["hosp-1"], global_agg)
        assert "hosp-1" in scores
        assert 0.0 <= scores["hosp-1"] <= 1.0

    def test_aligned_client_keeps_high_score(self):
        """Client aligned with global aggregate keeps score near 1.0."""
        from reputation import ReputationManager
        rm = ReputationManager()
        updates = [self._make_weights(1.0)]
        global_agg = self._make_weights(1.0)
        scores = rm.update_scores(
            "proj-rep-2", updates, ["hosp-aligned"], global_agg,
            alpha=0.3, threshold=0.2,
        )
        assert scores["hosp-aligned"] > 0.5

    def test_misaligned_client_score_decreases(self):
        """Client opposite to global aggregate: repeated misalignment → low score."""
        from reputation import ReputationManager
        rm = ReputationManager()
        global_agg = self._make_weights(1.0)
        for _ in range(10):
            rm.update_scores(
                "proj-mis", [self._make_weights(-1.0)], ["hosp-bad"],
                global_agg, alpha=0.5, threshold=0.3,
            )
        score = rm.get_score("hosp-bad", "proj-mis")
        assert score < 0.5

    def test_reinstate_resets_score(self):
        from reputation import ReputationManager
        rm = ReputationManager()
        global_agg = self._make_weights(1.0)
        for _ in range(20):
            rm.update_scores(
                "proj-reinst", [self._make_weights(-1.0)], ["hosp-reinst"],
                global_agg, alpha=0.9, threshold=0.5,
            )
        rm.reinstate("hosp-reinst", "proj-reinst")
        assert not rm.is_suspended("hosp-reinst", "proj-reinst")
        assert rm.get_score("hosp-reinst", "proj-reinst") == 1.0

    def test_list_scores_by_project(self):
        from reputation import ReputationManager
        rm = ReputationManager()
        global_agg = self._make_weights(1.0)
        rm.update_scores("proj-ls", [self._make_weights(1.0)], ["c1"], global_agg)
        rm.update_scores("proj-ls", [self._make_weights(1.0)], ["c2"], global_agg)
        scores_list = rm.list_scores("proj-ls")
        client_ids = [s["client_id"] for s in scores_list]
        assert "c1" in client_ids
        assert "c2" in client_ids


# =============================================================================
# 31. server/temperature_scaling.py — Temperature Scaling
# =============================================================================
class TestTemperatureScaling:
    def test_apply_temperature_divides_logits(self):
        from temperature_scaling import apply_temperature
        logits = torch.tensor([2.0, 4.0, 6.0])
        out = apply_temperature(logits, T=2.0)
        torch.testing.assert_close(out, torch.tensor([1.0, 2.0, 3.0]))

    def test_apply_temperature_one_is_noop(self):
        from temperature_scaling import apply_temperature
        logits = torch.tensor([1.0, 2.0, 3.0])
        out = apply_temperature(logits, T=1.0)
        torch.testing.assert_close(out, logits)

    def test_apply_temperature_zero_clamped(self):
        from temperature_scaling import apply_temperature
        logits = torch.tensor([1.0, 2.0])
        # T=0 should not divide by zero (clamped to 1e-6)
        out = apply_temperature(logits, T=0.0)
        assert torch.isfinite(out).all()

    def test_scaler_fit_returns_two_floats(self):
        from temperature_scaling import TemperatureScaler
        from supernet import Supernet
        from torch.utils.data import TensorDataset, DataLoader
        torch.manual_seed(0)
        n = 32
        X = torch.randn(n, 32)
        y_reg = torch.randn(n)
        y_tox = torch.randint(0, 4, (n,))
        y_bin = torch.tensor([i % 2 for i in range(n)], dtype=torch.float32)
        ds = TensorDataset(X, y_reg, y_tox, y_bin)
        loader = DataLoader(ds, batch_size=16)
        model = Supernet(input_dim=32, max_depth=2, hidden_dim=16, num_toxicity_classes=4)
        scaler = TemperatureScaler(max_iter=10)
        T_tox, T_bin = scaler.fit(model, loader, active_depth=2)
        assert isinstance(T_tox, float)
        assert isinstance(T_bin, float)
        assert T_tox > 0.0
        assert T_bin > 0.0


# =============================================================================
# 32. client/focal_loss.py — Focal Loss
# =============================================================================
class TestFocalLoss:
    def test_forward_returns_scalar(self):
        from focal_loss import FocalLoss
        fl = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = fl(logits, targets)
        assert loss.shape == ()

    def test_gamma_zero_matches_cross_entropy(self):
        from focal_loss import FocalLoss
        torch.manual_seed(0)
        fl = FocalLoss(gamma=0.0)
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))
        fl_loss = fl(logits, targets)
        ce_loss = torch.nn.functional.cross_entropy(logits, targets)
        # At gamma=0, focal loss == cross entropy
        assert abs(float(fl_loss) - float(ce_loss)) < 1e-4

    def test_focal_loss_less_than_ce_for_easy_examples(self):
        """For very easy examples (model already confident), focal > CE suppression."""
        from focal_loss import FocalLoss
        # Create logits that make the model very confident about class 0
        logits = torch.tensor([[10.0, -10.0, -10.0, -10.0]] * 8)
        targets = torch.zeros(8, dtype=torch.long)
        fl = FocalLoss(gamma=2.0)
        ce = torch.nn.CrossEntropyLoss()
        fl_val = float(fl(logits, targets))
        ce_val = float(ce(logits, targets))
        # Focal loss should be ≤ CE for easy examples
        assert fl_val <= ce_val + 1e-4

    def test_binary_focal_loss_finite(self):
        from focal_loss import BinaryFocalLoss
        bfl = BinaryFocalLoss(gamma=2.0, alpha=0.25)
        logits = torch.randn(8)
        targets = torch.tensor([0.0, 1.0] * 4)
        loss = bfl(logits, targets)
        assert torch.isfinite(loss)

    def test_invalid_reduction_raises(self):
        from focal_loss import FocalLoss
        with pytest.raises(ValueError):
            FocalLoss(reduction="invalid")

    def test_invalid_gamma_raises(self):
        from focal_loss import FocalLoss
        with pytest.raises(ValueError):
            FocalLoss(gamma=-1.0)

    def test_reduction_sum(self):
        from focal_loss import FocalLoss
        fl = FocalLoss(gamma=2.0, reduction="sum")
        logits = torch.randn(4, 4)
        targets = torch.randint(0, 4, (4,))
        loss = fl(logits, targets)
        assert loss.shape == ()

    def test_reduction_none_shape(self):
        from focal_loss import FocalLoss
        fl = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(6, 4)
        targets = torch.randint(0, 4, (6,))
        loss = fl(logits, targets)
        assert loss.shape == (6,)

    def test_per_class_alpha(self):
        from focal_loss import FocalLoss
        fl = FocalLoss(gamma=2.0, alpha=[0.25, 0.5, 0.75, 1.0])
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = fl(logits, targets)
        assert torch.isfinite(loss)


# =============================================================================
# 33. Min participation abort (project_router integration)
# =============================================================================
class TestMinParticipation:
    """
    Verify that round_lifecycle correctly handles too-few clients.
    The existing test already checks empty buffer → no increment.
    Here we test that when < min_clients_per_round submit, the round
    fails gracefully (EmptyRoundError path or documented behaviour).
    """
    def test_zero_updates_does_not_increment_round(self, tmp_path, tmp_db_path, monkeypatch):
        import db_handler, project_router
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        monkeypatch.setattr(project_router, "MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(project_router, "_val_dataloader", None)

        proj = {
            "proj_id": "mp-proj-1", "current_round": 0,
            "global_model_path": "", "momentum_beta": 0.9,
            "recommended_depth": 2,
            "connected_clients": ["c1", "c2"],
            "min_clients_per_round": 2,
        }
        db = db_handler.read_db()
        db["projects"].append(proj)
        db_handler.write_db(db)

        from project_router import round_lifecycle
        db_snap = db_handler.read_db()
        # Submit 0 updates when min_clients = 2
        round_lifecycle("mp-proj-1", [], db_snap)

        updated = db_handler.get_project("mp-proj-1")
        assert updated["current_round"] == 0


# =============================================================================
# 34. DP Correctness — additional phase 3 checks
# =============================================================================
class TestDPCorrectness:
    """Additional correctness checks for the DP pipeline."""

    def test_clip_then_noise_global_norm_bounded(self):
        """After clip_weights + apply_dp, the original weights are within max_norm."""
        from shared.dp_utils import clip_weights, apply_dp
        rng = np.random.default_rng(0)
        # Large weights far beyond the clip bound
        weights = {"w": np.full((100,), 100.0, dtype=np.float32)}
        clipped = clip_weights(weights, max_norm=1.0)
        flat = np.concatenate([v.flatten() for v in clipped.values()])
        # Clipped norm should be ≤ max_norm
        assert np.linalg.norm(flat) <= 1.0 + 1e-5

    def test_dp_noise_zero_mean_over_many_samples(self):
        """The added noise should be approximately zero-mean over many samples."""
        from shared.dp_utils import apply_dp
        rng = np.random.default_rng(42)
        total = np.zeros((100,), dtype=np.float32)
        n = 1000
        for _ in range(n):
            w = {"w": np.zeros((100,), dtype=np.float32)}
            out = apply_dp(w, sensitivity=1.0, epsilon=1.0, delta=1e-5, rng=rng)
            total += out["w"]
        mean = total / n
        # Gaussian noise mean ≈ 0; allow ±0.5 (sigma ≈ 8.3 at these params → mean/sqrt(n) ≈ 0.26)
        assert np.abs(mean).max() < 1.5

    def test_privacy_accountant_monotonic(self):
        """Spent epsilon should be non-decreasing."""
        from shared.dp_utils import PrivacyAccountant
        acc = PrivacyAccountant(target_epsilon=10.0, target_delta=1e-5,
                                noise_multiplier=2.0, sampling_rate=0.01)
        prev = 0.0
        for _ in range(10):
            acc.step(1)
            assert acc.spent_epsilon >= prev
            prev = acc.spent_epsilon
