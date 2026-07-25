"""
tests/test_phase3_remaining.py
Tests for the remaining Phase 3 modules:
  - server/schema_enforcement.py   (3.5)
  - server/data_residency.py       (3.8)
  - server/baa.py                  (3.9)
  - server/nas_profiler.py         (3.13 — also covered in test_integration.py)
  - server/milestone_eval.py       (3.16)
  - server/reconnection.py         (3.19)
"""
import os
import json
import time

import numpy as np
import pytest


# =============================================================================
# 3.5 — server/schema_enforcement.py
# =============================================================================
class TestSchemaEnforcer:
    def _make_enforcer(self, required=None, strict=False):
        from schema_enforcement import SchemaEnforcer
        required = required or ["age_at_diagnosis", "tumor_stage", "gene_expr"]
        return SchemaEnforcer(required_columns=required, strict=strict)

    def test_empty_required_raises(self):
        from schema_enforcement import SchemaEnforcer
        with pytest.raises(ValueError):
            SchemaEnforcer(required_columns=[])

    def test_validate_passes_with_all_columns(self):
        se = self._make_enforcer()
        se.validate(["age_at_diagnosis", "tumor_stage", "gene_expr", "extra_col"])

    def test_validate_raises_on_missing_column(self):
        from schema_enforcement import SchemaViolationError
        se = self._make_enforcer()
        with pytest.raises(SchemaViolationError, match="missing required columns"):
            se.validate(["age_at_diagnosis", "tumor_stage"])  # gene_expr missing

    def test_strict_mode_raises_on_extra_columns(self):
        from schema_enforcement import SchemaViolationError
        se = self._make_enforcer(strict=True)
        with pytest.raises(SchemaViolationError, match="disallowed columns"):
            se.validate(["age_at_diagnosis", "tumor_stage", "gene_expr", "extra"])

    def test_non_strict_mode_allows_extra_columns(self):
        se = self._make_enforcer(strict=False)
        se.validate(["age_at_diagnosis", "tumor_stage", "gene_expr", "extra"])

    def test_is_valid_true(self):
        se = self._make_enforcer()
        assert se.is_valid(["age_at_diagnosis", "tumor_stage", "gene_expr"]) is True

    def test_is_valid_false_on_missing(self):
        se = self._make_enforcer()
        assert se.is_valid(["age_at_diagnosis"]) is False

    def test_allowed_columns_returns_intersection(self):
        se = self._make_enforcer()
        allowed = se.allowed_columns(["age_at_diagnosis", "extra", "gene_expr"])
        assert "age_at_diagnosis" in allowed
        assert "gene_expr" in allowed
        assert "extra" not in allowed

    def test_build_enforcer_from_project(self):
        from schema_enforcement import build_enforcer_from_project, SchemaEnforcer
        proj = {"data_schema": ["col_a", "col_b"]}
        se = build_enforcer_from_project(proj)
        assert isinstance(se, SchemaEnforcer)
        se.validate(["col_a", "col_b"])

    def test_build_enforcer_uses_server_schema_as_fallback(self):
        from schema_enforcement import build_enforcer_from_project
        se = build_enforcer_from_project({})   # no data_schema key
        assert len(se.required) > 0


# =============================================================================
# 3.8 — server/data_residency.py
# =============================================================================
class TestResidencyManager:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db["rounds_history"] = []
        db_handler.write_db(db)

    def _weights(self, val=1.0):
        return {"w": np.full((4,), val, dtype=np.float32)}

    def test_record_arrival_returns_hash(self):
        from data_residency import ResidencyManager
        rm = ResidencyManager()
        h = rm.record_arrival("proj-r1", "hosp-1", self._weights())
        assert isinstance(h, str)
        assert len(h) == 64   # SHA-256 hex

    def test_pending_count_increments(self):
        from data_residency import ResidencyManager
        rm = ResidencyManager()
        rm.record_arrival("proj-r2", "hosp-a", self._weights())
        rm.record_arrival("proj-r2", "hosp-b", self._weights())
        assert rm.pending_count("proj-r2") == 2

    def test_purge_round_clears_records(self):
        from data_residency import ResidencyManager
        rm = ResidencyManager()
        rm.record_arrival("proj-r3", "hosp-x", self._weights())
        n = rm.purge_round("proj-r3")
        assert n == 1
        assert rm.pending_count("proj-r3") == 0

    def test_find_stale_returns_expired(self):
        from data_residency import ResidencyManager
        rm = ResidencyManager(ttl_seconds=0)   # expire immediately
        rm.record_arrival("proj-stale", "hosp-s", self._weights())
        time.sleep(0.01)
        stale = rm.find_stale()
        assert "proj-stale" in stale
        assert "hosp-s" in stale["proj-stale"]

    def test_evict_stale_removes_and_triggers_callback(self):
        from data_residency import ResidencyManager
        evicted_ids = []
        def callback(proj_id, user_ids):
            evicted_ids.extend(user_ids)

        rm = ResidencyManager(ttl_seconds=0, purge_callback=callback)
        rm.record_arrival("proj-ev", "hosp-ev", self._weights())
        time.sleep(0.01)
        n = rm.evict_stale()
        assert n == 1
        assert "hosp-ev" in evicted_ids

    def test_non_stale_not_evicted(self):
        from data_residency import ResidencyManager
        rm = ResidencyManager(ttl_seconds=9999)
        rm.record_arrival("proj-fresh", "hosp-f", self._weights())
        n = rm.evict_stale()
        assert n == 0
        assert rm.pending_count("proj-fresh") == 1

    def test_weight_hash_deterministic(self):
        from data_residency import _weight_hash
        w = {"a": np.ones((4,), dtype=np.float32)}
        h1 = _weight_hash(w)
        h2 = _weight_hash(w)
        assert h1 == h2

    def test_purge_round_audit_record_written(self):
        import db_handler
        from data_residency import ResidencyManager
        rm = ResidencyManager()
        rm.record_arrival("proj-audit", "hosp-au", self._weights())
        rm.purge_round("proj-audit")
        db = db_handler.read_db()
        events = [r for r in db["rounds_history"] if r.get("event") == "residency_purge"]
        assert len(events) >= 1


# =============================================================================
# 3.9 — server/baa.py
# =============================================================================
class TestBAAManager:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db["baa_records"] = []
        db["users"].append({
            "user_id": "baa-user-1", "username": "baa1",
            "password_hash": "x", "hospital_name": "H",
            "contact_email": "b@b.com", "approved_projects": [],
            "pending_projects": [], "baa_signed": False,
        })
        db_handler.write_db(db)

    def test_sign_creates_record(self):
        from baa import BAAManager
        bm = BAAManager()
        record = bm.sign("baa-user-1", signed_by="admin")
        assert record["user_id"] == "baa-user-1"
        assert record["active"] is True

    def test_has_signed_true_after_sign(self):
        from baa import BAAManager
        bm = BAAManager()
        bm.sign("baa-user-1", signed_by="admin")
        assert bm.has_signed("baa-user-1") is True

    def test_has_signed_false_before_sign(self):
        from baa import BAAManager
        bm = BAAManager()
        assert bm.has_signed("no-such-user") is False

    def test_revoke_deactivates(self):
        from baa import BAAManager
        bm = BAAManager()
        bm.sign("baa-user-1", signed_by="admin")
        n = bm.revoke("baa-user-1")
        assert n == 1
        assert bm.has_signed("baa-user-1") is False

    def test_check_or_raise_passes_when_signed(self):
        from baa import BAAManager
        bm = BAAManager()
        bm.sign("baa-user-1", signed_by="admin")
        bm.check_or_raise("baa-user-1")  # should not raise

    def test_check_or_raise_raises_when_not_signed(self):
        from baa import BAAManager, BAAViolationError
        bm = BAAManager()
        with pytest.raises(BAAViolationError):
            bm.check_or_raise("baa-user-1")

    def test_get_record_returns_active(self):
        from baa import BAAManager
        bm = BAAManager()
        bm.sign("baa-user-1", signed_by="admin")
        r = bm.get_record("baa-user-1")
        assert r is not None
        assert r["signed_by"] == "admin"

    def test_get_record_none_when_revoked(self):
        from baa import BAAManager
        bm = BAAManager()
        bm.sign("baa-user-1", signed_by="admin")
        bm.revoke("baa-user-1")
        assert bm.get_record("baa-user-1") is None

    def test_list_records_active_only(self):
        from baa import BAAManager
        bm = BAAManager()
        bm.sign("baa-user-1", signed_by="admin")
        records = bm.list_records(active_only=True)
        assert len(records) >= 1
        assert all(r["active"] for r in records)

    def test_sign_sets_baa_flag_on_user(self):
        import db_handler
        from baa import BAAManager
        bm = BAAManager()
        bm.sign("baa-user-1", signed_by="admin")
        db = db_handler.read_db()
        user = next(u for u in db["users"] if u["user_id"] == "baa-user-1")
        assert user["baa_signed"] is True


# =============================================================================
# 3.16 — server/milestone_eval.py
# =============================================================================
class TestMilestoneEvaluator:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db["rounds_history"] = []
        db_handler.write_db(db)

    def test_evaluate_without_loader_returns_none_metrics(self):
        from milestone_eval import MilestoneEvaluator
        me = MilestoneEvaluator()
        result = me.evaluate({}, None, {}, round_num=1)
        assert result["val_rmse"] is None
        assert result["milestone_reached"] is False

    def test_empty_result_structure(self):
        from milestone_eval import MilestoneEvaluator
        me = MilestoneEvaluator()
        result = me._empty_result(5)
        assert result["round"] == 5
        assert result["milestone_reached"] is False
        assert "val_rmse" in result

    def test_milestone_reached_when_all_thresholds_met(self):
        """Simulate a metrics dict that meets all thresholds."""
        from milestone_eval import MilestoneEvaluator

        class MockLoader:
            pass

        class MockValidation:
            @staticmethod
            def validate_global_model(weights, loader, config):
                return {"val_rmse": 0.5, "val_acc_tox": 0.95, "val_auc": 0.90}

        me = MilestoneEvaluator(
            rmse_threshold=0.85,
            tox_acc_threshold=0.80,
            auc_threshold=0.75,
        )
        # Patch validate_global_model
        import aggregation
        orig = getattr(aggregation, "validate_global_model", None)
        aggregation.validate_global_model = MockValidation.validate_global_model
        try:
            result = me.evaluate({}, MockLoader(), {}, round_num=3)
        finally:
            if orig is not None:
                aggregation.validate_global_model = orig
        assert result["milestone_reached"] is True

    def test_milestone_not_reached_when_below_threshold(self):
        from milestone_eval import MilestoneEvaluator
        import aggregation

        class MockLoader: pass

        def mock_validate(weights, loader, config):
            return {"val_rmse": 2.0, "val_acc_tox": 0.3, "val_auc": 0.2}

        me = MilestoneEvaluator()
        orig = getattr(aggregation, "validate_global_model", None)
        aggregation.validate_global_model = mock_validate
        try:
            result = me.evaluate({}, MockLoader(), {}, round_num=2)
        finally:
            if orig is not None:
                aggregation.validate_global_model = orig
        assert result["milestone_reached"] is False


# =============================================================================
# 3.19 — server/reconnection.py
# =============================================================================
class TestReconnectionManager:
    @pytest.fixture(autouse=True)
    def patch_db(self, tmp_db_path, monkeypatch):
        import db_handler
        monkeypatch.setattr(db_handler, "DB_PATH", tmp_db_path)
        db = db_handler.read_db()
        db["client_round_states"] = []
        db["projects"].append({
            "proj_id": "proj-rc",
            "current_round": 3,
            "global_model_path": "",
            "round_state": "idle",
            "connected_clients": ["hosp-rc"],
            "min_clients_per_round": 1,
            "momentum_beta": 0.9,
        })
        db_handler.write_db(db)

    def test_mark_submitted_sets_state(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        rm.mark_submitted("proj-rc", "hosp-rc", 3)
        assert rm.get_state("proj-rc", "hosp-rc", 3) == "submitted"

    def test_mark_pending_sets_state(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        rm.mark_pending("proj-rc", "hosp-rc", 3)
        assert rm.get_state("proj-rc", "hosp-rc", 3) == "pending"

    def test_close_round_marks_non_submitters_as_missed(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        rm.mark_pending("proj-rc", "hosp-a", 3)
        rm.mark_pending("proj-rc", "hosp-b", 3)
        missed = rm.close_round("proj-rc", 3, submitted_clients=["hosp-a"])
        assert "hosp-b" in missed
        assert "hosp-a" not in missed

    def test_get_state_unknown_for_new_client(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        assert rm.get_state("proj-rc", "new-hosp") == "unknown"

    def test_reconnect_catch_up_after_miss(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        rm.mark_pending("proj-rc", "hosp-rc", 2)
        rm.close_round("proj-rc", 2, submitted_clients=[])
        ctx = rm.reconnect("proj-rc", "hosp-rc")
        assert ctx["action"] == "catch_up"

    def test_reconnect_submit_when_pending(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        rm.mark_pending("proj-rc", "hosp-rc", 3)
        ctx = rm.reconnect("proj-rc", "hosp-rc")
        assert ctx["action"] == "submit"

    def test_reconnect_wait_when_submitted(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        rm.mark_submitted("proj-rc", "hosp-rc", 3)
        ctx = rm.reconnect("proj-rc", "hosp-rc")
        assert ctx["action"] == "wait"

    def test_reconnect_unknown_project_returns_unknown(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        ctx = rm.reconnect("no-such-project", "hosp-rc")
        assert ctx["action"] == "unknown"

    def test_list_missed_returns_only_missed(self):
        from reconnection import ReconnectionManager
        rm = ReconnectionManager()
        rm.mark_pending("proj-rc", "hosp-m1", 3)
        rm.mark_pending("proj-rc", "hosp-m2", 3)
        rm.close_round("proj-rc", 3, submitted_clients=[])
        missed = rm.list_missed("proj-rc")
        user_ids = [m["user_id"] for m in missed]
        assert "hosp-m1" in user_ids
        assert "hosp-m2" in user_ids
