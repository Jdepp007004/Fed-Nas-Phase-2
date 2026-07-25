"""
server/milestone_eval.py
Milestone evaluation on the held-out global test set (Phase 3 — M10).

After each round, if a server-side held-out test DataLoader is available,
the global model is evaluated on it and results are stored in round_history.

Milestone thresholds (configurable via env vars):
  MILESTONE_RMSE_THRESHOLD  — target val_rmse (default 0.85)
  MILESTONE_TOX_ACC         — target val_acc_tox (default 0.80)
  MILESTONE_AUC             — target val_auc (default 0.75)

When all thresholds are reached simultaneously, a "milestone_reached" event
is appended to rounds_history and logged at WARNING level.

Public API
----------
    from milestone_eval import MilestoneEvaluator

    evaluator = MilestoneEvaluator()
    result = evaluator.evaluate(global_weights, test_loader, model_config, round_num)
    # result: {"val_rmse": float, "val_acc_tox": float, "val_auc": float,
    #          "milestone_reached": bool}
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import numpy as np
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

# Milestone thresholds
RMSE_THRESHOLD = float(os.getenv("MILESTONE_RMSE_THRESHOLD", "0.85"))
TOX_ACC_THRESHOLD = float(os.getenv("MILESTONE_TOX_ACC", "0.80"))
AUC_THRESHOLD = float(os.getenv("MILESTONE_AUC", "0.75"))


class MilestoneEvaluator:
    """
    Evaluates the global model on the held-out test set and checks milestones.

    Uses the same `validate_global_model` function as `project_router.py`,
    then compares results against milestone thresholds.
    """

    def __init__(
        self,
        rmse_threshold: float = RMSE_THRESHOLD,
        tox_acc_threshold: float = TOX_ACC_THRESHOLD,
        auc_threshold: float = AUC_THRESHOLD,
    ) -> None:
        self.rmse_threshold = rmse_threshold
        self.tox_acc_threshold = tox_acc_threshold
        self.auc_threshold = auc_threshold

    def evaluate(
        self,
        global_weights: dict,
        test_loader,
        model_config: dict,
        round_num: int,
        proj_id: Optional[str] = None,
    ) -> dict:
        """
        Run evaluation on the test set and check milestone thresholds.

        Parameters
        ----------
        global_weights : dict — {param_name: np.ndarray} global model state
        test_loader    : DataLoader — server-side held-out test set
        model_config   : dict — MODEL_CONFIG from shared.model_schema
        round_num      : int — current round number (for logging)
        proj_id        : str | None — project UUID (for audit trail)

        Returns
        -------
        dict with keys:
            val_rmse          : float
            val_acc_tox       : float
            val_auc           : float
            milestone_reached : bool
            round             : int
        """
        if not _TORCH_AVAILABLE:
            logger.warning("torch not available — skipping milestone evaluation")
            return self._empty_result(round_num)

        if test_loader is None:
            logger.info("No test_loader provided — skipping milestone evaluation")
            return self._empty_result(round_num)

        try:
            from aggregation import validate_global_model
            metrics = validate_global_model(global_weights, test_loader, model_config)
        except Exception as e:
            logger.warning("Milestone evaluation failed: %s", e)
            return self._empty_result(round_num)

        val_rmse    = float(metrics.get("val_rmse", float("inf")))
        val_acc_tox = float(metrics.get("val_acc_tox", 0.0))
        val_auc     = float(metrics.get("val_auc", 0.0))

        milestone_reached = (
            val_rmse    <= self.rmse_threshold
            and val_acc_tox >= self.tox_acc_threshold
            and val_auc     >= self.auc_threshold
        )

        result = {
            "val_rmse":          val_rmse,
            "val_acc_tox":       val_acc_tox,
            "val_auc":           val_auc,
            "milestone_reached": milestone_reached,
            "round":             round_num,
        }

        logger.info(
            "Milestone eval round=%d: rmse=%.4f tox_acc=%.4f auc=%.4f milestone=%s",
            round_num, val_rmse, val_acc_tox, val_auc, milestone_reached,
        )

        if milestone_reached:
            logger.warning(
                "🎯 MILESTONE REACHED at round %d! rmse=%.4f tox_acc=%.4f auc=%.4f",
                round_num, val_rmse, val_acc_tox, val_auc,
            )
            if proj_id:
                self._record_milestone(proj_id, round_num, result)

        return result

    def _record_milestone(self, proj_id: str, round_num: int, metrics: dict) -> None:
        """Append milestone_reached event to rounds_history."""
        try:
            from db_handler import append_round_history
            append_round_history({
                "proj_id":  proj_id,
                "round":    round_num,
                "event":    "milestone_reached",
                "timestamp": _utcnow(),
                **metrics,
            })
        except Exception as e:
            logger.warning("Could not record milestone event: %s", e)

    def _empty_result(self, round_num: int) -> dict:
        return {
            "val_rmse":          None,
            "val_acc_tox":       None,
            "val_auc":           None,
            "milestone_reached": False,
            "round":             round_num,
        }


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
