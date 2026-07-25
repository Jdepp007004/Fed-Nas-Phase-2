"""Federated learning live-session backend.

Runs the same client training loop and server aggregation code as a real
run.  State is kept in-memory only — no production database writes.

Exposes two route prefixes:
  /api/sim/*   — used by simulation.html
  /api/demo/*  — kept for backward compatibility
"""
from __future__ import annotations

import copy
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[1]
CLIENT_DIR = ROOT / "client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))

from aggregation import aggregate_fedavg, update_with_momentum, validate_global_model
from data_loader import build_dataloaders_from_csv
from supernet import Supernet, load_global_weights
from train_loop import TrainConfig, run_local_training
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA

# ── Two routers: /api/sim (primary) and /api/demo (compat alias) ──────────────
sim_router  = APIRouter(prefix="/api/sim",  tags=["simulation"])
demo_router = APIRouter(prefix="/api/demo", tags=["demo-compat"])

SESSION_CLIENTS = (
    ("Oncology North", "client_1.csv", "8-core CPU"),
    ("Oncology South", "client_2.csv", "6-core CPU"),
    ("Oncology East",  "client_3.csv", "GPU workstation"),
    ("Oncology West",  "client_4.csv", "4-core CPU"),
)
MAX_ROWS_PER_CLIENT = 256
DEFAULT_ROUNDS = 6

_lock = threading.Lock()
_state: dict = {"status": "idle", "clients": [], "rounds": [], "error": None}


def _loader_for_session(path: Path):
    """Load a bounded, deterministic slice so the session runs quickly."""
    import pandas as pd
    import tempfile

    df = pd.read_csv(path, nrows=MAX_ROWS_PER_CLIENT)
    descriptor, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(descriptor)
    df.to_csv(temp_path, index=False)
    try:
        return build_dataloaders_from_csv(temp_path, SERVER_SCHEMA)
    finally:
        os.unlink(temp_path)


def _set_state(**updates) -> None:
    with _lock:
        _state.update(updates)


def _run_session(round_count: int) -> None:
    try:
        client_loaders, validation_sets = [], []
        clients = []
        for index, (name, filename, hardware) in enumerate(SESSION_CLIENTS, start=1):
            train_loader, val_loader = _loader_for_session(ROOT / "data" / filename)
            client_loaders.append(train_loader)
            validation_sets.append(val_loader.dataset)
            clients.append({
                "id": f"device-{index}", "name": name, "hardware": hardware,
                "dataset": filename, "rows": MAX_ROWS_PER_CLIENT,
                "status": "joined",
            })
        _set_state(status="running", clients=clients, rounds=[], error=None)

        from torch.utils.data import ConcatDataset, DataLoader
        validation = DataLoader(ConcatDataset(validation_sets), batch_size=64)
        initial_model = Supernet(**MODEL_CONFIG)
        global_weights = {key: value.detach().cpu().numpy().copy()
                          for key, value in initial_model.state_dict().items()}
        velocity: dict = {}
        for round_number in range(1, round_count + 1):
            updates, sample_counts, client_metrics = [], [], []
            for client, loader in zip(clients, client_loaders):
                model = Supernet(**MODEL_CONFIG)
                if global_weights:
                    load_global_weights(model, global_weights, strict=False)
                result = run_local_training(
                    model, loader,
                    TrainConfig(epochs=1, active_depth=MODEL_CONFIG["max_depth"], fedprox_mu=0.01),
                )
                updates.append(result["weights"])
                sample_counts.append(result["num_samples"])
                client_metrics.append({"device": client["id"], **result["metrics"]})

            aggregate = aggregate_fedavg(updates, sample_counts)
            global_weights, velocity = update_with_momentum(global_weights, aggregate, 0.9, velocity)
            metrics = validate_global_model(global_weights, validation, MODEL_CONFIG)

            # NAS: evaluate which depth is most efficient this round
            updates_by_depth: dict = {}
            for client, result in zip(clients, [{"weights": upd, "num_samples": sc, "metrics": m}
                                                 for upd, sc, m in zip(updates, sample_counts, client_metrics)]):
                hw = client.get("hardware", "")
                # Assign depth based on hardware (matches nas_controller logic)
                if "GPU" in hw or "gpu" in hw:
                    d = 4
                elif "8-core" in hw:
                    d = 4
                elif "6-core" in hw:
                    d = 4
                else:
                    d = 3
                updates_by_depth.setdefault(d, []).append({"weights": updates[clients.index(client)], "num_samples": sample_counts[clients.index(client)]})

            try:
                from nas_controller import evaluate_architecture_candidates
                recommended_depth = evaluate_architecture_candidates(updates_by_depth, global_weights)
            except Exception:
                recommended_depth = MODEL_CONFIG["max_depth"]

            record = {
                "round": round_number,
                "participants": len(clients),
                "samples": sum(sample_counts),
                "client_metrics": client_metrics,
                "mean_local_loss": float(np.mean([m["loss"] for m in client_metrics])),
                "recommended_depth": recommended_depth,
                **metrics,
            }
            with _lock:
                _state["rounds"].append(record)

        _set_state(status="completed")
    except Exception as exc:
        _set_state(status="failed", error=str(exc))


# ── Shared handler functions ──────────────────────────────────────────────────

def _status_response():
    with _lock:
        return copy.deepcopy(_state)


def _start_response(rounds: int):
    if not 1 <= rounds <= 12:
        raise HTTPException(422, "rounds must be between 1 and 12")
    with _lock:
        if _state["status"] == "running":
            raise HTTPException(409, "A session is already running")
        _state.update({"status": "starting", "clients": [], "rounds": [], "error": None})
    threading.Thread(target=_run_session, args=(rounds,), daemon=True, name="fl-session").start()
    return {"status": "starting", "rounds": rounds, "devices": len(SESSION_CLIENTS)}


# ── /api/sim routes (primary, used by simulation.html) ───────────────────────

@sim_router.get("/status")
def sim_status():
    return _status_response()


@sim_router.post("/start")
def start_sim(rounds: int = DEFAULT_ROUNDS):
    return _start_response(rounds)


# ── /api/demo routes (backward-compat alias) ──────────────────────────────────

@demo_router.get("/status")
def demo_status():
    return _status_response()


@demo_router.post("/start")
def start_demo(rounds: int = DEFAULT_ROUNDS):
    return _start_response(rounds)

# Export both so main.py can include_router both
router = demo_router  # legacy alias so existing imports don't break
