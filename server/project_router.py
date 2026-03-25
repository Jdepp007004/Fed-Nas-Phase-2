"""
server/project_router.py
M3: /api/projects/* endpoints + background round lifecycle.
Owner: Sunishka Sarkar
"""

import os
import sys
import uuid
import datetime
import copy
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from fastapi import APIRouter, Depends, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db_handler import read_db, write_db, get_project, update_project, append_round_history
from auth_router import verify_jwt
from aggregation import aggregate_fedavg, update_with_momentum, validate_global_model, EmptyRoundError
from nas_controller import recommend_subnet_depth, evaluate_architecture_candidates
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA

# ── Path to models directory ──────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

router = APIRouter(prefix="/api/projects", tags=["projects"])

# ── In-memory buffers (keyed by proj_id) ────────────────────────────────────
_pending_updates: dict = {}   # proj_id → list of update dicts
_velocity_state:  dict = {}   # proj_id → velocity dict for momentum
_buffer_lock = threading.Lock()

# ── Server-side validation dataloader (created at startup by main.py) ────────
_val_dataloader = None

def set_val_dataloader(dl):
    global _val_dataloader
    _val_dataloader = dl


# ─── JWT Dependency ───────────────────────────────────────────────────────────

def _get_current_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise _http_error(401, "Missing or invalid Authorization header.")
    token = authorization.split(" ", 1)[1]
    try:
        return verify_jwt(token)
    except ValueError as e:
        raise _http_error(401, str(e))


def _http_error(status: int, detail: str):
    from fastapi import HTTPException
    return HTTPException(status_code=status, detail=detail)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class JoinRequest(BaseModel):
    hardware_profile: dict


class UpdateRequest(BaseModel):
    round_id:     int
    active_depth: int
    weights:      dict   # encrypted payload
    num_samples:  int
    metrics:      dict


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("")
async def list_projects(current_user: dict = Depends(_get_current_user)):
    """GET /api/projects"""
    db = read_db()
    user_id = current_user["sub"]
    user = next((u for u in db["users"] if u["user_id"] == user_id), None)
    approved = set(user.get("approved_projects", [])) if user else set()

    visible = []
    for proj in db.get("projects", []):
        if proj.get("accepting_clients") or proj["proj_id"] in approved:
            entry = {k: proj[k] for k in proj if k != "global_model_path"}
            entry["i_am_connected"] = user_id in proj.get("connected_clients", [])
            entry["i_am_pending"]   = user_id in proj.get("pending_clients", [])
            visible.append(entry)
    return JSONResponse(status_code=200, content=visible)


@router.post("/{proj_id}/join")
async def join_project(
    proj_id: str,
    payload: JoinRequest,
    current_user: dict = Depends(_get_current_user),
):
    """POST /api/projects/{proj_id}/join"""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")
    if not proj.get("accepting_clients", True):
        raise _http_error(403, "Project is not accepting new clients.")

    user_id = current_user["sub"]
    recommended_depth = recommend_subnet_depth(user_id, payload.hardware_profile)

    # Add to pending_clients if not already there or in connected
    pending  = proj.get("pending_clients", [])
    connected = proj.get("connected_clients", [])
    if user_id not in pending and user_id not in connected:
        pending.append(user_id)
        update_project(proj_id, {"pending_clients": pending})

    return JSONResponse(status_code=200, content={
        "status":           "pending_approval",
        "recommended_depth": recommended_depth,
        "required_schema":  proj.get("data_schema", SERVER_SCHEMA),
        "schema_version":   proj.get("schema_version", "1.0.0"),
    })


@router.get("/{proj_id}/model")
async def get_global_model(
    proj_id: str,
    current_user: dict = Depends(_get_current_user),
):
    """GET /api/projects/{proj_id}/model"""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")

    user_id = current_user["sub"]
    if user_id not in proj.get("connected_clients", []):
        raise _http_error(403, "You are not an approved participant in this project.")

    model_path = proj.get("global_model_path")
    if model_path and os.path.exists(model_path):
        weights_raw = torch.load(model_path, map_location="cpu")
        weights_json = {k: (v.numpy().tolist() if hasattr(v, 'numpy') else v.tolist())
                        for k, v in weights_raw.items()}
    else:
        # Return empty weights dict for the first round
        weights_json = {}

    # Per-client depth assignment
    db = read_db()
    user = next((u for u in db["users"] if u["user_id"] == user_id), {})
    active_depth = proj.get("recommended_depth", MODEL_CONFIG["max_depth"])

    return JSONResponse(status_code=200, content={
        "round":        proj.get("current_round", 0),
        "active_depth": active_depth,
        "weights":      weights_json,
    })


@router.post("/{proj_id}/update")
async def post_model_update(
    proj_id: str,
    payload: UpdateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(_get_current_user),
):
    """POST /api/projects/{proj_id}/update"""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")

    user_id = current_user["sub"]
    if user_id not in proj.get("connected_clients", []):
        raise _http_error(403, "Not an approved participant.")

    # Round ID check
    current_round = proj.get("current_round", 0)
    if payload.round_id != current_round:
        return JSONResponse(status_code=409, content={
            "detail": "Round ID mismatch.",
            "expected_round": current_round,
        })

    # Decrypt weights
    try:
        from shared.encryption import decrypt_weights
        decrypted = decrypt_weights(payload.weights)
    except Exception as e:
        raise _http_error(400, f"Weight decryption failed: {e}")

    update_entry = {
        "user_id":     user_id,
        "weights":     decrypted,
        "num_samples": payload.num_samples,
        "active_depth": payload.active_depth,
        "metrics":     payload.metrics,
    }

    with _buffer_lock:
        _pending_updates.setdefault(proj_id, []).append(update_entry)
        submitted = len(_pending_updates[proj_id])

    expected = len(proj.get("connected_clients", []))
    min_clients = proj.get("min_clients_per_round", 1)
    trigger = submitted >= min(expected, min_clients)

    if trigger:
        db_snapshot = read_db()
        background_tasks.add_task(
            round_lifecycle, proj_id,
            list(_pending_updates.get(proj_id, [])),
            db_snapshot,
        )
        with _buffer_lock:
            _pending_updates[proj_id] = []

    return JSONResponse(status_code=202, content={
        "status":               "received",
        "clients_submitted":    submitted,
        "clients_expected":     expected,
        "aggregation_triggered": trigger,
    })


@router.get("/{proj_id}/history")
async def get_round_history(
    proj_id: str,
    current_user: dict = Depends(_get_current_user),
):
    """GET /api/projects/{proj_id}/history"""
    db = read_db()
    history = [r for r in db.get("rounds_history", []) if r.get("proj_id") == proj_id]
    return JSONResponse(status_code=200, content=history)


@router.get("/{proj_id}/approve/{user_id_to_approve}")
async def approve_client(
    proj_id: str,
    user_id_to_approve: str,
    current_user: dict = Depends(_get_current_user),
):
    """GET /api/projects/{proj_id}/approve/{user_id} — admin action."""
    proj = get_project(proj_id)
    if proj is None:
        raise _http_error(404, f"Project {proj_id} not found.")

    # Only admin can approve
    if current_user["sub"] != proj.get("admin_id"):
        raise _http_error(403, "Only the project admin can approve clients.")

    pending   = proj.get("pending_clients", [])
    connected = proj.get("connected_clients", [])

    if user_id_to_approve not in pending:
        raise _http_error(400, "User is not in pending_clients list.")

    pending.remove(user_id_to_approve)
    connected.append(user_id_to_approve)
    update_project(proj_id, {"pending_clients": pending, "connected_clients": connected})

    # Also update user's approved_projects list
    db = read_db()
    for user in db["users"]:
        if user["user_id"] == user_id_to_approve:
            if proj_id not in user.get("approved_projects", []):
                user.setdefault("approved_projects", []).append(proj_id)
            if proj_id in user.get("pending_projects", []):
                user["pending_projects"].remove(proj_id)
    write_db(db)

    return JSONResponse(status_code=200, content={"status": "approved", "user_id": user_id_to_approve})


# ─── Background: Round Lifecycle ─────────────────────────────────────────────

def round_lifecycle(proj_id: str, updates_buffer: list, db_snapshot: dict) -> None:
    """
    Full federated round pipeline (runs as FastAPI BackgroundTask):
      1. aggregate_fedavg
      2. update_with_momentum
      3. validate_global_model
      4. evaluate_architecture_candidates (if depth diversity)
      5. Save .pt, update DB, increment round
    """
    proj = next((p for p in db_snapshot.get("projects", []) if p["proj_id"] == proj_id), None)
    if proj is None:
        return

    try:
        # ── Load current global weights ──────────────────────────────────────
        model_path = proj.get("global_model_path")
        if model_path and os.path.exists(model_path):
            current_global = {k: v.numpy() for k, v in torch.load(model_path, map_location="cpu").items()}
        else:
            current_global = {}

        weight_dicts  = [u["weights"] for u in updates_buffer]
        sample_counts = [u["num_samples"] for u in updates_buffer]

        # ── Step 1: FedAvg ───────────────────────────────────────────────────
        fedavg_result = aggregate_fedavg(weight_dicts, sample_counts)

        # ── Step 2: Momentum ─────────────────────────────────────────────────
        velocity = _velocity_state.get(proj_id, {})
        momentum = proj.get("momentum_beta", 0.9)
        new_global, new_velocity = update_with_momentum(current_global, fedavg_result, momentum, velocity)
        _velocity_state[proj_id] = new_velocity

        # ── Step 3: Validate ─────────────────────────────────────────────────
        metrics = {}
        if _val_dataloader is not None:
            metrics = validate_global_model(new_global, _val_dataloader, MODEL_CONFIG)

        # ── Step 4: NAS ──────────────────────────────────────────────────────
        updates_by_depth = {}
        for u in updates_buffer:
            d = u.get("active_depth", MODEL_CONFIG["max_depth"])
            updates_by_depth.setdefault(d, []).append(u)

        if len(updates_by_depth) > 1:
            recommended_depth = evaluate_architecture_candidates(updates_by_depth, current_global)
        else:
            recommended_depth = proj.get("recommended_depth", MODEL_CONFIG["max_depth"])

        # ── Step 5: Save model ───────────────────────────────────────────────
        new_round = proj.get("current_round", 0) + 1
        pt_path = os.path.join(MODELS_DIR, f"{proj_id}_round{new_round}.pt")
        torch.save({k: torch.from_numpy(np.array(v)) for k, v in new_global.items()}, pt_path)

        # ── Update DB ────────────────────────────────────────────────────────
        update_project(proj_id, {
            "current_round":     new_round,
            "global_model_path": pt_path,
            "recommended_depth": recommended_depth,
        })

        # Append round history
        record = {
            "proj_id": proj_id,
            "round":   new_round,
            **metrics,
        }
        append_round_history(record)

    except EmptyRoundError as e:
        print(f"[round_lifecycle] EmptyRoundError for {proj_id}: {e}")
    except Exception as e:
        print(f"[round_lifecycle] ERROR for {proj_id}: {e}")
