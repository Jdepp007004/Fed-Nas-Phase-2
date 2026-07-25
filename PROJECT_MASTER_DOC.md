# FL Platform — Master Reference Document

> **Version**: 1.0 | **Generated from source**: June 2026  
> **Purpose**: Complete end-to-end reference. Every module, class, constant, exception, and function is documented with inputs, outputs, and purpose.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Environment Variables](#4-environment-variables)
5. [Folder Structure](#5-folder-structure)
6. [Database Schema](#6-database-schema)
7. [REST API Reference](#7-rest-api-reference)
8. [Module Reference](#8-module-reference)
   - [shared/model_schema.py](#81-sharedmodel_schemapy)
   - [shared/encryption.py](#82-sharedencryptionpy)
   - [client/supernet.py](#83-clientsupernetpy)
   - [client/train_loop.py](#84-clienttrain_looppy)
   - [client/data_loader.py](#85-clientdata_loaderpy)
   - [client/schema_validator.py](#86-clientschema_validatorpy)
   - [client/api_client.py](#87-clientapi_clientpy)
   - [client/client_app.py](#88-clientclient_apppy)
   - [client/visualizer.py](#89-clientvisualizerpy)
   - [server/main.py](#810-servermainpy)
   - [server/aggregation.py](#811-serveraggregationpy)
   - [server/nas_controller.py](#812-servernas_controllerpy)
   - [server/auth_router.py](#813-serverauth_routerpy)
   - [server/project_router.py](#814-serverproject_routerpy)
   - [server/db_handler.py](#815-serverdb_handlerpy)
   - [server/ngrok_tunnel.py](#816-serverngrok_tunnelpy)
   - [download_and_split.py](#817-download_and_splitpy)
9. [Federated Round Lifecycle (Step-by-Step)](#9-federated-round-lifecycle)
10. [Client CLI Reference](#10-client-cli-reference)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Module Ownership](#12-module-ownership)

---

## 1. Project Overview

**FL Platform** enables hospitals to collaboratively train a shared neural network on TCGA clinical data **without sharing raw patient records**. Each hospital trains locally on its own private data silo, then sends only encrypted model weight updates to a central coordinator.

### Key Principles

| Principle | Implementation |
|-----------|---------------|
| **Privacy** | Raw patient data never leaves the client machine. Only weight updates are transmitted. |
| **Security** | All weight updates are AES-256-GCM encrypted before transmission. |
| **Heterogeneity** | FedProx regularisation handles non-IID data distributions across hospitals. |
| **Scalability** | NAS-adaptive subnets allow clients with different hardware to participate at different model depths. |
| **Transparency** | Live dashboard for the server operator; live Matplotlib UI for each client. |

### Multi-Task Learning Targets

The model simultaneously trains on three clinical prediction tasks:

| Task | Target Column | Loss Function | Output Shape |
|------|--------------|---------------|--------------|
| Regression | `overall_survival` (days alive) | MSE | `(B, 1)` |
| Toxicity Classification | `treatment_outcome` (4 classes: 0–3) | CrossEntropy | `(B, 4)` |
| Binary Classification | `vital_status` (0=alive, 1=dead) | BCE with Logits | `(B, 1)` |

**Toxicity mapping**: `complete response→0`, `partial response→1`, `stable disease→2`, `progressive disease→3`

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FL Platform Server                     │
│  FastAPI + ngrok · aggregation · NAS · JWT auth · DB    │
└────────────────────────┬────────────────────────────────┘
                         │  HTTPS (ngrok tunnel)
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Client 1 │    │ Client 2 │    │ Client 3 │  ...
   │ Hospital │    │ Hospital │    │ Hospital │
   │ Supernet │    │ Supernet │    │ Supernet │
   └──────────┘    └──────────┘    └──────────┘
   Local CSV       Local CSV       Local CSV
```

### Federated Round Sequence

```
Client                             Server
  │── GET /model ─────────────────▶│  Fetch round number, depth, weights
  │◀─ {round, active_depth, weights}│
  │   [load_global_weights()]       │
  │   [build_dataloaders_from_csv()]│
  │   [run_local_training(FedProx)] │
  │── POST /update (encrypted) ────▶│  Buffer update
  │                                 │  (when min_clients threshold met):
  │                                 │  aggregate_fedavg()
  │                                 │  update_with_momentum()
  │                                 │  validate_global_model()
  │                                 │  evaluate_architecture_candidates() (if depth diversity)
  │                                 │  Save .pt, increment round in DB
  │── GET /history ────────────────▶│
  │◀─ [RMSE, AUC, ToxAcc, ...]     │
      ↑ repeats indefinitely
```

---

## 3. Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| ML Model | PyTorch | 2.0 |
| Server Framework | FastAPI + Uvicorn | 0.104+ |
| Tunneling | pyngrok | — |
| Authentication | bcrypt + PyJWT (HS256) | — |
| Encryption | cryptography (AES-256-GCM) | — |
| Database | Thread-safe JSON flat-file | — |
| Data | pandas, numpy, scikit-learn | — |
| Frontend (server) | Jinja2 + Vanilla HTML/CSS/JS (Material Design) | — |
| Frontend (client) | Single HTML file — no install | — |
| CI/CD | GitHub Actions | — |
| Containers | Docker + Docker Compose | — |
| Python | 3.10+ | — |

---

## 4. Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FL_ENCRYPTION_KEY` | ✅ Server + Client | `b"\x00" * 32` (dev only) | Base64-encoded 32-byte AES-256 key. Falls back to all-zeros key in dev (insecure). |
| `NGROK_AUTH_TOKEN` | ✅ Server | — | From [ngrok dashboard](https://dashboard.ngrok.com). |
| `JWT_SECRET` | ⚠️ | `"dev_secret_change_in_production"` | JWT signing secret. **Must be changed for production.** |
| `VAL_CSV_PATH` | ❌ | — | Path to a server-side held-out validation CSV. If not set, `validate_global_model()` is skipped. |
| `SERVER_PORT` | ❌ | `8000` | Port FastAPI listens on. |

> **⚠️ Critical**: `FL_ENCRYPTION_KEY` must be **identical** on every client and the server. If they differ, every `POST /update` call fails with an AES-GCM authentication error (`ValueError`) on the server side, and no client-visible error message is emitted. Always set this variable explicitly in production — never rely on the all-zeros dev default across machines.

---

## 5. Folder Structure

```
fl_platform/
├── shared/                         # Shared by server + client — no side effects
│   ├── model_schema.py             # All constants, column lists, schema dicts
│   └── encryption.py               # AES-256-GCM encrypt / decrypt
│
├── client/                         # Runs on each hospital machine
│   ├── supernet.py                 # PyTorch Supernet model definition
│   ├── train_loop.py               # FedProx local training loop
│   ├── data_loader.py              # TCGA CSV → DataLoader pipeline
│   ├── schema_validator.py         # CSV schema validation before training
│   ├── api_client.py               # Typed HTTP client (retry + JWT + encryption)
│   ├── client_app.py               # CLI entry point — orchestrates all steps
│   ├── client_ui.html              # Browser UI (no install required)
│   ├── visualizer.py               # Matplotlib live dashboard
│   └── requirements.txt
│
├── server/                         # Runs on the central coordinator
│   ├── main.py                     # FastAPI app entrypoint + ngrok + lifespan
│   ├── aggregation.py              # FedAvg + Nesterov momentum + validation
│   ├── nas_controller.py           # NAS depth selection logic
│   ├── auth_router.py              # /api/auth/* — register + login
│   ├── project_router.py           # /api/projects/* + round lifecycle
│   ├── db_handler.py               # Thread-safe JSON database operations
│   ├── ngrok_tunnel.py             # pyngrok tunnel start / stop
│   ├── templates/dashboard.html    # Jinja2 server operator dashboard
│   ├── models/                     # Auto-created; stores .pt weight checkpoints
│   ├── database.json               # Flat-file DB (auto-created)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
│
├── tests/
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── test_unit.py                # 12 test classes covering all modules
│   └── requirements-test.txt
│
├── data/                           # Generated by download_and_split.py
│   ├── full_dataset.csv
│   ├── client_1.csv ... client_N.csv
│
├── download_and_split.py           # Download TCGA from GDC API + split for N clients
├── HOW_TO_RUN.md                   # Full step-by-step operational guide
├── README.md                       # Project overview and badges
├── pytest.ini
├── .coveragerc
└── .github/workflows/ci.yml        # CI: lint → test → docker → integration
```

---

## 6. Database Schema

The database is stored as `server/database.json` — a flat JSON file with three top-level keys:

```json
{
  "users": [
    {
      "user_id":           "uuid-string",
      "username":          "hospital_1",
      "password_hash":     "bcrypt-hash",
      "hospital_name":     "City General Hospital",
      "contact_email":     "admin@citygeneral.org",
      "approved_projects": ["proj-uuid-1"],
      "pending_projects":  [],
      "created_at":        "2026-01-01T00:00:00Z",
      "last_active":       "2026-01-01T00:00:00Z"
    }
  ],
  "projects": [
    {
      "proj_id":               "proj-uuid",
      "name":                  "TCGA Federated Demo",
      "description":           "...",
      "current_round":         3,
      "global_model_path":     "server/models/proj-uuid_round3.pt",
      "recommended_depth":     4,
      "accepting_clients":     true,
      "pending_clients":       ["user-uuid-2"],
      "connected_clients":     ["user-uuid-1"],
      "min_clients_per_round": 1,
      "momentum_beta":         0.9,
      "data_schema":           { ...SERVER_SCHEMA... },
      "schema_version":        "1.0.0"
    }
  ],
  "rounds_history": [
    {
      "proj_id":             "proj-uuid",
      "round":               1,
      "global_val_rmse":     1234.5,
      "global_tox_accuracy": 0.72,
      "global_auc":          0.85,
      "timestamp":           "2026-01-01T00:00:00Z"
    }
  ]
}
```

> **Concurrency**: All reads and writes go through `threading.RLock`. Writes use a `.tmp` file + `os.replace()` for atomicity (crash-safe).

---

## 7. REST API Reference

All endpoints are prefixed with the ngrok public URL (e.g., `https://abc123.ngrok-free.app`).

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| `GET` | `/api/status` | None | Health check — returns server version + ngrok URL |
| `GET` | `/dashboard` | None | Jinja2 HTML operator dashboard |
| `POST` | `/api/auth/register` | None | Create a new hospital account |
| `POST` | `/api/auth/login` | None | Login — returns JWT Bearer token |
| `GET` | `/api/projects` | JWT | List projects visible to authenticated user |
| `POST` | `/api/projects/{proj_id}/join` | JWT | Request to join a project; receive recommended depth |
| `GET` | `/api/projects/{proj_id}/model` | JWT (approved) | Fetch current global model weights + round info |
| `POST` | `/api/projects/{proj_id}/update` | JWT (approved) | Submit encrypted local weight update |
| `GET` | `/api/projects/{proj_id}/history` | JWT | Get round-by-round validation metrics |
| `POST` | `/api/projects/{proj_id}/approve/{user_id}` | `X-Admin-Key` header | Approve a pending client (dashboard admin action) |

> **Authentication note**: `/approve/` is protected by `X-Admin-Key` (must equal `JWT_SECRET`), not by a user JWT. All other protected endpoints use `Authorization: Bearer <token>`.

---

## 8. Module Reference

---

### 8.1 `shared/model_schema.py`

**Purpose**: Single source of truth for all model constants and TCGA schema definitions. Has no functions — only constants and dicts. Imported by both server and client to guarantee identical feature spaces.

#### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `INPUT_DIM` | `512` | Number of input features after preprocessing |
| `MAX_DEPTH` | `6` | Maximum Supernet backbone layers (NAS range: 2–6) |
| `HIDDEN_DIM` | `256` | Width of each hidden layer |
| `NUM_TOXICITY_CLASSES` | `4` | Toxicity severity classes (grade 0–3) |
| `DEFAULT_ACTIVE_DEPTH` | `4` | Default NAS depth assigned before profiling |
| `DEFAULT_LR` | `1e-3` | Default Adam learning rate |
| `DEFAULT_EPOCHS` | `5` | Default local training epochs per round |
| `DEFAULT_BATCH_SIZE` | `32` | Default DataLoader batch size |
| `DEFAULT_FEDPROX_MU` | `0.01` | Default FedProx proximal regularisation coefficient |
| `DEFAULT_CLIP_NORM` | `1.0` | Default gradient clip norm |
| `DEFAULT_VAL_SPLIT` | `0.2` | Default train/val split fraction |
| `DEFAULT_MOMENTUM_BETA` | `0.9` | Default Nesterov momentum coefficient |
| `MIN_SAMPLES` | `100` | Minimum rows a client CSV must have |

#### Schema Dicts

| Dict | Description |
|------|-------------|
| `REQUIRED_COLUMNS` | List of 62 TCGA column names a client CSV must contain |
| `TARGET_COLUMNS` | `{"regression": "overall_survival", "toxicity": "treatment_outcome", "binary": "vital_status"}` |
| `FEATURE_RANGES` | `{col: [min, max]}` for 16 numerical columns — used for min-max normalisation |
| `CATEGORICAL_VALUES` | `{col: [known_categories]}` for 5 categorical columns |
| `COLUMN_TYPES` | `{col: "float"/"str"/"int"}` for 12 key columns — used for schema validation |
| `SERVER_SCHEMA` | Combined dict sent from server to client via `/join` response |
| `MODEL_CONFIG` | `{"input_dim": 512, "max_depth": 6, "hidden_dim": 256, "num_toxicity_classes": 4}` — passed to Supernet constructor |
| `DEFAULT_TASK_WEIGHTS` | `{"regression": 1.0, "toxicity": 0.8, "binary": 0.6}` |

---

### 8.2 `shared/encryption.py`

**Purpose**: AES-256-GCM symmetric encryption of model weight dictionaries. Used by the client to encrypt weights before sending, and by the server to decrypt received weights.

#### Functions

---

##### `_get_key() -> bytes`
- **Input**: None (reads `FL_ENCRYPTION_KEY` env var)
- **Output**: `bytes` — 32-byte AES key
- **Raises**: `ValueError` if env var decodes to anything other than 32 bytes
- **Behaviour**: Falls back to `b"\x00" * 32` with a `RuntimeWarning` if env var is not set (dev only)

---

##### `_weights_to_bytes(weights: dict) -> bytes`
- **Input**: `weights` — `{str: np.ndarray}` weight dict
- **Output**: `bytes` — UTF-8 JSON where arrays are converted to nested Python lists
- **Purpose**: Serialise weights to bytes for encryption

---

##### `_bytes_to_weights(data: bytes) -> dict`
- **Input**: `data` — UTF-8 JSON bytes
- **Output**: `{str: np.ndarray}` — restored as `float32` arrays
- **Purpose**: Deserialise bytes back to weight dict after decryption

---

##### `encrypt_weights(weights: dict) -> dict`
- **Input**: `weights` — `{param_name: np.ndarray}` as produced by `get_subnet_weights()`
- **Output**:
  ```python
  {
    "ciphertext": str,  # base64-encoded AES-GCM ciphertext (includes authentication tag)
    "nonce":      str,  # base64-encoded 12-byte random nonce
  }
  ```
- **Purpose**: Encrypt weight dict using AES-256-GCM. A fresh 12-byte random nonce is generated per call.

---

##### `decrypt_weights(encrypted: dict) -> dict`
- **Input**: `encrypted` — `{"ciphertext": str, "nonce": str}` (base64 strings)
- **Output**: `{param_name: np.ndarray}` — restored weight dict
- **Raises**: `ValueError("Weight decryption failed: ...")` if key is wrong or ciphertext is tampered
- **Purpose**: Decrypt a weight payload produced by `encrypt_weights()`

---

##### `generate_key_b64() -> str`
- **Input**: None
- **Output**: `str` — base64-encoded 32-byte random key
- **Purpose**: One-time utility for generating a fresh `FL_ENCRYPTION_KEY` value at setup time

---

### 8.3 `client/supernet.py`

**Purpose**: Core PyTorch model definition. Implements a depth-flexible Supernet with three simultaneous output heads for multi-task learning.

#### Class: `Supernet(nn.Module)`

A depth-flexible neural network.

**Architecture**:
- **Backbone**: `ModuleList` of `max_depth` sequential blocks. Each block = `Linear(in, hidden_dim)` + `BatchNorm1d(hidden_dim)` + `ReLU`. First block input is `input_dim`; subsequent blocks input is `hidden_dim`.
- **Head — regression**: `Linear(hidden_dim, 1)` — predicts continuous survival days
- **Head — toxicity**: `Linear(hidden_dim, num_toxicity_classes)` — predicts toxicity grade
- **Head — binary**: `Linear(hidden_dim, 1)` — predicts vital status

##### `__init__(self, input_dim=512, max_depth=6, hidden_dim=256, num_toxicity_classes=4)`
- **Input**: Architecture hyperparameters (all default from `model_schema.py`)
- **Output**: Initialised `Supernet` instance with `self.config` dict stored
- **Purpose**: Build backbone and three output heads

##### `forward_multi_head(self, x: Tensor, active_depth: int) -> dict`
- **Input**:
  - `x`: `torch.Tensor` of shape `(batch_size, input_dim)`
  - `active_depth`: `int` — number of backbone layers to activate (`1 <= active_depth <= max_depth`)
- **Output**: `{"regression": Tensor(B,1), "toxicity": Tensor(B,4), "binary": Tensor(B,1)}`
- **Raises**: `ValueError` if `active_depth` is out of range
- **Purpose**: Core forward pass — run first N backbone layers then all three heads simultaneously

##### `forward(self, x: Tensor) -> dict`
- **Input**: `x`: `torch.Tensor` of shape `(batch_size, input_dim)`
- **Output**: Same as `forward_multi_head` at `active_depth = max_depth`
- **Purpose**: Convenience wrapper for full-depth forward pass

#### Module-level Functions

##### `compute_joint_loss(predictions: dict, targets: dict, weights: dict) -> tuple`
- **Input**:
  - `predictions`: output of `forward_multi_head()`
  - `targets`: `{"regression": FloatTensor, "toxicity": LongTensor, "binary": FloatTensor}`
  - `weights`: `{"regression": float, "toxicity": float, "binary": float}` — loss scaling coefficients
- **Output**: `(total_loss: Tensor, breakdown: {"loss_reg": float, "loss_tox": float, "loss_bin": float})`
- **Behaviour**: Guards against NaN/Inf — replaces with `zeros(1, requires_grad=True)` if infinite
- **Loss formula**: `total = w_reg*MSE + w_tox*CrossEntropy + w_bin*BCEWithLogits`

##### `get_subnet_weights(model: Supernet, active_depth: int) -> dict`
- **Input**: `model` — trained Supernet; `active_depth` — which layers were active this round
- **Output**: `{param_name: np.ndarray}` — CPU numpy arrays for backbone layers `0..active_depth-1` plus all three heads
- **Purpose**: Extract only the active subnet parameters for transmission

##### `load_global_weights(model: Supernet, weights: dict, strict: bool = False) -> None`
- **Input**: `model`, `weights: {param_name: np.ndarray}`, `strict: bool`
- **Output**: None (modifies model in-place)
- **Purpose**: Load server-provided global weights into the model. `strict=False` allows partial loading when deeper backbone layers are not in the received weights dict.

---

### 8.4 `client/train_loop.py`

**Purpose**: Execute local FedProx training for one federated round.

#### Named Tuple: `TrainConfig`

```python
class TrainConfig(NamedTuple):
    epochs:       int   = DEFAULT_EPOCHS        # 5
    lr:           float = DEFAULT_LR            # 1e-3
    active_depth: int   = DEFAULT_ACTIVE_DEPTH  # 4
    fedprox_mu:   float = DEFAULT_FEDPROX_MU    # 0.01
    task_weights: dict  = None                  # defaults to DEFAULT_TASK_WEIGHTS
    clip_norm:    float = DEFAULT_CLIP_NORM     # 1.0
```

#### Functions

##### `apply_fedprox_penalty(local_params: Iterator, global_params: Iterator, mu: float) -> torch.Tensor`
- **Input**:
  - `local_params`: `model.parameters()` iterator of currently-training model
  - `global_params`: frozen parameter tensors from the global model snapshot (created via `copy.deepcopy`)
  - `mu`: FedProx regularisation coefficient
- **Output**: `torch.Tensor` — scalar penalty value
- **Formula**: `(mu/2) * sum(||w_local - w_global||^2)`
- **Purpose**: Penalise divergence from the global model to prevent client drift

##### `run_local_training(model: Supernet, dataloader, config: TrainConfig = None, axes: dict = None) -> dict`
- **Input**:
  - `model`: Supernet pre-loaded with global weights via `load_global_weights()`
  - `dataloader`: `(train_loader, val_loader)` tuple from `create_federated_dataloader()`, or a single loader
  - `config`: `TrainConfig` NamedTuple (defaults to `TrainConfig()` if `None`)
  - `axes`: optional Matplotlib axes dict for live loss subplot update
- **Output**:
  ```python
  {
    "weights":     dict,   # from get_subnet_weights()
    "num_samples": int,    # total training samples (counted in epoch 0)
    "metrics": {
      "loss":         float,  # mean loss of last epoch
      "val_rmse":     float,  # regression RMSE on val set
      "val_acc_tox":  float,  # toxicity classification accuracy on val set
      "val_auc":      float,  # AUC-ROC for binary head on val set
    }
  }
  ```
- **Purpose**: Full local training loop. Uses Adam optimiser with FedProx penalty and gradient clipping. Calls `_run_validation()` after training if val_loader is provided.

##### `_run_validation(model, val_loader, active_depth, task_weights, device) -> tuple`
- **Input**: trained model, validation DataLoader, active depth, task weights dict, torch device
- **Output**: `(val_rmse: float, val_acc_tox: float, val_auc: float)`
- **Purpose**: Internal helper — runs no-grad inference on val set, computes RMSE (regression), accuracy (toxicity), and AUC-ROC (binary). If AUC cannot be computed (only one class), returns `0.0`.

---

### 8.5 `client/data_loader.py`

**Purpose**: Full pipeline from raw TCGA CSV to PyTorch DataLoaders.

#### Custom Exception

- **`SchemaValidationError`**: Raised when a CSV fails minimum sample count or is missing required columns.

#### Functions

##### `load_tcga_dataset(csv_path: str, schema: dict) -> pd.DataFrame`
- **Input**:
  - `csv_path`: absolute path to local TCGA CSV
  - `schema`: dict from server (from `join_project` response), with `required_columns`, `min_samples`
- **Output**: cleaned `pd.DataFrame` — only schema columns, `>70%` non-null rows, NaN-filled
- **Raises**: `FileNotFoundError`, `SchemaValidationError`
- **Cleaning steps**: lower-case column names, filter to required columns, drop rows with `>30%` missing, fill numeric NaN with median, fill categorical NaN with mode

##### `preprocess_features(df: pd.DataFrame, schema: dict) -> tuple`
- **Input**: `df` — cleaned output of `load_tcga_dataset()`; `schema` — with `feature_ranges`, `target_columns`, `categorical_values`
- **Output**: `(X: np.ndarray (N, INPUT_DIM), y: {"regression": float32, "toxicity": int64, "binary": float32})`
- **Steps**:
  1. Extract three target arrays (before touching features)
  2. Label-encode all categorical columns
  3. Min-max normalise numeric columns using schema `feature_ranges`
  4. `hstack` all encoded parts into `X_raw`
  5. Pad with zeros or truncate to exactly `INPUT_DIM=512` features

##### `_extract_regression_target(df, col) -> np.ndarray`
- **Input**: df, column name (`overall_survival`)
- **Output**: `float32` array — numeric coerce, NaN → 0.0

##### `_extract_toxicity_target(df, col) -> np.ndarray`
- **Input**: df, column name (`treatment_outcome`)
- **Output**: `int64` array — maps `{"complete response":0, "partial response":1, "stable disease":2, "progressive disease":3}`, unknown → 0

##### `_extract_binary_target(df, col) -> np.ndarray`
- **Input**: df, column name (`vital_status`)
- **Output**: `float32` array — `"dead"→1.0`, all else → `0.0`

##### `create_federated_dataloader(X: np.ndarray, y: dict, split: float = 0.2, batch_size: int = 32) -> tuple`
- **Input**: feature matrix, target dict, val fraction, batch size
- **Output**: `(train_loader: DataLoader, val_loader: DataLoader)`
- **Splitting**: `StratifiedShuffleSplit` on the binary target with `random_state=42`
- **train_loader**: `shuffle=True, drop_last=True`
- **val_loader**: `shuffle=False, drop_last=False`
- **Batch format**: each batch is `(X, y_regression, y_toxicity, y_binary)`

##### `build_dataloaders_from_csv(csv_path: str, schema: dict, split: float = 0.2, batch_size: int = 32) -> tuple`
- **Input**: same as above
- **Output**: `(train_loader, val_loader)`
- **Purpose**: One-call convenience wrapper: `load_tcga_dataset` → `preprocess_features` → `create_federated_dataloader`

---

### 8.6 `client/schema_validator.py`

**Purpose**: Pre-flight validation of a client CSV against the server's required schema, before attempting to join or train.

#### Return Type: `ValidationResult`

A named tuple / dataclass with at minimum:
- `passed: bool` — `True` if all checks pass
- Fields for individual check results (missing columns, type mismatches, range violations, etc.)

#### Functions

##### `validate_schema(df: pd.DataFrame, expected_schema: dict) -> ValidationResult`
- **Input**: `df` — raw or partially-read client CSV; `expected_schema` — the server schema dict (`SERVER_SCHEMA`)
- **Output**: `ValidationResult` — comprehensive multi-check result object
- **Checks performed**: missing required columns, column type conformance, value range checks, minimum sample count, categorical value conformance

##### `format_validation_report(result: ValidationResult) -> str`
- **Input**: `ValidationResult`
- **Output**: human-readable multi-line string suitable for printing to the CLI or browser UI
- **Purpose**: Present validation errors and warnings in a readable format

---

### 8.7 `client/api_client.py`

**Purpose**: Typed HTTP wrapper for all server REST API calls. Handles retry logic, JWT injection, encryption, and error translation into typed exceptions.

#### Custom Exceptions

| Exception | Trigger |
|-----------|---------|
| `ServerUnreachableError` | Connection error or timeout after 3 retries |
| `AuthError` | HTTP 401 or 403 response |
| `SchemaError` | HTTP 422 Unprocessable Entity |
| `RoundConflictError` | HTTP 409 Conflict (wrong round_id) |

#### Class: `APIClient`

##### `__init__(self, server_url: str, token: str = None)`
- **Input**: `server_url` — ngrok HTTPS URL; `token` — JWT (None before login)
- **Purpose**: Creates a `requests.Session` with retry strategy: 3 retries, 1.5s backoff factor, on HTTP 500/502/503/504

##### `_headers(self) -> dict`
- **Output**: `{"Content-Type": "application/json"}` + `"Authorization": "Bearer <token>"` if token is set

##### `_url(self, path: str) -> str`
- **Output**: `server_url + path`

##### `_request(self, method: str, path: str, **kwargs) -> dict`
- **Input**: HTTP method, path, any `requests` keyword args
- **Output**: parsed JSON response `dict`
- **Raises**: typed exceptions based on HTTP status codes
- **Timeout**: 30 seconds per request

##### `check_status(self) -> dict`
- **Calls**: `GET /api/status`
- **Output**: `{"server_version": str, "ngrok_url": str, ...}`

##### `register(self, username, password, hospital_name, contact_email) -> dict`
- **Calls**: `POST /api/auth/register`
- **Output**: `{"user_id": str, "username": str}`

##### `login(self, username, password) -> dict`
- **Calls**: `POST /api/auth/login`
- **Output**: `{"access_token": str, "token_type": "bearer", "user_id": str, "approved_projects": list}`
- **Side effect**: Sets `self.token = result["access_token"]`

##### `list_projects(self) -> list`
- **Calls**: `GET /api/projects`
- **Output**: list of project dicts, each including `i_am_connected`, `i_am_pending` booleans

##### `join_project(self, proj_id: str, hardware_profile: dict) -> dict`
- **Input**: `hardware_profile` — `{"ram_gb": float, "cpu_cores": int, "gpu_available": bool, "local_data_size": int}`
- **Calls**: `POST /api/projects/{proj_id}/join`
- **Output**: `{"status": "pending_approval", "recommended_depth": int, "required_schema": dict, "schema_version": str}`

##### `fetch_global_model(self, proj_id: str) -> dict`
- **Calls**: `GET /api/projects/{proj_id}/model`
- **Output**: `{"round": int, "active_depth": int, "weights": {param: np.ndarray}}` — weights are deserialised from JSON lists to `float32` numpy arrays

##### `post_local_update(self, proj_id, weights, num_samples, metrics, round_id, active_depth) -> dict`
- **Input**: `weights` — raw weight dict (not yet encrypted); all other metadata
- **Side effect**: Calls `encrypt_weights(weights)` before posting
- **Calls**: `POST /api/projects/{proj_id}/update`
- **Output**: `{"status": "received", "clients_submitted": int, "clients_expected": int, "aggregation_triggered": bool}`

##### `get_round_history(self, proj_id: str) -> list`
- **Calls**: `GET /api/projects/{proj_id}/history`
- **Output**: list of round metric records `[{"round": int, "global_val_rmse": float, ...}]`

##### `approve_client(self, proj_id: str, user_id: str) -> dict`
- **Calls**: `GET /api/projects/{proj_id}/approve/{user_id}` *(Note: this is a GET in the client — server endpoint is POST)*

---

### 8.8 `client/client_app.py`

**Purpose**: CLI entry point that orchestrates the full client federated learning lifecycle.

#### Functions

##### `parse_args() -> argparse.Namespace`
- **Output**: parsed CLI arguments

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--server` | str | ✅ | — | ngrok server URL |
| `--username` | str | ✅ | — | Account username |
| `--password` | str | ✅ | — | Account password |
| `--hospital` | str | ✅ | — | Hospital display name |
| `--email` | str | ✅ | — | Contact email |
| `--csv` | str | ✅ | — | Path to local TCGA CSV |
| `--proj` | str | ✅ | — | Project UUID to join |
| `--ram` | float | ❌ | `8.0` | RAM in GB (for NAS profiling) |
| `--cores` | int | ❌ | `4` | CPU core count |
| `--gpu` | flag | ❌ | `False` | Set if GPU is available |
| `--no-ui` | flag | ❌ | `False` | Disable Matplotlib dashboard |

##### `main() -> None`
- **Input**: None (reads from `parse_args()`)
- **Output**: None (runs indefinitely until Ctrl-C)
- **Orchestration steps**:
  1. **Connectivity check**: `check_status()`
  2. **Auth**: `login()` → on failure, attempt `register()` then `login()`
  3. **Schema validation**: reads first 500 rows, runs `validate_schema()`; exits if failed
  4. **Join project**: sends hardware profile, receives `recommended_depth` and `schema`
  5. **Wait for approval**: polls `list_projects()` every 10 seconds until `i_am_connected == True`
  6. **Init visualizer**: creates Matplotlib figure/axes (skipped with `--no-ui`)
  7. **Training loop** (infinite):
     - Fetch global model weights + current round
     - `load_global_weights()` into Supernet
     - `build_dataloaders_from_csv()` with server schema
     - `run_local_training()` with FedProx config
     - `post_local_update()` — encrypts weights automatically
     - `get_round_history()` and update dashboard
     - Sleep 5 seconds before next round

---

### 8.9 `client/visualizer.py`

**Purpose**: Live Matplotlib dashboard showing training progress for the client operator.

#### Functions

##### `init_metrics_dashboard() -> tuple`
- **Output**: `(fig, axes)` — Matplotlib Figure and a dict of 4 Axes (`2×2` grid)
- **Subplots**:
  - Top-left: Global validation RMSE (regression)
  - Top-right: Global toxicity accuracy
  - Bottom-left: Global AUC-ROC (binary)
  - Bottom-right: Local training loss (current round)

##### `update_global_metrics(axes: dict, round_history: list) -> None`
- **Input**: axes dict, list of round history records from server
- **Output**: None (redraws plots in-place)
- **Purpose**: Redraw the three global metric subplots with all rounds of history

##### `update_local_loss(axes: dict, epoch_losses: list) -> None`
- **Input**: axes dict, growing list of per-epoch loss values
- **Output**: None
- **Purpose**: Live-update the local training loss subplot during an active training round

##### `_style_ax(ax, title, xlabel, ylabel)` *(internal)*
- Applies consistent styling to a single axis

##### `_redraw_ax(ax, x, y, ylabel, title, color, marker)` *(internal)*
- Clears and redraws a single metric subplot

---

### 8.10 `server/main.py`

**Purpose**: FastAPI application entrypoint — wires up routers, starts ngrok, prepares the server-side validation dataloader.

#### Functions

##### `lifespan(app) -> AsyncGenerator`
- **Input**: FastAPI `app` instance (used as `@asynccontextmanager`)
- **Purpose**: Startup: calls `start_ngrok_tunnel()`, creates default project if none exists, optionally loads server-side validation CSV (`VAL_CSV_PATH` env var) into a DataLoader and passes it to `set_val_dataloader()`. Shutdown: pyngrok cleanup happens via `atexit`.

##### `_ensure_default_project() -> None`
- **Input**: None (reads DB)
- **Purpose**: If the database has no projects, creates one demo project with a UUID. Ensures the server is immediately usable after first boot.

##### `status() -> JSONResponse`
- **Route**: `GET /api/status`
- **Auth**: None
- **Output**: `{"status": "ok", "server_version": "...", "ngrok_url": "..."}`

##### `dashboard(request: Request) -> TemplateResponse`
- **Route**: `GET /dashboard`
- **Auth**: None
- **Output**: Rendered `templates/dashboard.html` Jinja2 template with database state injected

---

### 8.11 `server/aggregation.py`

**Purpose**: Federated optimisation engine — FedAvg, Nesterov momentum, global validation.

#### Custom Exception

- **`EmptyRoundError`**: Raised by `aggregate_fedavg()` when called with an empty update list or all-zero sample counts.

#### Functions

##### `aggregate_fedavg(client_updates: list, sample_counts: list) -> dict`
- **Input**:
  - `client_updates`: `list[dict]` — weight dicts from each client
  - `sample_counts`: `list[int]` — number of local training samples per client
- **Output**: `dict` — aggregated global weight dict (`float32`)
- **Raises**: `EmptyRoundError`, `ValueError` (if list lengths differ)
- **Algorithm**: Sample-proportional weighted average. Handles the case where different clients contribute different subsets of parameters (partial/subnet updates). Only clients that include a given key contribute to that key's average.

##### `update_with_momentum(current_global: dict, fedavg_aggregate: dict, momentum: float, velocity: dict) -> tuple`
- **Input**:
  - `current_global`: current global model weights
  - `fedavg_aggregate`: raw FedAvg output
  - `momentum`: beta coefficient (typically `0.9`)
  - `velocity`: running velocity dict from previous round (zeros at round 0)
- **Output**: `(new_global_weights: dict, updated_velocity: dict)`
- **Formula**:
  ```
  delta   = fedavg_aggregate - current_global
  v_t     = beta * v_{t-1} + (1 - beta) * delta
  new_w   = current_global + v_t
  ```
- **Note**: Handles shape mismatches gracefully — if shapes differ, takes the new aggregate directly.

##### `validate_global_model(global_weights: dict, val_dataloader, config: dict) -> dict`
- **Input**:
  - `global_weights`: freshly aggregated weights
  - `val_dataloader`: server-side held-out validation DataLoader
  - `config`: model config dict (e.g., `MODEL_CONFIG`)
- **Output**:
  ```python
  {
    "global_val_rmse":     float,  # RMSE on regression task
    "global_tox_accuracy": float,  # accuracy on toxicity task
    "global_auc":          float,  # AUC-ROC on binary task (0.0 if single class)
    "timestamp":           str,    # ISO 8601 UTC timestamp
  }
  ```
- **Purpose**: Instantiates a fresh Supernet, loads the new global weights, runs eval-mode inference on the server's validation set.
- **⚠️ Cross-tier import (undocumented runtime dependency)**: At call time this function executes `sys.path.insert(0, "../client")` and does `from supernet import Supernet`. The `client/` directory **must** be reachable relative to `server/aggregation.py` at runtime. This is satisfied by the repo layout and by `server/main.py`'s `sys.path` bootstrap, but any deployment that moves the server out of the monorepo must ensure `client/supernet.py` remains importable.

---

### 8.12 `server/nas_controller.py`

**Purpose**: Neural Architecture Search — maps client hardware profiles to optimal subnet depths and selects the globally best depth after each round.

#### Module-level State

| Variable | Type | Description |
|----------|------|-------------|
| `_DEPTH_LOOKUP` | `list[tuple]` | Lookup table: `(min_ram_gb, needs_gpu, min_data_size, max_data_size, depth)` |
| `_depth_cache` | `dict` | Per-client cache: `{client_id: assigned_depth}` |

#### `_DEPTH_LOOKUP` Table

| min_ram_gb | needs_gpu | min_data | depth |
|------------|-----------|----------|-------|
| 32 | True | 0 | 6 |
| 16 | True | 0 | 5 |
| 8 | True | 0 | 4 |
| 16 | False | 5,000 | 5 |
| 8 | False | 2,000 | 4 |
| 4 | False | 500 | 3 |
| 0 | False | 0 | 2 |

#### Functions

##### `recommend_subnet_depth(client_id: str, client_profile: dict) -> int`
- **Input**:
  - `client_id`: unique client identifier
  - `client_profile`: `{"ram_gb": float, "cpu_cores": int, "gpu_available": bool, "local_data_size": int}`
- **Output**: `int` in range `[2, MAX_DEPTH]`
- **Algorithm**: Walks `_DEPTH_LOOKUP` top-to-bottom; picks first row where RAM, GPU, and data size conditions are all satisfied. Result is clamped to `[2, 6]` and cached in `_depth_cache`.

##### `evaluate_architecture_candidates(updates_by_depth: dict, global_weights: dict) -> int`
- **Input**:
  - `updates_by_depth`: `{depth: [update_dicts]}` — client updates grouped by their active depth
  - `global_weights`: current global model weights
- **Output**: `int` — globally recommended depth for the next round, clamped to `[2, MAX_DEPTH]`
- **Algorithm**: For each depth group, computes a mini-FedAvg aggregate, then scores it as `depth_cost / delta_norm` (lower is better — high improvement per unit compute). Returns depth with best score. Falls back to `DEFAULT_ACTIVE_DEPTH` if no valid candidates.
- **Only called** if `len(updates_by_depth) > 1` (depth diversity exists among clients)
- **✅ Bug fixed (Phase 0.5 Fix A)**: Original code at line 123 read `best_score = best_depth` (integer assignment to float accumulator), causing the function to always return the last-iterated depth regardless of quality. Corrected to `best_score = score`.

---

### 8.13 `server/auth_router.py`

**Purpose**: `/api/auth/*` endpoints — user registration and login with bcrypt password hashing and JWT issuance.

#### Pydantic Request Models

- **`RegisterRequest`**: `{username: str, password: str, hospital_name: str, contact_email: str}`
- **`LoginRequest`**: `{username: str, password: str}`

#### Constants

| Constant | Value |
|----------|-------|
| `JWT_SECRET` | From `JWT_SECRET` env var (default: `"dev_secret_change_in_production"`) |
| `JWT_ALGO` | `"HS256"` |
| `JWT_EXP_HOURS` | `24` |

> **⚠️ Note**: JWTs expire after 24 hours and there is **no refresh endpoint**. A client running a multi-day training session will receive a `401 Unauthorized` when the token expires. The client must re-login (`APIClient.login()`) and retry. `client_app.py` does not currently handle this automatically.

#### Functions

##### `create_jwt(user_id: str) -> str`
- **Input**: `user_id` — UUID string
- **Output**: signed JWT string with `{"sub": user_id, "exp": now+24h}`

##### `verify_jwt(token: str) -> dict`
- **Input**: JWT string
- **Output**: payload dict `{"sub": user_id, "exp": ...}`
- **Raises**: `ValueError("Token has expired.")` or `ValueError("Invalid token: ...")` on any JWT error

##### `register_user(payload: RegisterRequest) -> JSONResponse`
- **Route**: `POST /api/auth/register`
- **Behaviour**: Checks username uniqueness (409 if taken), bcrypt-hashes password, creates user record with UUID, persists to DB
- **Returns**: `201 {"user_id": str, "username": str}` or `409 {"detail": "..."}` if duplicate

##### `login_user(payload: LoginRequest) -> JSONResponse`
- **Route**: `POST /api/auth/login`
- **Behaviour**: Looks up user by username, bcrypt-checks password, updates `last_active`, issues JWT
- **Returns**: `200 {"access_token": str, "token_type": "bearer", "approved_projects": list, "user_id": str}` or `401` on failure

---

### 8.14 `server/project_router.py`

**Purpose**: `/api/projects/*` endpoints plus the background `round_lifecycle` task.

#### Pydantic Request Models

- **`JoinRequest`**: `{"hardware_profile": dict}` — hardware profile from the client
- **`UpdateRequest`**: `{"round_id": int, "active_depth": int, "weights": dict, "num_samples": int, "metrics": dict}`

#### Module-level State

| Variable | Type | Description |
|----------|------|-------------|
| `_pending_updates` | `dict[str, list]` | In-memory buffer: `{proj_id: [update_dicts]}` — flushed after each round |
| `_velocity_state` | `dict[str, dict]` | Momentum velocity state: `{proj_id: velocity_dict}` — persists across rounds |
| `_buffer_lock` | `threading.Lock` | Protects `_pending_updates` from concurrent writes |
| `_val_dataloader` | `DataLoader | None` | Server-side validation dataloader set at startup |

> **⚠️ Persistence warning**: `_pending_updates` and `_velocity_state` are **process-local in-memory state only**. A server restart mid-round silently loses all buffered client updates (the round never completes) and resets accumulated momentum to zero (affecting convergence for the next round). There is no recovery path. Operators should avoid restarting the server while a round is in progress.

#### Functions

##### `set_val_dataloader(dl) -> None`
- **Purpose**: Setter called by `main.py` at startup to inject the validation DataLoader into this module.

##### `_get_current_user(authorization: str = Header) -> dict`
- **Purpose**: FastAPI dependency — extracts and verifies JWT from `Authorization: Bearer ...` header
- **Raises**: `HTTPException(401)` on missing or invalid token

##### `_http_error(status: int, detail: str) -> HTTPException`
- **Purpose**: Factory for raising `HTTPException` with a status code and detail string

##### `list_projects(current_user) -> JSONResponse`
- **Route**: `GET /api/projects`
- **Auth**: JWT
- **Output**: list of project dicts. A project is visible if it is `accepting_clients=True` OR the user is already approved. Each entry includes `i_am_connected: bool` and `i_am_pending: bool`. The `global_model_path` field is stripped from responses.

##### `join_project(proj_id, payload: JoinRequest, current_user) -> JSONResponse`
- **Route**: `POST /api/projects/{proj_id}/join`
- **Auth**: JWT
- **Behaviour**: Calls `recommend_subnet_depth()` with the hardware profile, adds user to `pending_clients` if not already there or in `connected_clients`
- **Output**: `{"status": "pending_approval", "recommended_depth": int, "required_schema": dict, "schema_version": str}`

##### `get_global_model(proj_id, current_user) -> JSONResponse`
- **Route**: `GET /api/projects/{proj_id}/model`
- **Auth**: JWT (must be in `connected_clients`)
- **Behaviour**: Loads `.pt` file from `global_model_path` if it exists; returns empty weights dict `{}` for round 0
- **Output**: `{"round": int, "active_depth": int, "weights": {param: list}}`

##### `post_model_update(proj_id, payload: UpdateRequest, background_tasks, current_user) -> JSONResponse`
- **Route**: `POST /api/projects/{proj_id}/update`
- **Auth**: JWT (must be in `connected_clients`)
- **Behaviour**:
  - Validates `round_id` matches `current_round` (409 if mismatch)
  - Decrypts `payload.weights` via `decrypt_weights()`
  - Appends to `_pending_updates[proj_id]`
  - If `submitted >= min(expected_clients, min_clients_per_round)`: launches `round_lifecycle` as a `BackgroundTask` and flushes the buffer
- **Output**: `{"status": "received", "clients_submitted": int, "clients_expected": int, "aggregation_triggered": bool}`

##### `get_round_history(proj_id, current_user) -> JSONResponse`
- **Route**: `GET /api/projects/{proj_id}/history`
- **Auth**: JWT
- **Output**: list of round records for this project from `rounds_history` in DB

##### `approve_client(proj_id, user_id_to_approve, request: Request) -> JSONResponse`
- **Route**: `POST /api/projects/{proj_id}/approve/{user_id}`
- **Auth**: `X-Admin-Key` header must equal `JWT_SECRET` (not a user JWT)
- **Behaviour**: Moves user from `pending_clients` to `connected_clients`, updates user's `approved_projects` list
- **Output**: `{"status": "approved", "user_id": str}`

##### `round_lifecycle(proj_id: str, updates_buffer: list, db_snapshot: dict) -> None`
- **Input**: project ID, snapshot of pending updates, DB snapshot at time of trigger
- **Output**: None (runs as a background task, updates DB as side effect)
- **5-step pipeline**:
  1. **FedAvg**: `aggregate_fedavg(weight_dicts, sample_counts)`
  2. **Momentum**: `update_with_momentum(current_global, fedavg_result, beta, velocity)` — velocity persisted in `_velocity_state`
  3. **Validate**: `validate_global_model(new_global, _val_dataloader, MODEL_CONFIG)` (skipped if `_val_dataloader is None`)
  4. **NAS**: `evaluate_architecture_candidates(updates_by_depth, current_global)` (only if depth diversity `> 1`)
  5. **Save + DB**: serialize new weights to `.pt` at `models/{proj_id}_round{N}.pt`, update `current_round`, `global_model_path`, `recommended_depth` in DB, append metrics to `rounds_history`
- **Error handling**: `EmptyRoundError` and all other exceptions are caught and printed (non-fatal)

---

### 8.15 `server/db_handler.py`

**Purpose**: Thread-safe JSON flat-file database operations.

#### Custom Exceptions

- **`ServerStorageError`**: Raised by `write_db()` when an `OSError` occurs (disk full, permissions error, etc.)

#### Module-level State

| Variable | Description |
|----------|-------------|
| `DB_PATH` | `os.path.join(server_dir, "database.json")` |
| `_db_lock` | `threading.RLock()` — re-entrant, allows same thread to re-acquire |

#### Functions

##### `read_db() -> dict`
- **Input**: None
- **Output**: `{"users": list, "projects": list, "rounds_history": list}`
- **Behaviour**: Returns empty valid structure if file doesn't exist or is corrupted JSON

##### `write_db(data: dict) -> None`
- **Input**: full new database state dict
- **Output**: None
- **Behaviour**: Writes to `database.json.tmp` first, then `os.replace()` for atomic crash-safe swap
- **Raises**: `ServerStorageError`

##### `get_project(proj_id: str) -> dict | None`
- **Input**: project UUID string
- **Output**: project dict or `None`

##### `update_project(proj_id: str, updates: dict) -> None`
- **Input**: project UUID; dict of keys to shallow-merge
- **Raises**: `KeyError` if proj_id not found
- **Behaviour**: Read → merge → write (all under `_db_lock`)

##### `get_user(user_id: str = None, username: str = None) -> dict | None`
- **Input**: either `user_id` or `username` (or both — first match wins)
- **Output**: user dict or `None`

##### `append_round_history(record: dict) -> None`
- **Input**: round metric record dict
- **Behaviour**: Read → append to `rounds_history` → write (under `_db_lock`)

---

### 8.16 `server/ngrok_tunnel.py`

**Purpose**: pyngrok tunnel lifecycle management.

#### Module-level State

| Variable | Description |
|----------|-------------|
| `_tunnel_url` | `str | None` — active public ngrok URL |

#### Functions

##### `start_ngrok_tunnel(local_port: int, auth_token: str) -> str`
- **Input**: `local_port` (e.g., `8000`), `auth_token` (ngrok auth token)
- **Output**: `str` — public HTTPS URL (e.g., `https://abc123.ngrok-free.app`)
- **Behaviour**: Calls `ngrok.set_auth_token()`, opens `ngrok.connect(local_port, proto="http")`, registers `_close_tunnel` as `atexit` handler, stores URL in `_tunnel_url`

##### `get_tunnel_url() -> str`
- **Output**: currently active ngrok URL as a string
- **Raises**: `RuntimeError("ngrok tunnel has not been initialized. Call start_ngrok_tunnel() first.")` if called before `start_ngrok_tunnel()`. Does **not** return `""` — callers must wrap in `try/except RuntimeError` (as `main.py` does).

##### `_close_tunnel() -> None`
- **Purpose**: `atexit` handler — calls `ngrok.disconnect()` and `ngrok.kill()` for clean shutdown

---

### 8.17 `download_and_split.py`

**Purpose**: One-time utility to download TCGA clinical data from the NIH GDC API and split it into per-client CSV files.

#### Functions

##### `_flatten_case(case: dict) -> dict`
- **Input**: nested GDC API case record (deeply nested JSON)
- **Output**: flat single-level dict with all relevant fields extracted to the top level

##### `download_tcga(max_cases: int) -> pd.DataFrame`
- **Input**: `max_cases` — maximum number of clinical cases to fetch
- **Output**: raw `pd.DataFrame` — paginated GDC API responses (`500` per page), concatenated
- **Source**: `https://api.gdc.cancer.gov` — no login required

##### `clean_and_align(raw_df: pd.DataFrame) -> pd.DataFrame`
- **Input**: raw GDC dataframe from `download_tcga()`
- **Output**: cleaned dataframe with GDC field names renamed to match FL Platform schema (`REQUIRED_COLUMNS`)
- **Purpose**: Bridge between GDC's field naming convention and the project's internal schema

##### `split_into_clients(df: pd.DataFrame, n_clients: int, output_dir: str) -> None`
- **Input**: cleaned dataframe, number of client silos, output directory path
- **Output**: None (writes files to disk)
- **Writes**:
  - `{output_dir}/full_dataset.csv` — all records
  - `{output_dir}/client_1.csv` ... `{output_dir}/client_N.csv` — roughly equal splits
- **CLI usage**:
  ```bash
  python download_and_split.py --max-cases 10000 --n-clients 4
  ```

---

## 9. Federated Round Lifecycle

Step-by-step detail of what happens in a single federated round:

| Step | Location | What Happens |
|------|----------|-------------|
| 1 | Client | `fetch_global_model()` → receives `{round, active_depth, weights}` |
| 2 | Client | `load_global_weights(supernet, weights)` — loads server weights, `strict=False` |
| 3 | Client | `build_dataloaders_from_csv()` — load → clean → preprocess → stratified split |
| 4 | Client | `run_local_training()` — Adam + FedProx penalty + gradient clipping × `epochs` |
| 5 | Client | `get_subnet_weights()` — extract active backbone layers + heads as numpy dict |
| 6 | Client | `encrypt_weights()` — AES-256-GCM encrypt with fresh nonce |
| 7 | Client | `post_local_update()` — POST encrypted weights + metrics to server |
| 8 | Server | Decrypt weights via `decrypt_weights()`, buffer in `_pending_updates` |
| 9 | Server | When `submitted >= min_clients_per_round`: trigger `round_lifecycle` as BackgroundTask |
| 10 | Server | `aggregate_fedavg()` — sample-proportional weighted average of all client weights |
| 11 | Server | `update_with_momentum()` — Nesterov smooth update, persist velocity |
| 12 | Server | `validate_global_model()` — eval on server-side held-out set (if `VAL_CSV_PATH` set) |
| 13 | Server | `evaluate_architecture_candidates()` — pick best depth for next round (if depth diversity) |
| 14 | Server | Save `.pt` file, update DB (`current_round`, `global_model_path`, `recommended_depth`) |
| 15 | Server | Append round metrics to `rounds_history` in DB |
| 16 | Client | `get_round_history()` → update Matplotlib dashboard |
| 17 | Client | Sleep 5s → go to step 1 |

---

## 10. Client CLI Reference

```powershell
python client/client_app.py `
  --server   https://abc123.ngrok-free.app `   # ← ngrok URL from server console
  --username hospital_1 `
  --password secret1 `
  --hospital "City General Hospital" `
  --email    admin@citygeneral.org `
  --csv      data/client_1.csv `
  --proj     YOUR_PROJECT_UUID `
  --ram      16.0 `                             # optional: RAM in GB (default 8.0)
  --cores    8 `                                # optional: CPU cores (default 4)
  --gpu `                                       # optional: flag, set if GPU present
  --no-ui                                       # optional: disable Matplotlib dashboard
```

---

## 11. CI/CD Pipeline

`.github/workflows/ci.yml` runs on every push and PR to `main`:

| Job | Command | Gate |
|-----|---------|------|
| **Lint** | `flake8` on all Python files | Fail on any style error |
| **Test** | `pytest tests/ --cov=. --cov-report=term-missing` | 60% coverage minimum (achieved: **81%** after Phase 1 implementation) |
| **Docker Build** | `docker build server/` | Fail if Dockerfile is broken |
| **Integration** | Start real server, `GET /api/status`, register + login | Fail if server doesn't start or auth fails |

---

## 12. Module Ownership

| Module | Milestone | Owner |
|--------|-----------|-------|
| `client/supernet.py` | M1 | Praneeth Raj V |
| `client/train_loop.py` | M1 / M2 | Praneeth Raj V / T Dheeraj Sai Skand |
| `server/aggregation.py` | M2 | T Dheeraj Sai Skand |
| `server/nas_controller.py` | M2 | T Dheeraj Sai Skand |
| `server/auth_router.py` | M3 | Sunishka Sarkar |
| `server/project_router.py` | M3 | Sunishka Sarkar |
| `server/db_handler.py` | M3 | Sunishka Sarkar |
| `client/api_client.py` | M3 / M4 | Nikhil Garuda |
| `client/data_loader.py` | M4 | Nikhil Garuda |
| `client/client_app.py` | M4 | Nikhil Garuda |
| `shared/model_schema.py` | M1 | Shared / all |
| `shared/encryption.py` | M1 | Shared / all |
