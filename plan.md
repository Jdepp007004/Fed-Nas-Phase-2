**E2E FEDERATED LEARNING PLATFORM**

**FOR HETEROGENEOUS SYSTEMS**

Complete Architectural & Technical Design Document

| Project ID: 18 | Guide: Prof. Ruby DinakarT Dheeraj Sai Skand (PES2UG23CS637) | Sunishka Sarkar (PES2UG23CS628)Praneeth Raj V (PES2UG23CS433) | Nikhil Garuda (PES2UG23CS195) |
| --- |

# 1\. System Architecture Overview

This document provides a complete, low-level architectural specification of the E2E Federated Learning Platform. It details every module, every function, all inter-module communication pathways, data flows, and the precise contracts between components. This serves as the authoritative technical reference beyond the slide deck.

## 1.1 High-Level Architecture Summary

The platform is organized into four primary software modules that operate across two deployment tiers: a central Server Tier and a distributed Client Tier.

| Module | Name | Tier | Owner |
| --- | --- | --- | --- |
| M1 | Core Supernet & Multi-Task ML | Server + Client (shared model weights) | Praneeth Raj V |
| M2 | Federated Optimization Engine | Server-side aggregation engine | T Dheeraj Sai Skand |
| M3 | Server Infrastructure & REST API | Server-side HTTP layer + tunneling | Sunishka Sarkar |
| M4 | Data Processing & Visualization | Client-side data + UI dashboards | Nikhil Garuda |

## 1.2 Deployment Topology

The platform operates across two physical locations connected through the internet via ngrok dynamic tunneling:

*   Central Server Node: Runs Docker containers hosting M2 (FL Engine), M3 (API Layer), and serves the global model weights produced by M1.
*   Distributed Client Nodes (N hospitals): Each independently runs M4 (data pipeline + UI) and a local training loop that instantiates the M1 Supernet architecture on their own hardware.
*   Communication Protocol: All inter-node communication uses HTTPS REST APIs. Clients never communicate with each other — all coordination passes through the central server.

## 1.3 Technology Stack & Dependency Map

The following technologies are used across modules and their inter-dependencies are:

| Technology | Used In | License | Purpose |
| --- | --- | --- | --- |
| PyTorch 2.x | M1, M2, M4 | BSD (Open-Source) | Neural network definition, training loop, tensor ops, autograd |
| FastAPI | M3 | MIT (Open-Source) | HTTP API server, request routing, OpenAPI docs auto-generation |
| Jinja2 | M3 | BSD (Open-Source) | Server-side HTML templating for the server dashboard |
| ngrok SDK | M3 | Proprietary (Free Tier) | Exposes local Docker FastAPI port to a public HTTPS URL |
| Docker / Compose | M3 | Apache 2.0 | Containerizes the entire server stack, manages port bindings |
| Python json module | M3, M4 | PSF (stdlib) | Flat-file JSON database for users/projects/rounds state |
| Python requests | M4 | Apache 2.0 | Client-side HTTP calls to the server REST API |
| Matplotlib | M4 | PSF (Open-Source) | Renders real-time training metric subplots on client UI |
| Bootstrap 5 | M3, M4 | MIT (Open-Source) | CSS framework for server and client HTML dashboards |
| Pandas / NumPy | M4 | BSD (Open-Source) | TCGA CSV ingestion, feature engineering, normalization |
| scikit-learn | M4 | BSD (Open-Source) | Stratified train/val splitting, label encoding, AUC metrics |

# 2\. Complete File & Directory Structure

The repository is organized into clearly delineated top-level directories per deployment context:

| Path | Purpose / Contents |
| --- | --- |
| fl_platform/ | Repository root |
| server/ | All server-side code (M2, M3 live here) |
| main.py | FastAPI app entry point — mounts all routers, starts ngrok |
| auth_router.py | M3: /api/auth/* endpoints (register, login) |
| project_router.py | M3: /api/projects/* endpoints (list, join, model, update) |
| aggregation.py | M2: FedAvg, FedProx, Momentum aggregation logic |
| nas_controller.py | M2: NAS depth search and subnet selection logic |
| db_handler.py | M3: Thread-safe JSON database read/write operations |
| ngrok_tunnel.py | M3: ngrok tunnel initialization and URL management |
| database.json | M3: Persistent flat-file state store (users, projects, rounds) |
| templates/ | M3: Jinja2 HTML templates for server dashboard |
| dashboard.html | Live project/client/round progress UI |
| static/ | M3: CSS/JS assets for server dashboard |
| Dockerfile | M3: Server container build instructions |
| docker-compose.yml | M3: Multi-service orchestration config |
| requirements.txt | Python package dependencies for server |
| client/ | All client-side code (M1, M4 live here) |
| client_app.py | Client entry point — launches local GUI, starts training |
| supernet.py | M1: Full Supernet PyTorch class definition |
| train_loop.py | M1/M2: Local FedProx training loop logic |
| data_loader.py | M4: TCGA dataset ingestion, cleaning, DataLoader creation |
| schema_validator.py | M4: Validates client CSV upload against server schema |
| api_client.py | M4/M3: HTTP wrapper for all server API calls |
| visualizer.py | M4: Matplotlib dashboard for metrics display |
| client_ui.html | M4: Bootstrap local web UI (file picker, ngrok input) |
| requirements.txt | Python package dependencies for client |
| shared/ | Code shared between server and client |
| model_schema.py | M1: Canonical feature list and model config constants |
| encryption.py | Utility: Weight encryption/decryption (AES-256) |
| MODULE 1: CORE SUPERNET & MULTI-TASK MLOwner: Praneeth Raj V |
| --- |

## 3.1 Module Overview

Module 1 is the machine learning heart of the entire platform. It defines the shared PyTorch model architecture (the Supernet) that every participant — server and clients — uses. The Supernet is a depth-flexible neural network with three output heads, each targeting a different clinical prediction task. This module runs on both tiers: on the server to maintain the global model, and on each client to execute local training.

## 3.2 Key Design Decisions

*   Supernet (vs. fixed-depth model): Allows clients with limited compute to train smaller subnets (fewer depth layers) while still contributing gradients that improve the shared backbone. NAS (in M2) selects the optimal depth per client.
*   Multi-task heads (vs. single task): The three prediction targets (tumor regression, toxicity classification, treatment success) share a common feature extractor backbone, enabling transfer learning across tasks and reducing total model size.
*   Gradient clipping (vs. unconstrained training): Healthcare data can produce large gradient spikes due to class imbalance. Clipping ensures training stability without requiring per-client learning rate tuning.

## 3.3 File: supernet.py

| Supernet.__init__() |
| --- |
| Signature | __init__(self, input_dim: int, max_depth: int, hidden_dim: int, num_toxicity_classes: int) |
| Purpose | PyTorch nn.Module constructor. Builds the shared backbone as a ModuleList of (max_depth) linear layers with BatchNorm and ReLU activations, and registers three independent output heads as separate nn.Linear modules. Stores config metadata as instance attributes. |
| Parameters | input_dim: int — Number of input features (must match TCGA feature count from model_schema.py, typically 512)max_depth: int — Maximum number of shared backbone layers (default 6, range 2–8)hidden_dim: int — Width of each hidden layer (default 256)num_toxicity_classes: int — Number of toxicity severity classes (default 4) |
| Returns | None (constructor). Sets self.backbone (nn.ModuleList), self.head_regression (nn.Linear), self.head_toxicity (nn.Linear), self.head_binary (nn.Linear), self.config (dict). |
| Called By | Instantiated by client_app.py on each client, and by main.py once at server startup. |
| Calls | torch.nn.ModuleList(), torch.nn.Linear(), torch.nn.BatchNorm1d(), torch.nn.ReLU() |
| Supernet.forward_multi_head() |
| --- |
| Signature | forward_multi_head(self, x: torch.Tensor, active_depth: int) -> dict |
| Purpose | The core forward pass. Feeds input tensor through the first (active_depth) layers of the backbone (ignoring deeper layers, which is how the subnet/NAS concept works), then simultaneously feeds the resulting embedding through all three task heads in parallel. This single forward pass produces all three predictions at once. |
| Parameters | x: torch.Tensor — Input batch of shape (batch_size, input_dim)active_depth: int — Number of backbone layers to activate, as selected by NAS controller (M2). Must satisfy 1 <= active_depth <= max_depth. |
| Returns | dict with keys: {'regression': Tensor(batch, 1), 'toxicity': Tensor(batch, num_classes), 'binary': Tensor(batch, 1)} |
| Called By | compute_joint_loss() within the same file; train_loop.py during client training. |
| Calls | torch.nn.functional.relu(), torch.nn.functional.sigmoid(), torch.nn.functional.softmax() |
| compute_joint_loss() |
| --- |
| Signature | compute_joint_loss(predictions: dict, targets: dict, weights: dict) -> tuple[torch.Tensor, dict] |
| Purpose | Combines the three individual task losses into one scalar value for backpropagation. Uses Mean Squared Error (MSE) for the regression head, Cross-Entropy for the toxicity head, and Binary Cross-Entropy with Logits (BCEWithLogitsLoss) for the binary head. Applies configurable task weights to balance the contribution of each loss term. Also applies gradient clipping after computing the total loss. |
| Parameters | predictions: dict — Output dict from forward_multi_head() (same keys)targets: dict — Ground truth labels with keys: {'regression': Tensor, 'toxicity': Tensor (LongTensor), 'binary': Tensor}weights: dict — Relative task importance weights, e.g. {'regression': 1.0, 'toxicity': 0.8, 'binary': 0.6} |
| Returns | tuple: (total_loss: torch.Tensor (scalar, requires_grad=True), breakdown: dict with keys 'loss_reg', 'loss_tox', 'loss_bin' as float values for logging) |
| Called By | train_loop.py (called once per batch during local training). Also called by server to compute validation loss on a small held-out shard. |
| Calls | torch.nn.MSELoss(), torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss(), torch.nn.utils.clip_grad_norm_() |
| get_subnet_weights() |
| --- |
| Signature | get_subnet_weights(model: Supernet, active_depth: int) -> dict |
| Purpose | Extracts only the weight tensors corresponding to the active subnet (the first active_depth backbone layers plus all three head layers) and returns them as a serializable dictionary of CPU numpy arrays. This is the function that packages what will be sent to the server — critically, it does NOT include the deeper layers the client did not train, so the update payload is minimal. |
| Parameters | model: Supernet — The trained local model instanceactive_depth: int — Which backbone layers were active during this round's training |
| Returns | dict mapping parameter name strings to numpy arrays. Keys follow the pattern: 'backbone.{i}.weight', 'backbone.{i}.bias', 'head_regression.weight', etc. |
| Called By | api_client.py (post_model_update()) before sending the HTTP request to the server. |
| Calls | model.state_dict(), tensor.cpu().numpy() |
| load_global_weights() |
| --- |
| Signature | load_global_weights(model: Supernet, weights: dict, strict: bool = False) -> None |
| Purpose | Deserializes the server-provided global model weights (received as a dict of numpy arrays via the API) back into the local Supernet instance. Uses strict=False by default to allow partial loading — this is essential when a client's active_depth is less than max_depth, since the global weights include all layers but the client model only needs a subset. |
| Parameters | model: Supernet — The local model instance to update in-placeweights: dict — Global weight dict (numpy arrays) received from the server via GET /api/projects/{id}/modelstrict: bool — Whether to require all keys to match (default False for subnet compatibility) |
| Returns | None. Modifies model in-place via model.load_state_dict(). |
| Called By | train_loop.py at the start of each federated round, before local training begins. |
| Calls | model.load_state_dict(), torch.from_numpy() |

## 3.4 File: train\_loop.py

| run_local_training() |
| --- |
| Signature | run_local_training(model: Supernet, dataloader: DataLoader, config: TrainConfig) -> dict |
| Purpose | The main local training loop that executes for a configured number of epochs on the client's local data. Integrates FedProx regularization directly into the loss computation (calling apply_fedprox_penalty() on each batch). This is the function that connects M1 (model architecture + loss) with M2 (FedProx penalty) on the client side. Returns a metrics summary and the updated model weights. |
| Parameters | model: Supernet — The model pre-loaded with global weights (via load_global_weights)dataloader: DataLoader — Batch iterator produced by M4's create_federated_dataloader()config: TrainConfig — Named tuple: {epochs, lr, active_depth, fedprox_mu, task_weights, clip_norm} |
| Returns | dict: {'weights': dict (from get_subnet_weights()), 'num_samples': int, 'metrics': {'loss': float, 'val_rmse': float, 'val_acc_tox': float, 'val_auc': float}} |
| Called By | client_app.py — called once per federated round after the client receives the global model. |
| Calls | compute_joint_loss() (M1), apply_fedprox_penalty() (M2 logic embedded), torch.optim.Adam() |
| apply_fedprox_penalty() |
| --- |
| Signature | apply_fedprox_penalty(local_params: Iterator, global_params: Iterator, mu: float) -> torch.Tensor |
| Purpose | Computes the FedProx proximal term that penalizes the local model for drifting too far from the global model. This is a regularization term added to the standard joint loss: (mu/2) * sum(||w_local - w_global||^2) over all parameter tensors. The mu hyperparameter controls the penalty strength; higher mu = clients stay closer to global model, which helps with non-IID data but may slow local adaptation. |
| Parameters | local_params: Iterator — model.parameters() of the currently-training local modelglobal_params: Iterator — frozen copy of the global model parameters (stored at round start, not updated during local training)mu: float — FedProx regularization coefficient (default 0.01, tunable per project config) |
| Returns | torch.Tensor — scalar penalty term to be added to total_loss before calling .backward() |
| Called By | run_local_training() — called on every batch inside the training loop. |
| Calls | torch.norm(), arithmetic tensor operations |
| MODULE 2: FEDERATED OPTIMIZATION ENGINEOwner: T Dheeraj Sai Skand |
| --- |

## 4.1 Module Overview

Module 2 is the server-side intelligence of the federated learning system. It receives raw model updates from all participating clients, and through a sequence of mathematical operations — weighted averaging, momentum smoothing, and depth-adaptive NAS selection — produces a single improved global model for the next round. This module has no direct user interface; it is invoked by M3's API endpoints upon receiving client updates.

## 4.2 File: aggregation.py

| aggregate_fedavg() |
| --- |
| Signature | aggregate_fedavg(client_updates: list[dict], sample_counts: list[int]) -> dict |
| Purpose | Implements the canonical FedAvg algorithm with sample-proportional weighting. For each parameter tensor (identified by its key name), computes a weighted sum across all client updates where each client's weight is proportional to the number of training samples it used in that round. This ensures clients with more data have more influence, correcting for imbalanced hospital sizes. Handles missing keys gracefully (for subnet partial updates) by using only the clients that include a given layer. |
| Parameters | client_updates: list[dict] — List of weight dicts (one per client), as produced by M1's get_subnet_weights(). Keys may be a subset of all model params if a client trained a shallow subnet.sample_counts: list[int] — Corresponding list of num_samples per client, used to compute weighted average coefficients |
| Returns | dict — Aggregated global weight dict (same format as client update dicts, containing ALL layers). Layers not covered by any client update in this round retain their previous global values (via server's in-memory global model state). |
| Called By | update_with_momentum() — immediately after this function produces the raw FedAvg aggregate. |
| Calls | numpy.average() with weights parameter, dict key union operations |
| update_with_momentum() |
| --- |
| Signature | update_with_momentum(current_global: dict, fedavg_aggregate: dict, momentum: float, velocity: dict) -> tuple[dict, dict] |
| Purpose | Applies server-side Nesterov-style momentum to smooth the global model update across rounds. The intuition: if the model has been moving in a consistent direction across rounds, momentum amplifies that movement, accelerating convergence. If updates are noisy/oscillating (common with non-IID data), momentum dampens the oscillations. Updates the velocity dict in-place and returns both the new global weights and the updated velocity state (which must be persisted by the server between rounds). |
| Parameters | current_global: dict — The current global model weights (before this round's update)fedavg_aggregate: dict — The raw FedAvg result from aggregate_fedavg()momentum: float — Momentum coefficient beta (default 0.9; range 0.8–0.99)velocity: dict — Running velocity state, same structure as weight dicts (initialized to zeros at round 0, persisted across rounds) |
| Returns | tuple: (new_global_weights: dict, updated_velocity: dict) |
| Called By | server's round_lifecycle() orchestrator in project_router.py (M3) — called after aggregate_fedavg(). |
| Calls | numpy element-wise arithmetic (multiply, add) |
| validate_global_model() |
| --- |
| Signature | validate_global_model(global_weights: dict, val_dataloader: DataLoader, config: dict) -> dict |
| Purpose | After aggregation, computes global validation metrics using a small held-out server-side validation shard of the TCGA dataset (managed by M4 on the server side). Instantiates a temporary Supernet at full depth, loads the new global weights, and runs inference to compute RMSE (regression head), macro accuracy (toxicity head), and AUC-ROC (binary head). Returns the metrics dict which gets logged to the round history in database.json. |
| Parameters | global_weights: dict — Freshly aggregated global weights from update_with_momentum()val_dataloader: DataLoader — Server-side validation dataloader (created once at startup by M4's create_val_dataloader())config: dict — Model config containing input_dim, max_depth, hidden_dim, num_toxicity_classes |
| Returns | dict: {'round': int, 'global_val_rmse': float, 'global_tox_accuracy': float, 'global_auc': float, 'timestamp': str} |
| Called By | project_router.py after update_with_momentum() completes a round; metrics are stored to database.json and pushed to the dashboard. |
| Calls | Supernet.forward_multi_head() (M1), compute_joint_loss() (M1), sklearn.metrics.roc_auc_score() |

## 4.3 File: nas\_controller.py

| recommend_subnet_depth() |
| --- |
| Signature | recommend_subnet_depth(client_id: str, client_profile: dict) -> int |
| Purpose | The NAS policy function that maps a client's reported hardware profile to an optimal subnet depth. Uses a simple rule-based lookup table (expandable to a learned policy in future work): maps (RAM_GB, CPU_cores, data_size) combinations to active_depth values between 2 and max_depth. The goal is to avoid assigning a depth that causes out-of-memory errors on resource-constrained clients while still maximizing the depth (and thus model quality) for well-resourced clients. |
| Parameters | client_id: str — Unique client identifier (for logging and caching prior assignments)client_profile: dict — Hardware self-report from client: {'ram_gb': float, 'cpu_cores': int, 'gpu_available': bool, 'local_data_size': int} |
| Returns | int — Recommended active_depth value (in range [2, max_depth]). This value is returned to the client in the project join response. |
| Called By | project_router.py's join_project() endpoint handler — called when a client POSTs to /api/projects/{id}/join. |
| Calls | Internal lookup table (dict), no external function calls |
| evaluate_architecture_candidates() |
| --- |
| Signature | evaluate_architecture_candidates(updates_by_depth: dict[int, list], global_weights: dict) -> int |
| Purpose | A more sophisticated NAS function that runs at the end of a federated round (if enough clients with different depths have participated). It compares the validation loss contribution from client groups using depth=2, 3, 4, etc., and selects the depth that produced the best average validation improvement per additional compute cost. This allows the system to adapt the global depth recommendation over time as more data about client contributions accumulates. |
| Parameters | updates_by_depth: dict[int, list] — Groups the client updates from this round by their active_depth value. Keys are depth integers, values are lists of weight dicts.global_weights: dict — Current global weights for computing relative improvement per depth group |
| Returns | int — Globally recommended depth for the next round (broadcast to all clients in the next model response). |
| Called By | validate_global_model() — called inside this function to evaluate each depth candidate; project_router.py round_lifecycle() — calls this function at round end. |
| Calls | validate_global_model() (M2), aggregate_fedavg() (M2) — runs multiple mini-aggregations per depth group |
| MODULE 3: SERVER INFRASTRUCTURE & REST APIOwner: Sunishka Sarkar |
| --- |

## 5.1 Module Overview

Module 3 is the connectivity and state management backbone. It exposes the entire server functionality to clients via a well-defined REST API, manages all persistent state through a thread-safe JSON database, handles authentication and authorization, and coordinates the lifecycle of federated rounds. It also tunnels the local server to the public internet via ngrok and serves the server operator dashboard.

## 5.2 File: ngrok\_tunnel.py

| start_ngrok_tunnel() |
| --- |
| Signature | start_ngrok_tunnel(local_port: int, auth_token: str) -> str |
| Purpose | Initializes the ngrok SDK, authenticates with the provided auth token, and opens a persistent HTTPS tunnel to the given local port where FastAPI is listening. Polls the ngrok API until the tunnel is confirmed active. Stores the resulting public URL in a module-level singleton. Also registers a cleanup handler (atexit) to close the tunnel gracefully when the server process terminates. |
| Parameters | local_port: int — The localhost port FastAPI is bound to (typically 8000)auth_token: str — ngrok authentication token, loaded from environment variable NGROK_AUTH_TOKEN |
| Returns | str — The full public HTTPS URL (e.g., 'https://abc123.ngrok.io'). This URL is displayed on the server dashboard and must be manually shared with clients. |
| Called By | main.py startup event — called once when the FastAPI application starts. |
| Calls | ngrok.set_auth_token(), ngrok.connect(), atexit.register() |
| get_tunnel_url() |
| --- |
| Signature | get_tunnel_url() -> str |
| Purpose | Simple accessor that returns the currently active ngrok public URL from the module-level singleton. Used by the dashboard template renderer and by API health-check endpoints to display the public URL to the server operator. |
| Parameters | None |
| Returns | str — Active ngrok URL, or raises RuntimeError if tunnel has not been initialized. |
| Called By | dashboard Jinja2 template render function in main.py; GET /api/status endpoint. |
| Calls | None — reads from module-level singleton variable |

## 5.3 File: db\_handler.py

| read_db() |
| --- |
| Signature | read_db() -> dict |
| Purpose | Thread-safe read of the entire database.json file. Acquires a threading.RLock before opening the file to prevent concurrent read/write races in the async FastAPI environment. Deserializes the JSON content into a Python dict and returns it. Uses a re-entrant lock to allow nested calls within the same thread (e.g., a function that calls read_db() and then later calls write_db() without deadlocking). |
| Parameters | None — reads from the module-level DB_PATH constant (path to database.json) |
| Returns | dict — Full database state. Top-level keys: 'users', 'projects', 'rounds_history'. |
| Called By | All API endpoint handlers in auth_router.py and project_router.py that need to read state. |
| Calls | json.load(), threading.RLock() |
| write_db() |
| --- |
| Signature | write_db(data: dict) -> None |
| Purpose | Thread-safe write of the full database state. Acquires the same RLock as read_db(), then writes the provided dict to database.json using json.dump() with indent=2 for human readability. Performs an atomic write by writing to a .tmp file first, then os.replace() to swap it in, preventing data corruption if the process is killed mid-write. |
| Parameters | data: dict — The complete new database state to persist. Callers are responsible for first calling read_db(), modifying the result, and then passing the full modified dict to write_db(). |
| Returns | None. Side effect: overwrites database.json. |
| Called By | All API endpoint handlers that modify state (register, login, join, post update). |
| Calls | json.dump(), os.replace(), threading.RLock() |
| get_project() |
| --- |
| Signature | get_project(proj_id: str) -> dict | None |
| Purpose | Convenience accessor that calls read_db() and extracts the specific project entry matching proj_id from the 'projects' list. Returns None if no matching project is found (instead of raising KeyError), allowing callers to produce clean 404 HTTP responses. |
| Parameters | proj_id: str — The UUID string identifying the project |
| Returns | dict (single project record) or None if not found. |
| Called By | All project_router.py endpoints that operate on a specific project. |
| Calls | read_db() |
| update_project() |
| --- |
| Signature | update_project(proj_id: str, updates: dict) -> None |
| Purpose | Reads the full DB, finds the project with matching proj_id in the 'projects' list, merges the updates dict into that project record (shallow merge — updates keys override existing keys), and writes the full modified DB back. This is the canonical way to modify any project-level field (e.g., incrementing current_round, appending to connected_clients, saving new global_model_path). |
| Parameters | proj_id: str — Target project UUIDupdates: dict — Key-value pairs to merge into the project record, e.g. {'current_round': 3, 'global_model_path': '/models/round3.pt'} |
| Returns | None. Side effect: modifies database.json. |
| Called By | project_router.py's round_lifecycle(), join_project(), post_model_update() handlers. |
| Calls | read_db(), write_db() |

## 5.4 File: auth\_router.py (FastAPI APIRouter)

| register_user() [POST /api/auth/register] |
| --- |
| Signature | register_user(payload: RegisterRequest) -> JSONResponse |
| Purpose | Handles new client hospital registration. Validates that the username is unique (reads DB, checks 'users' list). Hashes the provided password using bcrypt with a random salt. Creates a new user record with a UUID, hashed password, empty approved_projects list, and pending_projects list. Writes the new user to DB and returns a 201 response with the assigned user_id. |
| Parameters | payload: RegisterRequest — Pydantic model: {username: str, password: str, hospital_name: str, contact_email: str} |
| Returns | JSONResponse 201: {'user_id': str, 'username': str} or JSONResponse 409 if username exists. |
| Called By | HTTP POST from client's api_client.py register() function. |
| Calls | read_db(), write_db(), bcrypt.hashpw(), uuid.uuid4() |
| login_user() [POST /api/auth/login] |
| --- |
| Signature | login_user(payload: LoginRequest) -> JSONResponse |
| Purpose | Authenticates a client. Reads DB, finds user by username, verifies the password against the stored bcrypt hash. If valid, generates a signed JWT token (using the server's JWT_SECRET env variable) with a 24-hour expiry containing the user_id as the subject claim. Returns the token and a summary of the user's approved projects. The JWT is used as a Bearer token in all subsequent API calls. |
| Parameters | payload: LoginRequest — Pydantic model: {username: str, password: str} |
| Returns | JSONResponse 200: {'access_token': str, 'token_type': 'bearer', 'approved_projects': list[str]} or JSONResponse 401 if credentials invalid. |
| Called By | HTTP POST from client's api_client.py login() function. |
| Calls | read_db(), bcrypt.checkpw(), jwt.encode() (PyJWT library) |

## 5.5 File: project\_router.py (FastAPI APIRouter)

| list_projects() [GET /api/projects] |
| --- |
| Signature | list_projects(current_user: dict = Depends(verify_jwt)) -> JSONResponse |
| Purpose | Returns the list of all active FL projects from the database, filtered to show only projects the authenticated user is approved for (if any) plus all open-enrollment projects. Each project entry includes the data_schema the client must conform to, the current_round number, the minimum required data samples, and whether the project is accepting new clients. This is the first call a new client makes after login to discover what they can join. |
| Parameters | current_user: dict — Injected by FastAPI's Depends() using verify_jwt() dependency function |
| Returns | JSONResponse 200: list of project objects, each containing {proj_id, name, description, admin_id, data_schema, current_round, min_samples, accepting_clients: bool} |
| Called By | HTTP GET from client's api_client.py list_available_projects() function, called during project selection UI. |
| Calls | read_db(), verify_jwt() (dependency) |
| join_project() [POST /api/projects/{proj_id}/join] |
| --- |
| Signature | join_project(proj_id: str, payload: JoinRequest, current_user: dict = Depends(verify_jwt)) -> JSONResponse |
| Purpose | Processes a client's request to participate in a federated project. Validates that the project exists and is accepting clients. Calls M2's recommend_subnet_depth() with the client's hardware profile (from payload) to determine the appropriate active_depth. Adds the client to the project's pending_clients list in the DB (admin approval required before they appear in connected_clients). Returns the recommended active_depth and the required data schema. |
| Parameters | proj_id: str — Target project UUID (from URL path)payload: JoinRequest — Pydantic model: {hardware_profile: dict (ram_gb, cpu_cores, gpu_available, local_data_size)}current_user: dict — From JWT |
| Returns | JSONResponse 200: {'status': 'pending_approval', 'recommended_depth': int, 'required_schema': dict, 'schema_version': str} or 404/403 if project not found/full. |
| Called By | HTTP POST from client's api_client.py join_project() function. |
| Calls | get_project() (M3), recommend_subnet_depth() (M2), update_project() (M3) |
| get_global_model() [GET /api/projects/{proj_id}/model] |
| --- |
| Signature | get_global_model(proj_id: str, current_user: dict = Depends(verify_jwt)) -> JSONResponse |
| Purpose | Returns the current global model weights for the specified project. Verifies the requesting client is in the project's connected_clients list (not just pending). Loads the model weights from the path stored in database.json (global_model_path), serializes the numpy weight dict to JSON-safe format (lists of floats via .tolist()), and returns them. Also returns the current round number and the client's assigned active_depth. |
| Parameters | proj_id: str — Target project UUIDcurrent_user: dict — From JWT |
| Returns | JSONResponse 200: {'round': int, 'active_depth': int, 'weights': dict (param_name -> list of floats)} or 403 if client not approved. |
| Called By | HTTP GET from client's api_client.py fetch_global_model() function, called at the start of each training round. |
| Calls | get_project() (M3), verify_jwt(), numpy serialization, torch.save()/load() for .pt files |
| post_model_update() [POST /api/projects/{proj_id}/update] |
| --- |
| Signature | post_model_update(proj_id: str, payload: UpdateRequest, current_user: dict = Depends(verify_jwt)) -> JSONResponse |
| Purpose | The most critical endpoint — receives a trained local model update from a client. Decrypts the received weight payload, validates its schema matches the project's expected model architecture, and appends it to the in-memory pending_updates buffer for this project/round. Checks if all expected clients for this round have submitted. If yes, triggers round_lifecycle() asynchronously via FastAPI's BackgroundTasks to run the full aggregation pipeline (M2) without blocking the HTTP response. Returns acknowledgment immediately. |
| Parameters | proj_id: str — Target project UUIDpayload: UpdateRequest — Pydantic model: {round_id: int, active_depth: int, weights: dict (encrypted, serialized), num_samples: int, metrics: dict}current_user: dict — From JWT |
| Returns | JSONResponse 202 Accepted: {'status': 'received', 'clients_submitted': int, 'clients_expected': int, 'aggregation_triggered': bool} |
| Called By | HTTP POST from client's api_client.py post_local_update() function, called after local training completes. |
| Calls | decrypt_weights() (shared/encryption.py), db_handler functions, BackgroundTasks.add_task(round_lifecycle) |
| round_lifecycle() [Background Task] |
| --- |
| Signature | round_lifecycle(proj_id: str, updates_buffer: list, db_snapshot: dict) -> None |
| Purpose | The orchestrator function that runs the complete federated round pipeline as a background task. Calls the M2 aggregation functions in sequence: (1) aggregate_fedavg() to compute weighted average, (2) update_with_momentum() to apply smoothing, (3) validate_global_model() to compute quality metrics, (4) evaluate_architecture_candidates() if enough depth diversity exists. Saves the new global model to disk, updates database.json with the new round number/model path/metrics history, and broadcasts a round-complete event to the dashboard via Server-Sent Events (SSE). |
| Parameters | proj_id: str — Project being updatedupdates_buffer: list — All client update payloads collected this round (already decrypted)db_snapshot: dict — Snapshot of DB state at aggregation time (to avoid re-reading mid-operation) |
| Returns | None. Side effects: writes new .pt model file to disk, updates database.json, emits SSE event. |
| Called By | Called by post_model_update() via BackgroundTasks (asynchronously, does not block HTTP responses). |
| Calls | aggregate_fedavg() (M2), update_with_momentum() (M2), validate_global_model() (M2), evaluate_architecture_candidates() (M2), update_project() (M3), torch.save() |
| MODULE 4: DATA PROCESSING & VISUALIZATIONOwner: Nikhil Garuda |
| --- |

## 6.1 Module Overview

Module 4 runs entirely on the client side and handles two distinct responsibilities: (1) transforming raw TCGA clinical CSV files into PyTorch DataLoaders ready for M1's training loop, and (2) rendering the client-side graphical interface — both the data upload UI and the real-time training metrics dashboard. It also contains the schema validation logic that ensures a client's data is compatible with the project they wish to join.

## 6.2 File: data\_loader.py

| load_tcga_dataset() |
| --- |
| Signature | load_tcga_dataset(csv_path: str, schema: dict) -> pd.DataFrame |
| Purpose | Loads the TCGA Clinical Subset CSV from the client's local filesystem. Performs initial cleanup: drops rows with more than 30% missing values, fills remaining missing values using column medians (numerical) or mode (categorical). Filters the dataframe to only include columns listed in the server-provided schema dict. Performs basic sanity checks (minimum row count, required column presence) and raises SchemaValidationError if checks fail. |
| Parameters | csv_path: str — Absolute path to the client's local TCGA CSV fileschema: dict — Required schema from the project (received from list_projects() or join_project()). Contains: {'required_columns': list[str], 'target_columns': dict, 'min_samples': int} |
| Returns | pd.DataFrame — Cleaned dataframe with only schema-required columns, no NaN values, minimum min_samples rows. |
| Called By | preprocess_features() — called immediately after to encode and normalize the clean dataframe. |
| Calls | pandas.read_csv(), pd.DataFrame.fillna(), pd.DataFrame.dropna() |
| preprocess_features() |
| --- |
| Signature | preprocess_features(df: pd.DataFrame, schema: dict) -> tuple[np.ndarray, dict] |
| Purpose | Transforms the clean dataframe into model-ready feature arrays. Applies label encoding to categorical columns (e.g., cancer stage, tissue type), min-max normalization to all numerical columns using the ranges provided in the schema (so all clients normalize identically, ensuring weight compatibility), and one-hot encoding to low-cardinality categoricals. Splits the dataframe into feature matrix X and target dict y (regression target, toxicity labels, binary labels as separate arrays). Returns both X and y. |
| Parameters | df: pd.DataFrame — Cleaned dataframe from load_tcga_dataset()schema: dict — Same schema dict, contains 'feature_ranges' for normalization bounds, 'target_columns' for label extraction, 'encoding_maps' for categorical mappings |
| Returns | tuple: (X: np.ndarray of shape (N, input_dim), y: dict {'regression': np.ndarray(N,), 'toxicity': np.ndarray(N,), 'binary': np.ndarray(N,)}) |
| Called By | create_federated_dataloader() — called after preprocessing to wrap arrays into PyTorch DataLoaders. |
| Calls | sklearn.preprocessing.LabelEncoder(), sklearn.preprocessing.MinMaxScaler(), pandas.get_dummies() |
| create_federated_dataloader() |
| --- |
| Signature | create_federated_dataloader(X: np.ndarray, y: dict, split: float, batch_size: int) -> tuple[DataLoader, DataLoader] |
| Purpose | Wraps the preprocessed numpy arrays into PyTorch TensorDataset objects and creates DataLoaders for training and validation. Uses stratified splitting (on the binary target column) to ensure both splits contain balanced positive/negative examples. The training DataLoader uses shuffle=True and drop_last=True (to avoid batch norm errors with size-1 batches). The validation DataLoader uses shuffle=False. |
| Parameters | X: np.ndarray — Feature matrix from preprocess_features()y: dict — Target arrays from preprocess_features()split: float — Fraction of data for validation (default 0.2)batch_size: int — Training batch size (default 32) |
| Returns | tuple: (train_loader: DataLoader, val_loader: DataLoader) |
| Called By | run_local_training() (M1) — passes the train_loader to the training loop; validate_global_model() (M2) — uses a server-side equivalent DataLoader for validation. |
| Calls | torch.utils.data.TensorDataset, torch.utils.data.DataLoader, sklearn.model_selection.StratifiedShuffleSplit() |

## 6.3 File: schema\_validator.py

| validate_schema() |
| --- |
| Signature | validate_schema(df: pd.DataFrame, expected_schema: dict) -> ValidationResult |
| Purpose | Performs a comprehensive multi-check validation of the client's dataframe against the project's required schema. Checks are: (1) all required columns present, (2) column data types match expected types, (3) value ranges for numerical columns are within expected bounds, (4) categorical columns contain only known categories, (5) row count >= min_samples. Returns a ValidationResult namedtuple containing a boolean passed field and a list of human-readable error/warning messages for display in the client UI. |
| Parameters | df: pd.DataFrame — The client's loaded dataexpected_schema: dict — Schema from the server project listing. Contains 'required_columns', 'column_types', 'value_bounds', 'categorical_values', 'min_samples' |
| Returns | ValidationResult namedtuple: (passed: bool, errors: list[str], warnings: list[str]) |
| Called By | client_app.py — called after the user selects their data file but before joining a project; result is displayed in the client UI. |
| Calls | pandas type checking operations, custom bounds-checking logic |

## 6.4 File: visualizer.py

| init_metrics_dashboard() |
| --- |
| Signature | init_metrics_dashboard() -> tuple[plt.Figure, dict] |
| Purpose | Creates a persistent Matplotlib figure with three subplots arranged in a 2x2 grid: (1) Global Validation RMSE over rounds (line plot), (2) Toxicity Classification Accuracy over rounds (line plot), (3) AUC-ROC over rounds (line plot), (4) Local training loss per epoch in the current round (line plot). Returns the figure object and a dict of Axes objects keyed by plot name, so they can be updated incrementally without re-creating the figure. |
| Parameters | None — creates a fresh figure |
| Returns | tuple: (fig: plt.Figure, axes: dict with keys 'rmse', 'tox_acc', 'auc', 'local_loss') |
| Called By | client_app.py startup — called once to initialize the dashboard window before the first training round. |
| Calls | matplotlib.pyplot.subplots(), matplotlib.pyplot.ion() (interactive mode) |
| update_global_metrics() |
| --- |
| Signature | update_global_metrics(axes: dict, round_history: list[dict]) -> None |
| Purpose | Called after each federated round completes (when the client receives a response from the server indicating a new global model is available). Re-draws the three global metric subplots (RMSE, Tox Accuracy, AUC) with the full history of rounds to date, animating the convergence curves. Uses plt.pause() with a short timeout for non-blocking rendering inside the training loop. |
| Parameters | axes: dict — The axes dict from init_metrics_dashboard()round_history: list[dict] — List of round metric records, each containing 'round', 'global_val_rmse', 'global_tox_accuracy', 'global_auc' |
| Returns | None. Side effect: updates the Matplotlib figure in-place. |
| Called By | client_app.py main loop — called after fetch_global_model() returns new round info. |
| Calls | matplotlib.axes.Axes.clear(), Axes.plot(), Axes.set_title(), plt.pause() |
| update_local_loss() |
| --- |
| Signature | update_local_loss(axes: dict, epoch_losses: list[float]) -> None |
| Purpose | Updates the local training loss subplot in real-time during a training round, called after each epoch completes. Clears and re-draws the local loss subplot with the accumulated per-epoch loss values so far in the current round, giving the client operator a live view of training convergence. |
| Parameters | axes: dict — The axes dict from init_metrics_dashboard()epoch_losses: list[float] — List of mean epoch losses accumulated so far in the current round |
| Returns | None. Side effect: updates local_loss subplot. |
| Called By | run_local_training() (M1) — called at the end of each epoch inside the training loop. |
| Calls | matplotlib.axes.Axes.clear(), Axes.plot(), plt.pause() |

## 6.5 File: api\_client.py

| APIClient class (all methods) |
| --- |
| Signature | APIClient(server_url: str, token: str = None) |
| Purpose | A thin wrapper class around the requests library that encapsulates all HTTP communication with the server. Stores the server base URL (ngrok URL) and the JWT token, and provides named methods for each REST endpoint. All methods use a shared requests.Session with retry logic (3 retries with exponential backoff) for resilience against ngrok latency spikes. Raises custom exceptions (ServerUnreachableError, AuthError, SchemaError) rather than raw HTTP errors. |
| Parameters | server_url: str — The ngrok HTTPS URL entered by the client operatortoken: str — JWT token (populated after successful login(), None before) |
| Returns | APIClient instance. Key methods: register(payload), login(payload), list_projects(), join_project(proj_id, hardware_profile), fetch_global_model(proj_id), post_local_update(proj_id, weights, num_samples, metrics). |
| Called By | client_app.py — instantiated once at startup with the operator-provided ngrok URL. |
| Calls | requests.Session, requests.adapters.HTTPAdapter, urllib3.util.retry.Retry |

# 7\. Complete Inter-Module Connection Map

This section exhaustively documents every point at which one module calls into or depends on another, the data format exchanged, and the error handling contract.

**7.1 M4 (Client Data) → M1 (Training)**

| From → To | Function/Endpoint | Data Exchanged | Trigger Condition |
| --- | --- | --- | --- |
| M4 → M1 | create_federated_dataloader() → run_local_training() | DataLoader (batches of X: Tensor, y: dict) | Client starts a new training round |
| M1 → M4 | update_local_loss() | list[float] (epoch loss values) | After each training epoch |
| M4 → M1 | load_tcga_dataset() feeds preprocess_features() feeds DataLoader | Tensors of shape (N, 512) matching Supernet input_dim | One-time at round start |

**7.2 M1 (Training) → M2 (Optimization) — Server-side**

| From → To | Function/Endpoint | Data Exchanged | Trigger Condition |
| --- | --- | --- | --- |
| M1 → M2 | get_subnet_weights() output → aggregate_fedavg() | dict of param_name: np.ndarray | After all clients post updates |
| M2 → M1 | validate_global_model() instantiates Supernet | global weights dict → Supernet loaded via load_global_weights() | After every aggregation |
| M1 → M2 | apply_fedprox_penalty() called inside run_local_training() | frozen global_params Iterator | Each batch during local training |

**7.3 M3 (API) → M2 (Aggregation) — Server-side orchestration**

| From → To | Function/Endpoint | Data Exchanged | Trigger Condition |
| --- | --- | --- | --- |
| M3 → M2 | round_lifecycle() calls aggregate_fedavg() | list of client weight dicts + sample counts | When all clients have submitted for a round |
| M3 → M2 | round_lifecycle() calls update_with_momentum() | raw FedAvg aggregate + velocity state | Immediately after aggregate_fedavg() |
| M3 → M2 | round_lifecycle() calls validate_global_model() | new global weights + val DataLoader | After momentum update |
| M3 → M2 | join_project() calls recommend_subnet_depth() | client hardware_profile dict | When a client POSTs to /join |
| M2 → M3 | evaluate_architecture_candidates() returns int | recommended global depth for next round | Stored to DB by update_project() |

**7.4 M4 (Client API Client) → M3 (Server REST API)**

| From → To | Function/Endpoint | Data Exchanged | Trigger Condition |
| --- | --- | --- | --- |
| M4 → M3 | APIClient.register() | JSON: {username, password, hospital_name} | First-time setup |
| M4 → M3 | APIClient.login() | JSON: {username, password} | Each session start |
| M4 → M3 | APIClient.list_projects() | Bearer token in header | After login, project selection |
| M4 → M3 | APIClient.join_project() | JSON: {hardware_profile} | Once per project join |
| M4 → M3 | APIClient.fetch_global_model() | Bearer token | Start of each FL round |
| M4 → M3 | APIClient.post_local_update() | JSON: {weights (encrypted), num_samples, metrics} | After local training completes |
| M3 → M4 | GET /model response | JSON: {weights dict, round, active_depth} | Response to fetch request |

# 8\. Database Schema (database.json)

The entire server state is persisted in a single JSON file. Below is the complete schema with field-level documentation.

## 8.1 Top-Level Structure

{

"users": \[ ...UserRecord \],

"projects": \[ ...ProjectRecord \],

"rounds\_history": \[ ...RoundRecord \]

}

## 8.2 UserRecord

| Field | Type | Description |
| --- | --- | --- |
| user_id | string (UUID) | Primary key. Auto-generated at registration. Format: uuid4. |
| username | string | Unique hospital identifier. Checked for uniqueness on registration. |
| password_hash | string | bcrypt hash of password (never store plaintext). |
| hospital_name | string | Display name of the participating hospital. |
| contact_email | string | Admin email for project notifications. |
| approved_projects | list[string] | List of proj_id strings this user is approved to participate in. |
| pending_projects | list[string] | List of proj_id strings where join request is awaiting admin approval. |
| created_at | string (ISO8601) | Registration timestamp. |
| last_active | string (ISO8601) | Timestamp of most recent API call (updated on each authenticated request). |

## 8.3 ProjectRecord

| Field | Type | Description |
| --- | --- | --- |
| proj_id | string (UUID) | Primary key. Set at project creation by server admin. |
| name | string | Human-readable project name shown on client UI. |
| admin_id | string (UUID) | user_id of the project creator/admin. |
| data_schema | dict | Required columns, types, bounds, encodings. Sent to clients on list/join. |
| schema_version | string (semver) | e.g., '1.2.0'. Clients must match this to participate. |
| current_round | int | Zero-indexed federated round counter. |
| max_rounds | int | Stop condition. Training ends when current_round == max_rounds. |
| min_clients_per_round | int | Aggregation only triggers when this many updates are received. |
| connected_clients | list[string] | user_ids of approved, active clients. |
| pending_clients | list[string] | user_ids of clients awaiting approval. |
| global_model_path | string | Filesystem path to the .pt file for the current global model. |
| fedprox_mu | float | FedProx mu value for this project (may differ per project). |
| momentum_beta | float | Server momentum coefficient for this project. |
| recommended_depth | int | Current NAS-recommended subnet depth for all clients. |
| accepting_clients | bool | Controls whether new join requests are accepted. |
| created_at | string (ISO8601) | Project creation timestamp. |

# 9\. End-to-End Sequence Flows

## 9.1 Client Onboarding Flow (One-time)

The sequence of events when a new hospital client sets up for the first time:

| Step | Actor | Action | Result |
| --- | --- | --- | --- |
| 1 | Client Operator | Opens client_ui.html, enters ngrok URL | UI sends GET /api/status to verify connectivity |
| 2 | M4 APIClient | Calls APIClient.register() | M3 register_user() creates DB record, returns user_id |
| 3 | M4 APIClient | Calls APIClient.login() | M3 login_user() validates, returns JWT access_token |
| 4 | M4 APIClient | Calls APIClient.list_projects() | M3 list_projects() returns project list with schemas |
| 5 | Client Operator | Selects project, uploads CSV file | M4 validate_schema() checks CSV against project schema |
| 6 | M4 APIClient | Calls APIClient.join_project() with hardware profile | M3 join_project() → M2 recommend_subnet_depth() → returns active_depth |
| 7 | Server Admin | Approves client in dashboard | M3 moves client from pending_clients → connected_clients in DB |
| 8 | M4 APIClient | Next poll of GET /api/projects returns approved status | Client proceeds to federated training loop |

## 9.2 Federated Training Round Flow

The sequence of events for a single federated learning round (repeated max\_rounds times):

| Step | Actor | Function Called | Output |
| --- | --- | --- | --- |
| R1 | Each Client (M4) | APIClient.fetch_global_model(proj_id) | Receives global weights dict + active_depth + round number |
| R2 | Each Client (M1) | load_global_weights(model, weights) | Local model updated with server global weights |
| R3 | Each Client (M4) | load_tcga_dataset() → preprocess_features() → create_federated_dataloader() | Train/val DataLoaders ready |
| R4 | Each Client (M1+M2) | run_local_training(model, dataloader, config) | Local model trained; apply_fedprox_penalty() called each batch |
| R5 | Each Client (M1) | get_subnet_weights(model, active_depth) | Serialized weight dict of trained subnet only |
| R6 | Each Client (M4) | APIClient.post_local_update(proj_id, weights, num_samples, metrics) | Server receives encrypted update, returns 202 Accepted |
| R7 | Server (M3) | post_model_update() counts submissions | When all N expected clients have submitted, triggers BackgroundTask |
| R8 | Server (M2) | aggregate_fedavg(updates, sample_counts) | Raw FedAvg aggregate dict |
| R9 | Server (M2) | update_with_momentum(current, aggregate, beta, velocity) | Smoothed new global weights + updated velocity |
| R10 | Server (M2) | validate_global_model(new_weights, val_loader) | Metrics: RMSE, Tox Accuracy, AUC |
| R11 | Server (M3) | update_project() + torch.save() | New .pt file saved; DB updated with new round, model path, metrics |
| R12 | Server (M3) | SSE broadcast to dashboard | Dashboard shows new round complete + metrics |
| R13 | Each Client (M4) | update_global_metrics(axes, round_history) | Client metric dashboard re-draws convergence curves |

# 10\. Complete REST API Contract

All endpoints require HTTPS. Authenticated endpoints require 'Authorization: Bearer <token>' header except /auth/\* endpoints.

| Method + Path | Auth Required | Request Body | Response (Success) |
| --- | --- | --- | --- |
| POST /api/auth/register | No | {username, password, hospital_name, contact_email} | 201: {user_id, username} |
| POST /api/auth/login | No | {username, password} | 200: {access_token, token_type, approved_projects} |
| GET /api/status | No | — | 200: {status: 'ok', ngrok_url, server_version} |
| GET /api/projects | Bearer JWT | — | 200: list of project objects with schemas |
| POST /api/projects/{proj_id}/join | Bearer JWT | {hardware_profile: {ram_gb, cpu_cores, gpu_available, local_data_size}} | 200: {status, recommended_depth, required_schema, schema_version} |
| GET /api/projects/{proj_id}/model | Bearer JWT | — | 200: {round, active_depth, weights: {param: [floats]}} |
| POST /api/projects/{proj_id}/update | Bearer JWT | {round_id, active_depth, weights (encrypted), num_samples, metrics} | 202: {status, clients_submitted, clients_expected, aggregation_triggered} |
| GET /api/projects/{proj_id}/history | Bearer JWT | — | 200: list of RoundRecord objects (for client metrics dashboard) |
| GET /dashboard | Cookie session | — | 200: HTML Jinja2 dashboard page |
| GET /api/projects/{proj_id}/approve/{user_id} | Admin cookie | — | 200: moves user from pending to connected |

# 11\. Error Handling & Edge Case Contracts

Every module-to-module interface must handle failure modes gracefully. The following table defines the expected behavior at each failure point:

| Failure Point | Error Type | Handling Module | Recovery Action |
| --- | --- | --- | --- |
| Client CSV has missing columns | SchemaValidationError | M4 schema_validator.py | Display error list in client UI; block join_project() call until resolved |
| Client disconnects mid-round | Missing update in buffer | M3 project_router.py | Wait min_clients timeout; proceed with available updates if >= min_clients_per_round |
| ngrok tunnel drops | ConnectionError on client | M4 APIClient (retry logic) | HTTPAdapter retries 3x with backoff; if persistent, display 'Server unreachable' in UI |
| Gradient explosion in M1 | Loss becomes NaN/Inf | M1 compute_joint_loss() | clip_grad_norm_() prevents; if NaN detected, skip weight update for that batch |
| DB write fails (disk full) | IOError in write_db() | M3 db_handler.py | Raise ServerStorageError; return 503 to client; alert server dashboard |
| Client sends wrong round update | round_id mismatch | M3 post_model_update() | Return 409 Conflict with expected round_id; client must re-fetch global model |
| FedAvg receives no updates | Empty list to aggregate_fedavg() | M2 aggregation.py | Raise EmptyRoundError; skip round increment; notify dashboard |
| Schema version mismatch | schema_version != expected | M3 join_project() | Return 422 Unprocessable with schema diff; client must update data |

# 12\. References

\[1\] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS, 2017.

\[2\] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated Optimization in Heterogeneous Networks," Proceedings of Machine Learning and Systems, vol. 2, pp. 429-450, 2020.

\[3\] H. Cai, C. Gan, T. Wang, Z. Zhang, and S. Han, "Once-for-All: Train One Network and Specialize it for Efficient Deployment," in International Conference on Learning Representations (ICLR), 2020.

\[4\] A. Khare et al., "Cost-Efficient Federated Neural Architecture Search for On-Device," in European Conference on Computer Vision (ECCV), 2024.

— End of Architectural Design Document —