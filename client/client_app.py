"""
client/client_app.py
Client entry point — launches local FL training loop.
Owner: Nikhil Garuda (M4 orchestration)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import json
import argparse
import threading

from api_client import APIClient, ServerUnreachableError, AuthError
from supernet import Supernet, load_global_weights
from train_loop import run_local_training, TrainConfig
from data_loader import build_dataloaders_from_csv
from schema_validator import validate_schema, format_validation_report
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA, DEFAULT_BATCH_SIZE


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Federated Learning Client")
    p.add_argument("--server",    required=True, help="ngrok server URL, e.g. https://xxx.ngrok.io")
    p.add_argument("--username",  required=True)
    p.add_argument("--password",  required=True)
    p.add_argument("--hospital",  required=True, help="Hospital display name")
    p.add_argument("--email",     required=True, help="Contact email")
    p.add_argument("--csv",       required=True, help="Path to local TCGA CSV file")
    p.add_argument("--proj",      required=True, help="Project ID to join/participate in")
    p.add_argument("--ram",       type=float, default=8.0)
    p.add_argument("--cores",     type=int,   default=4)
    p.add_argument("--gpu",       action="store_true")
    p.add_argument("--no-ui",     action="store_true", help="Disable matplotlib dashboard")
    return p.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    client = APIClient(args.server)

    # ── Connectivity check ────────────────────────────────────────────────────
    print(f"[*] Connecting to server: {args.server}")
    try:
        status = client.check_status()
        print(f"[+] Server OK — version {status.get('server_version', '?')}")
    except ServerUnreachableError as e:
        print(f"[!] Cannot reach server: {e}")
        sys.exit(1)

    # ── Auth ──────────────────────────────────────────────────────────────────
    print("[*] Attempting login…")
    try:
        result = client.login(args.username, args.password)
        print(f"[+] Logged in as {args.username}")
    except AuthError:
        print("[*] Login failed — attempting registration…")
        try:
            client.register(args.username, args.password, args.hospital, args.email)
            client.login(args.username, args.password)
            print(f"[+] Registered and logged in as {args.username}")
        except Exception as e:
            print(f"[!] Auth failed: {e}")
            sys.exit(1)

    # ── Schema Validation ─────────────────────────────────────────────────────
    print(f"[*] Validating CSV: {args.csv}")
    import pandas as pd
    df_check = pd.read_csv(args.csv, low_memory=False, nrows=500)
    val_result = validate_schema(df_check, SERVER_SCHEMA)
    print(format_validation_report(val_result))
    if not val_result.passed:
        print("[!] Schema validation failed. Please fix errors before joining.")
        sys.exit(1)

    # ── Join Project ──────────────────────────────────────────────────────────
    hw_profile = {
        "ram_gb": args.ram,
        "cpu_cores": args.cores,
        "gpu_available": args.gpu,
        "local_data_size": df_check.shape[0],
    }
    print(f"[*] Joining project: {args.proj}")
    try:
        join_resp = client.join_project(args.proj, hw_profile)
        active_depth = join_resp.get("recommended_depth", 4)
        schema = join_resp.get("required_schema", SERVER_SCHEMA)
        print(f"[+] Join request submitted. Recommended depth: {active_depth}")
        print(f"[*] Status: {join_resp.get('status', '?')} — waiting for admin approval…")
    except Exception as e:
        print(f"[!] Join failed: {e}")
        sys.exit(1)

    # ── Wait for approval ─────────────────────────────────────────────────────
    print("[*] Polling for approval (press Ctrl-C to abort)…")
    approved = False
    while not approved:
        try:
            projects = client.list_projects()
            proj = next((p for p in projects if p.get("proj_id") == args.proj), None)
            if proj and proj.get("i_am_connected"):
                approved = True
                break
            # Fallback: try fetching model — 403 means still pending, 200 means approved
            model_resp = client.fetch_global_model(args.proj)
            approved = True
        except AuthError:
            pass  # still pending
        except Exception:
            pass
        if not approved:
            time.sleep(10)

    print("[+] Approved! Starting federated training loop…")

    # ── Build model ───────────────────────────────────────────────────────────
    supernet = Supernet(**MODEL_CONFIG)

    # ── Init visualizer ───────────────────────────────────────────────────────
    fig, axes = None, None
    if not args.no_ui:
        try:
            from visualizer import init_metrics_dashboard
            fig, axes = init_metrics_dashboard()
        except Exception as e:
            print(f"[!] Visualizer init failed: {e}. Running headless.")

    # ── Federated training loop ───────────────────────────────────────────────
    round_history = []
    while True:
        # R1: Fetch global model
        print("[*] Fetching global model…")
        model_data = client.fetch_global_model(args.proj)
        current_round = model_data["round"]
        active_depth  = model_data.get("active_depth", active_depth)
        global_weights = model_data["weights"]
        print(f"[+] Round {current_round} | Active depth: {active_depth}")

        # R2: Load global weights
        load_global_weights(supernet, global_weights, strict=False)

        # R3: Load & preprocess data
        print("[*] Preparing data loaders…")
        train_loader, val_loader = build_dataloaders_from_csv(
            args.csv, schema, batch_size=DEFAULT_BATCH_SIZE
        )

        # R4: Local training
        print("[*] Starting local training…")
        cfg = TrainConfig(active_depth=active_depth)
        result = run_local_training(supernet, (train_loader, val_loader), cfg, axes=axes)
        print(f"[+] Training done | Loss: {result['metrics']['loss']:.4f} | "
              f"RMSE: {result['metrics']['val_rmse']:.4f} | "
              f"ToxAcc: {result['metrics']['val_acc_tox']:.4f} | "
              f"AUC: {result['metrics']['val_auc']:.4f}")

        # R6: Post update
        print("[*] Posting model update to server…")
        post_resp = client.post_local_update(
            proj_id=args.proj,
            weights=result["weights"],
            num_samples=result["num_samples"],
            metrics=result["metrics"],
            round_id=current_round,
            active_depth=active_depth,
        )
        print(f"[+] Update received. Clients submitted: "
              f"{post_resp.get('clients_submitted')}/{post_resp.get('clients_expected')} | "
              f"Aggregation triggered: {post_resp.get('aggregation_triggered')}")

        # R13: Update dashboard
        round_history = client.get_round_history(args.proj)
        if axes and round_history:
            try:
                from visualizer import update_global_metrics
                update_global_metrics(axes, round_history)
            except Exception:
                pass

        # Wait before next round
        print(f"[*] Waiting for next round… (sleeping 5s)")
        time.sleep(5)


if __name__ == "__main__":
    main()
