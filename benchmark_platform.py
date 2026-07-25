"""Reproducible benchmark of this repository's clinical FL implementation.

It evaluates the *actual* client Supernet/FedProx loop and server FedAvg +
momentum aggregation on data/client_1.csv through client_4.csv.  The test set
is made from the held-out partition of every silo; it is never used to train
any method.

Example (quick, suitable before a demo):
    python benchmark_platform.py --rounds 6 --local-epochs 1

For a reportable run, use several seeds and more optimisation budget:
    python benchmark_platform.py --rounds 20 --local-epochs 3 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "client"))
sys.path.insert(0, str(ROOT / "server"))

from aggregation import aggregate_fedavg, update_with_momentum, validate_global_model
from data_loader import build_dataloaders_from_csv
from supernet import Supernet, load_global_weights
from train_loop import TrainConfig, run_local_training
from shared.model_schema import MODEL_CONFIG, SERVER_SCHEMA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metric_row(name: str, y_true, probabilities, extra: dict | None = None) -> dict:
    probabilities = np.asarray(probabilities)
    prediction = (probabilities >= 0.5).astype(int)
    row = {
        "method": name,
        "auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }
    if extra:
        row.update(extra)
    return row


def datasets():
    """Use each real device CSV and preserve the project's local preprocessing."""
    train_sets, test_sets = [], []
    for path in sorted((ROOT / "data").glob("client_[1-4].csv")):
        train, test = build_dataloaders_from_csv(str(path), SERVER_SCHEMA, batch_size=64)
        train_sets.append(train.dataset)
        test_sets.append(test.dataset)
    if len(train_sets) != 4:
        raise FileNotFoundError("Expected data/client_1.csv through data/client_4.csv")
    return train_sets, test_sets


def arrays(dataset):
    # TensorDataset layout is (features, regression, toxicity, binary).
    parts = [dataset[i] for i in range(len(dataset))]
    x = torch.stack([p[0] for p in parts]).numpy()
    y = torch.stack([p[3] for p in parts]).numpy().astype(int)
    return x, y


class NormalCNN(nn.Module):
    """A deliberately conventional 1-D CNN baseline for the requested comparison."""
    def __init__(self, features: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):
        return self.head(self.net(x.unsqueeze(1)).squeeze(-1)).squeeze(-1)


def run_cnn(train_x, train_y, test_x, test_y, epochs: int, device) -> dict:
    model = NormalCNN(train_x.shape[1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y).float()),
                        batch_size=64, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward(); optimiser.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(test_x).to(device))).cpu().numpy()
    return metric_row("Centralised 1D CNN", test_y, probabilities)


def predict_supernet(weights: dict, test_loader) -> tuple[np.ndarray, np.ndarray]:
    model = Supernet(**MODEL_CONFIG)
    load_global_weights(model, weights, strict=False)
    model.eval()
    truth, probabilities = [], []
    with torch.no_grad():
        for x, _, _, y in test_loader:
            probabilities.extend(torch.sigmoid(model.forward_multi_head(x, MODEL_CONFIG["max_depth"])["binary"]).squeeze(1).numpy())
            truth.extend(y.numpy().astype(int))
    return np.asarray(truth), np.asarray(probabilities)


def run_federated(train_sets, test_sets, rounds: int, local_epochs: int, fedprox_mu: float, momentum: float, name: str):
    """Runs the repository's client training and server aggregation components."""
    initial_model = Supernet(**MODEL_CONFIG)
    global_weights = {key: value.detach().cpu().numpy().copy()
                      for key, value in initial_model.state_dict().items()}
    velocity = {}
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=64, shuffle=False)
    history = []
    for round_number in range(1, rounds + 1):
        updates, counts = [], []
        for local_data in train_sets:
            model = Supernet(**MODEL_CONFIG)
            if global_weights:
                load_global_weights(model, global_weights, strict=False)
            result = run_local_training(model, DataLoader(local_data, batch_size=64, shuffle=True), TrainConfig(
                epochs=local_epochs, active_depth=MODEL_CONFIG["max_depth"], fedprox_mu=fedprox_mu,
            ))
            updates.append(result["weights"]); counts.append(result["num_samples"])
        aggregate = aggregate_fedavg(updates, counts)
        global_weights, velocity = update_with_momentum(global_weights, aggregate, momentum, velocity)
        server_metrics = validate_global_model(global_weights, test_loader, MODEL_CONFIG)
        history.append({"round": round_number, **server_metrics})
    y, probabilities = predict_supernet(global_weights, test_loader)
    return metric_row(name, y, probabilities, {
        "global_rmse": history[-1]["global_val_rmse"],
        "global_toxicity_accuracy": history[-1]["global_tox_accuracy"],
        "round_history": history,
    })


def one_seed(seed: int, rounds: int, local_epochs: int, cnn_epochs: int) -> list[dict]:
    set_seed(seed)
    train_sets, test_sets = datasets()
    train_x, train_y = arrays(ConcatDataset(train_sets))
    test_x, test_y = arrays(ConcatDataset(test_sets))
    methods = []
    for name, estimator in (
        ("Logistic regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ("Random forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)),
        ("Histogram gradient boosting", HistGradientBoostingClassifier(random_state=seed)),
    ):
        estimator.fit(train_x, train_y)
        methods.append(metric_row(name, test_y, estimator.predict_proba(test_x)[:, 1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods.append(run_cnn(train_x, train_y, test_x, test_y, cnn_epochs, device))
    methods.append(run_federated(train_sets, test_sets, rounds, local_epochs, 0.0, 0.0, "FedAvg Supernet"))
    methods.append(run_federated(train_sets, test_sets, rounds, local_epochs, 0.01, 0.9, "FL Platform (FedProx + momentum)"))
    for result in methods:
        result["seed"] = seed
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()
    results = [row for seed in args.seeds for row in one_seed(seed, args.rounds, args.local_epochs, args.cnn_epochs)]
    Path(args.output).write_text(json.dumps({"config": vars(args), "results": results}, indent=2), encoding="utf-8")
    print("\nClinical survival prediction — identical held-out data across methods")
    print(f"{'Method':<36} {'AUC':>7} {'Accuracy':>10} {'Bal.Acc':>9} {'F1':>7}")
    for row in results:
        print(f"{row['method']:<36} {row['auc']:>7.3f} {row['accuracy']:>10.3f} {row['balanced_accuracy']:>9.3f} {row['f1']:>7.3f}")
    print(f"\nSaved full per-round results to {args.output}")


if __name__ == "__main__":
    main()
