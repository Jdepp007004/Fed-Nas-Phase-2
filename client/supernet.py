"""
client/supernet.py
M1: Core Supernet & Multi-Task ML model definition.
Owner: Praneeth Raj V
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from shared.model_schema import INPUT_DIM, MAX_DEPTH, HIDDEN_DIM, NUM_TOXICITY_CLASSES  # noqa: E402


# ─── Supernet ────────────────────────────────────────────────────────────────

class Supernet(nn.Module):
    """
    Depth-flexible neural network with three multi-task output heads.

    Backbone: ModuleList of (max_depth) blocks, each block = Linear + BatchNorm1d + ReLU.
    Heads:
      - head_regression: predicts continuous survival days (regression)
      - head_toxicity:   predicts toxicity grade (multi-class classification)
      - head_binary:     predicts vital status (binary classification)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        max_depth: int = MAX_DEPTH,
        hidden_dim: int = HIDDEN_DIM,
        num_toxicity_classes: int = NUM_TOXICITY_CLASSES,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_depth = max_depth
        self.hidden_dim = hidden_dim
        self.num_toxicity_classes = num_toxicity_classes

        # Store config metadata
        self.config = {
            "input_dim": input_dim,
            "max_depth": max_depth,
            "hidden_dim": hidden_dim,
            "num_toxicity_classes": num_toxicity_classes,
        }

        # Build shared backbone as a ModuleList of depth blocks
        backbone_layers = []
        for i in range(max_depth):
            in_features = input_dim if i == 0 else hidden_dim
            backbone_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.backbone = nn.ModuleList(backbone_layers)

        # Three independent output heads
        self.head_regression = nn.Linear(hidden_dim, 1)
        self.head_toxicity = nn.Linear(hidden_dim, num_toxicity_classes)
        self.head_binary = nn.Linear(hidden_dim, 1)

    def forward_multi_head(self, x: torch.Tensor, active_depth: int) -> dict:
        """
        Forward pass through the first `active_depth` backbone layers,
        then simultaneously through all three heads.

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch_size, input_dim).
        active_depth : int
            Number of backbone layers to activate (1 <= active_depth <= max_depth).

        Returns
        -------
        dict with keys: 'regression', 'toxicity', 'binary'
        """
        if active_depth < 1 or active_depth > self.max_depth:
            raise ValueError(
                f"active_depth={active_depth} out of range [1, {self.max_depth}]"
            )

        embedding = x
        for i in range(active_depth):
            embedding = self.backbone[i](embedding)

        regression = self.head_regression(embedding)          # (B, 1)
        toxicity = self.head_toxicity(embedding)              # (B, num_classes)
        binary = self.head_binary(embedding)                  # (B, 1)

        return {
            "regression": regression,
            "toxicity": toxicity,
            "binary": binary,
        }

    def forward(self, x: torch.Tensor) -> dict:
        """Convenience: forward at full depth."""
        return self.forward_multi_head(x, self.max_depth)


# ─── Loss ────────────────────────────────────────────────────────────────────

def compute_joint_loss(
    predictions: dict,
    targets: dict,
    weights: dict,
) -> tuple:
    """
    Combine three task losses into a single scalar.

    Parameters
    ----------
    predictions : dict  — output of forward_multi_head()
    targets     : dict  — {'regression': Tensor, 'toxicity': LongTensor, 'binary': Tensor}
    weights     : dict  — {'regression': float, 'toxicity': float, 'binary': float}

    Returns
    -------
    tuple: (total_loss: Tensor, breakdown: dict with float values)
    """
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    loss_reg = mse(predictions["regression"].squeeze(1), targets["regression"].float())
    loss_tox = ce(predictions["toxicity"], targets["toxicity"].long())
    loss_bin = bce(predictions["binary"].squeeze(1), targets["binary"].float())

    total_loss = (
        weights.get("regression", 1.0) * loss_reg
        + weights.get("toxicity", 0.8) * loss_tox
        + weights.get("binary", 0.6) * loss_bin
    )

    # Guard against NaN/Inf
    if not torch.isfinite(total_loss):
        total_loss = torch.zeros(1, requires_grad=True)

    return total_loss, {
        "loss_reg": loss_reg.item(),
        "loss_tox": loss_tox.item(),
        "loss_bin": loss_bin.item(),
    }


# ─── Weight Extraction / Loading ─────────────────────────────────────────────

def get_subnet_weights(model: Supernet, active_depth: int) -> dict:
    """
    Extract only the active subnet parameters as a dict of CPU numpy arrays.

    Parameters
    ----------
    model        : Supernet
    active_depth : int — which backbone layers were active this round

    Returns
    -------
    dict  {param_name: np.ndarray}
    """
    state = model.state_dict()
    subnet = {}

    # Backbone layers 0..active_depth-1
    for i in range(active_depth):
        # Each backbone[i] is nn.Sequential with sub-keys: 0.weight, 0.bias, 1.*
        prefix = f"backbone.{i}."
        for k, v in state.items():
            if k.startswith(prefix):
                subnet[k] = v.cpu().numpy()

    # All three heads
    for head_name in ("head_regression", "head_toxicity", "head_binary"):
        prefix = f"{head_name}."
        for k, v in state.items():
            if k.startswith(prefix):
                subnet[k] = v.cpu().numpy()

    return subnet


def load_global_weights(model: Supernet, weights: dict, strict: bool = False) -> None:
    """
    Load server-provided global weights (dict of numpy arrays) into the model.

    Parameters
    ----------
    model   : Supernet
    weights : dict  {param_name: np.ndarray}
    strict  : bool  — default False allows partial loading for subnets
    """
    tensor_dict = {k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(tensor_dict, strict=strict)
    # In non-strict mode, missing keys are expected (deeper layers not sent)
