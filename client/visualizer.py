"""
client/visualizer.py
M4: Matplotlib dashboard for real-time training metrics display.
Owner: Nikhil Garuda
"""

import matplotlib
matplotlib.use("TkAgg")  # fallback: safe for most desktops; override via env if needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Dashboard Initialisation ────────────────────────────────────────────────

def init_metrics_dashboard() -> tuple:
    """
    Create a persistent Matplotlib figure with four subplots (2×2 grid):
      (1) Global Validation RMSE over rounds
      (2) Toxicity Classification Accuracy over rounds
      (3) AUC-ROC over rounds
      (4) Local training loss per epoch in the current round

    Returns
    -------
    tuple: (fig: plt.Figure, axes: dict)
        axes keys: 'rmse', 'tox_acc', 'auc', 'local_loss'
    """
    plt.ion()  # enable interactive (non-blocking) mode
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Federated Learning – Live Training Dashboard", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_rmse      = fig.add_subplot(gs[0, 0])
    ax_tox_acc   = fig.add_subplot(gs[0, 1])
    ax_auc       = fig.add_subplot(gs[1, 0])
    ax_local_loss = fig.add_subplot(gs[1, 1])

    # Initial styling
    _style_ax(ax_rmse,       "Global Val RMSE",        "Round",   "RMSE")
    _style_ax(ax_tox_acc,    "Toxicity Accuracy",       "Round",   "Accuracy")
    _style_ax(ax_auc,        "AUC-ROC (Binary Head)",   "Round",   "AUC")
    _style_ax(ax_local_loss, "Local Training Loss",     "Epoch",   "Loss")

    plt.tight_layout()
    plt.pause(0.01)

    axes = {
        "rmse": ax_rmse,
        "tox_acc": ax_tox_acc,
        "auc": ax_auc,
        "local_loss": ax_local_loss,
    }
    return fig, axes


# ─── Global Metrics Update ───────────────────────────────────────────────────

def update_global_metrics(axes: dict, round_history: list) -> None:
    """
    Redraw the three global metric subplots with the full round history.

    Parameters
    ----------
    axes          : dict — from init_metrics_dashboard()
    round_history : list[dict] — each dict has keys:
                    'round', 'global_val_rmse', 'global_tox_accuracy', 'global_auc'
    """
    if not round_history:
        return

    rounds    = [r["round"] for r in round_history]
    rmse_vals = [r.get("global_val_rmse", 0) for r in round_history]
    tox_vals  = [r.get("global_tox_accuracy", 0) for r in round_history]
    auc_vals  = [r.get("global_auc", 0) for r in round_history]

    _redraw_ax(axes["rmse"],    rounds, rmse_vals, "RMSE",           "Global Val RMSE",       "royalblue")
    _redraw_ax(axes["tox_acc"], rounds, tox_vals,  "Accuracy",       "Toxicity Accuracy",     "darkorange")
    _redraw_ax(axes["auc"],     rounds, auc_vals,  "AUC",            "AUC-ROC (Binary Head)", "seagreen")

    plt.pause(0.05)


# ─── Local Loss Update ───────────────────────────────────────────────────────

def update_local_loss(axes: dict, epoch_losses: list) -> None:
    """
    Update the local training loss subplot in real-time during a training round.

    Parameters
    ----------
    axes         : dict       — from init_metrics_dashboard()
    epoch_losses : list[float] — per-epoch mean losses accumulated so far
    """
    if not epoch_losses:
        return

    epochs = list(range(1, len(epoch_losses) + 1))
    _redraw_ax(
        axes["local_loss"], epochs, epoch_losses,
        ylabel="Loss",
        title=f"Local Training Loss  (latest: {epoch_losses[-1]:.4f})",
        color="crimson",
        marker="o",
    )
    plt.pause(0.01)


# ─── Internal Helpers ─────────────────────────────────────────────────────────

def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.text(
        0.5, 0.5, "Waiting for data…",
        ha="center", va="center",
        transform=ax.transAxes,
        color="gray", fontsize=8,
    )


def _redraw_ax(ax, x, y, ylabel, title, color="royalblue", marker=None):
    ax.cla()
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", alpha=0.4)
    kwargs = {"color": color, "linewidth": 1.8}
    if marker:
        kwargs["marker"] = marker
        kwargs["markersize"] = 4
    ax.plot(x, y, **kwargs)
