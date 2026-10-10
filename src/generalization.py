"""Part 4: Memorization vs generalization, and data closes the gap.

Claim: A high-capacity model on tiny data drives train loss to ~0 while held-out loss
stays high; growing the dataset closes the gap.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_classification


def generate_generalization_data(
    n_train_max: int = 2000,
    n_test: int = 1500,
    n_features: int = 25,
    n_informative: int = 8,
    flip_y: float = 0.05,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate noisy high-dimensional classification data with fixed held-out test split."""
    total_samples = n_train_max + n_test
    X, y = make_classification(
        n_samples=total_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=4,
        flip_y=flip_y,
        random_state=random_state,
    )

    X_train = torch.tensor(X[:n_train_max], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train_max], dtype=torch.float32).unsqueeze(1)

    X_test = torch.tensor(X[n_train_max:], dtype=torch.float32)
    y_test = torch.tensor(y[n_train_max:], dtype=torch.float32).unsqueeze(1)

    return X_train, y_train, X_test, y_test


class OverparameterizedMLP(nn.Module):
    """High-capacity multi-layer network with sufficient capacity to memorize noise."""

    def __init__(self, in_features: int = 25, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_evaluate(
    X_train_sub: torch.Tensor,
    y_train_sub: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 250,
    lr: float = 0.003,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train the model on subset and record train/test dynamics over epochs."""
    torch.manual_seed(seed)
    model = OverparameterizedMLP(in_features=X_train_sub.shape[1], hidden_dim=256)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits_train = model(X_train_sub)
        loss_train = criterion(logits_train, y_train_sub)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_test = model(X_test)
            loss_test = criterion(logits_test, y_test)

            train_preds = (torch.sigmoid(logits_train) >= 0.5).float()
            test_preds = (torch.sigmoid(logits_test) >= 0.5).float()

            train_acc = (train_preds == y_train_sub).float().mean().item()
            test_acc = (test_preds == y_test).float().mean().item()

        history["train_loss"].append(loss_train.item())
        history["test_loss"].append(loss_test.item())
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

    return history


def run_experiment_4(save_path: str = "assets/04_generalization_gap.png") -> Dict[int, Dict[str, float]]:
    """Run Part 4 across dataset sizes N in [20, 200, 2000]."""
    X_tr_full, y_tr_full, X_te, y_te = generate_generalization_data(
        n_train_max=2000, n_test=1500, n_features=25, n_informative=8, flip_y=0.05, random_state=42
    )

    sizes = [20, 200, 2000]
    histories = {}
    summary = {}

    for n in sizes:
        X_sub = X_tr_full[:n]
        y_sub = y_tr_full[:n]
        hist = train_and_evaluate(X_sub, y_sub, X_te, y_te, epochs=250, lr=0.003, seed=42)
        histories[n] = hist
        summary[n] = {
            "final_train_loss": hist["train_loss"][-1],
            "final_test_loss": hist["test_loss"][-1],
            "final_train_acc": hist["train_acc"][-1],
            "final_test_acc": hist["test_acc"][-1],
            "generalization_gap": hist["test_loss"][-1] - hist["train_loss"][-1],
        }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Subplot 1: Learning curves over epochs
    colors = {20: "#d62728", 200: "#ff7f0e", 2000: "#2ca02c"}
    for n in sizes:
        axes[0].plot(
            histories[n]["train_loss"],
            linestyle="--",
            color=colors[n],
            label=f"N={n} (Train Loss)",
            alpha=0.85,
        )
        axes[0].plot(
            histories[n]["test_loss"],
            linestyle="-",
            color=colors[n],
            linewidth=2,
            label=f"N={n} (Test Loss)",
        )

    axes[0].set_title("Loss Curves over Training (Solid = Test, Dashed = Train)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary Cross Entropy Loss", fontsize=11)
    axes[0].set_yscale("log")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Subplot 2: Generalization gap vs dataset size
    gaps = [summary[n]["generalization_gap"] for n in sizes]
    train_accs = [summary[n]["final_train_acc"] * 100 for n in sizes]
    test_accs = [summary[n]["final_test_acc"] * 100 for n in sizes]

    axes[1].plot(sizes, gaps, marker="o", color="#d62728", linewidth=2.5, markersize=8, label="Generalization Gap (Test - Train Loss)")
    axes[1].set_xscale("log")
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels([f"N={n}" for n in sizes])
    axes[1].set_title("Generalization Gap vs. Training Set Size", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Training Set Size (Log Scale)", fontsize=11)
    axes[1].set_ylabel(r"Generalization Gap ($\mathcal{L}_{test} - \mathcal{L}_{train}$)", fontsize=11, color="#d62728")
    axes[1].tick_params(axis="y", labelcolor="#d62728")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Annotations for accuracy
    for i, n in enumerate(sizes):
        axes[1].annotate(
            f"Train Acc: {train_accs[i]:.0f}%\nTest Acc: {test_accs[i]:.0f}%\nGap: {gaps[i]:.2f}",
            (sizes[i], gaps[i]),
            xytext=(10, 15 if i != 0 else -35),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    fig.suptitle(
        "Proof 4: Memorization vs. Generalization (Data Closes the Gap)",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=300)
    plt.close()

    return summary
