"""Part 1: Activations exist for a reason.

Claim: A model with no nonlinearity can only draw a straight boundary,
so it cannot separate two interleaved/concentric rings; adding one ReLU hidden
layer can.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_circles


def generate_ring_data(
    n_samples: int = 300, noise: float = 0.08, factor: float = 0.45, random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate concentric ring dataset with inner class 0 and outer class 1."""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return X_t, y_t


class LinearModel(nn.Module):
    """Single linear layer mapping R^2 -> R (no hidden nonlinearity)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class OneHiddenLayerReLUModel(nn.Module):
    """Model with one hidden layer followed by ReLU nonlinearity."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 400,
    lr: float = 0.05,
    seed: int = 42,
) -> float:
    """Train a binary classifier using BCEWithLogitsLoss and return final accuracy."""
    torch.manual_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) >= 0.5).float()
        acc = (preds == y).float().mean().item()
    return acc


def plot_decision_boundary(
    ax: plt.Axes,
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    title: str,
    acc: float,
):
    """Plot the 2D decision boundary alongside ground truth points."""
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(
        np.linspace(x_min.item(), x_max.item(), 300),
        np.linspace(y_min.item(), y_max.item(), 300),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()

    # Probability contour
    contour = ax.contourf(xx, yy, probs, levels=25, cmap="coolwarm", alpha=0.35)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")

    # Scatter points
    X_np = X.numpy()
    y_np = y.squeeze().numpy()
    ax.scatter(
        X_np[y_np == 0, 0],
        X_np[y_np == 0, 1],
        c="#1f77b4",
        edgecolors="k",
        s=40,
        label="Class 0 (Inner)",
        alpha=0.9,
    )
    ax.scatter(
        X_np[y_np == 1, 0],
        X_np[y_np == 1, 1],
        c="#d62728",
        edgecolors="k",
        s=40,
        label="Class 1 (Outer)",
        alpha=0.9,
    )

    ax.set_title(f"{title}\nAccuracy: {acc * 100:.1f}%", fontsize=12, fontweight="bold")
    ax.set_xlabel("$x_1$", fontsize=11)
    ax.set_ylabel("$x_2$", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    return contour


def run_experiment_1(save_path: str = "assets/01_activations_boundary.png") -> Tuple[float, float]:
    """Run Part 1 experiment: train linear & ReLU models and save comparison plot."""
    X, y = generate_ring_data(n_samples=300, noise=0.08, factor=0.45, random_state=42)

    linear_model = LinearModel()
    relu_model = OneHiddenLayerReLUModel(hidden_dim=32)

    linear_acc = train_classifier(linear_model, X, y, epochs=300, lr=0.05)
    relu_acc = train_classifier(relu_model, X, y, epochs=350, lr=0.05)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    plot_decision_boundary(
        axes[0],
        linear_model,
        X,
        y,
        title="Single Linear Layer + Sigmoid\n(Only Straight Separators Possible)",
        acc=linear_acc,
    )
    plot_decision_boundary(
        axes[1],
        relu_model,
        X,
        y,
        title="One ReLU Hidden Layer (32 units)\n(Curved Boundary Wraps Ring)",
        acc=relu_acc,
    )

    fig.suptitle(
        "Proof 1: Activations Exist for a Reason (Linear vs. ReLU on Concentric Rings)",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=300)
    plt.close()
    return linear_acc, relu_acc
