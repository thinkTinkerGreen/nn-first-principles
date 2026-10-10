"""Part 2: Depth without nonlinearity is a lie.

Claim: Five stacked linear layers collapse to a single linear map,
so a 5-layer linear net is no stronger than 1 layer; both fail the ring task identically,
and inserting ReLUs between the same five layers suddenly solves it.
Bonus proof: Multiply the five weight matrices numerically and show the product is one matrix.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.activations import generate_ring_data, plot_decision_boundary, train_classifier


class OneLayerLinear(nn.Module):
    """1-layer linear classifier: R^2 -> R."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class FiveLayerLinear(nn.Module):
    """5 stacked linear layers with zero activations in between."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiveLayerReLU(nn.Module):
    """5 linear layers with ReLU activations interleaved."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collapse_linear_network(model: FiveLayerLinear) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Numerically compute the single equivalent weight matrix W_eff and bias b_eff.

    In PyTorch, nn.Linear applies: y = x W^T + b.
    For composition f_k(x) = f_{k-1}(x) W_k^T + b_k:
    We iteratively maintain (W_eff, b_eff) such that y = x W_eff^T + b_eff.
    Initially: W_eff = W_1, b_eff = b_1.
    Next layer: y_2 = (x W_eff^T + b_eff) W_2^T + b_2 = x (W_2 W_eff)^T + (b_eff W_2^T + b_2).
    """
    linear_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]

    # Start with first layer
    w_eff = linear_layers[0].weight.detach().clone()  # shape (out_1, in)
    b_eff = linear_layers[0].bias.detach().clone()    # shape (out_1,)

    for layer in linear_layers[1:]:
        w = layer.weight.detach().clone()  # shape (out_k, in_k)
        b = layer.bias.detach().clone()    # shape (out_k,)
        # Effective update:
        w_eff = torch.matmul(w, w_eff)
        b_eff = torch.matmul(b_eff, w.t()) + b

    return w_eff, b_eff


def verify_linear_collapse(model: FiveLayerLinear, test_points: torch.Tensor) -> float:
    """Verify max absolute difference between 5-layer execution and single collapsed map."""
    w_eff, b_eff = collapse_linear_network(model)

    model.eval()
    with torch.no_grad():
        out_model = model(test_points)
        out_collapsed = torch.matmul(test_points, w_eff.t()) + b_eff

    max_diff = torch.max(torch.abs(out_model - out_collapsed)).item()
    return max_diff


def run_experiment_2(save_path: str = "assets/02_depth_collapse.png") -> Tuple[float, float, float, float]:
    """Run Part 2 experiment: 1-layer vs 5-layer linear vs 5-layer ReLU."""
    X, y = generate_ring_data(n_samples=300, noise=0.08, factor=0.45, random_state=42)

    m1_linear = OneLayerLinear()
    m5_linear = FiveLayerLinear(hidden_dim=16)
    m5_relu = FiveLayerReLU(hidden_dim=16)

    acc1 = train_classifier(m1_linear, X, y, epochs=300, lr=0.05, seed=42)
    acc5_lin = train_classifier(m5_linear, X, y, epochs=300, lr=0.02, seed=42)
    acc5_relu = train_classifier(m5_relu, X, y, epochs=350, lr=0.02, seed=42)

    # Numerical collapse proof
    max_error = verify_linear_collapse(m5_linear, X)
    w_eff, b_eff = collapse_linear_network(m5_linear)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    plot_decision_boundary(
        axes[0],
        m1_linear,
        X,
        y,
        title="1-Layer Linear (No Activation)\nSingle Linear Boundary",
        acc=acc1,
    )
    plot_decision_boundary(
        axes[1],
        m5_linear,
        X,
        y,
        title=f"5-Layer Linear (No Activation)\nCollapsed Line (Max Diff: {max_error:.1e})",
        acc=acc5_lin,
    )
    plot_decision_boundary(
        axes[2],
        m5_relu,
        X,
        y,
        title="5-Layer with ReLU Activations\nNonlinear Solution Achieved",
        acc=acc5_relu,
    )

    fig.suptitle(
        f"Proof 2: Depth Without Nonlinearity is a Lie\n"
        f"Numerical Matrix Product Collapse: W_eff shape {tuple(w_eff.shape)}, Max Absolute Error = {max_error:.2e}",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(save_path, dpi=300)
    plt.close()

    return acc1, acc5_lin, acc5_relu, max_error
