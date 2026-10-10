"""Tests for Part 2: Depth without nonlinearity is a lie."""

import torch

from src.activations import generate_ring_data, train_classifier
from src.linear_depth import (
    FiveLayerLinear,
    FiveLayerReLU,
    OneLayerLinear,
    collapse_linear_network,
    verify_linear_collapse,
)


def test_linear_depth_collapse_exactness():
    """Verify that a 5-layer linear network mathematically collapses to a single matrix."""
    model = FiveLayerLinear(hidden_dim=16)
    test_points = torch.randn(100, 2)

    max_err = verify_linear_collapse(model, test_points)
    assert max_err < 1e-5, f"Numerical collapse error too large: {max_err}"

    w_eff, b_eff = collapse_linear_network(model)
    assert w_eff.shape == (1, 2)
    assert b_eff.shape == (1,)


def test_depth_with_and_without_relu():
    """Verify 5-layer linear fails like 1-layer, but 5-layer ReLU succeeds."""
    X, y = generate_ring_data(n_samples=300, noise=0.08, factor=0.45, random_state=42)

    m1_linear = OneLayerLinear()
    m5_linear = FiveLayerLinear(hidden_dim=16)
    m5_relu = FiveLayerReLU(hidden_dim=16)

    acc1 = train_classifier(m1_linear, X, y, epochs=250, lr=0.05, seed=42)
    acc5_lin = train_classifier(m5_linear, X, y, epochs=250, lr=0.02, seed=42)
    acc5_relu = train_classifier(m5_relu, X, y, epochs=300, lr=0.02, seed=42)

    # Both linear models fail
    assert acc1 < 0.65
    assert acc5_lin < 0.65

    # 5-layer ReLU solves it
    assert acc5_relu >= 0.95
