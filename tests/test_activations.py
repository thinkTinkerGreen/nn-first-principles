"""Tests for Part 1: Activations exist for a reason."""

from src.activations import (
    LinearModel,
    OneHiddenLayerReLUModel,
    generate_ring_data,
    train_classifier,
)


def test_ring_dataset_shapes():
    X, y = generate_ring_data(n_samples=300)
    assert X.shape == (300, 2)
    assert y.shape == (300, 1)
    assert set(y.squeeze().tolist()).issubset({0.0, 1.0})


def test_linear_vs_relu_performance():
    X, y = generate_ring_data(n_samples=300, noise=0.08, factor=0.45, random_state=42)

    linear_model = LinearModel()
    relu_model = OneHiddenLayerReLUModel(hidden_dim=32)

    acc_linear = train_classifier(linear_model, X, y, epochs=250, lr=0.05, seed=42)
    acc_relu = train_classifier(relu_model, X, y, epochs=300, lr=0.05, seed=42)

    # Linear classifier cannot separate concentric rings (~50% chance baseline)
    assert acc_linear < 0.65, f"Linear accuracy unexpectedly high: {acc_linear}"

    # Single ReLU hidden layer wraps the ring to near-perfection
    assert acc_relu >= 0.95, f"ReLU accuracy did not reach target: {acc_relu}"
