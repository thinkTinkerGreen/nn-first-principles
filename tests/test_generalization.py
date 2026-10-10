"""Tests for Part 4: Memorization vs generalization."""

from src.generalization import (
    generate_generalization_data,
    train_and_evaluate,
)


def test_generalization_gap_decay():
    """Verify that larger training data reduces the generalization gap."""
    X_tr_full, y_tr_full, X_te, y_te = generate_generalization_data(
        n_train_max=1000, n_test=500, n_features=20, n_informative=8, flip_y=0.05, random_state=42
    )

    # Train small vs large
    h_small = train_and_evaluate(X_tr_full[:20], y_tr_full[:20], X_te, y_te, epochs=150, lr=0.003, seed=42)
    h_large = train_and_evaluate(X_tr_full[:1000], y_tr_full[:1000], X_te, y_te, epochs=150, lr=0.003, seed=42)

    gap_small = h_small["test_loss"][-1] - h_small["train_loss"][-1]
    gap_large = h_large["test_loss"][-1] - h_large["train_loss"][-1]

    # Memorization: train loss on small dataset is near zero
    assert h_small["train_loss"][-1] < 0.05

    # Generalization gap is significantly larger on small dataset
    assert gap_small > gap_large
    assert h_large["test_acc"][-1] > h_small["test_acc"][-1]
