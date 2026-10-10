"""Tests for Part 3: Embeddings learn similarity from next-token."""

from src.embeddings import (
    NextTokenEmbeddingModel,
    build_synthetic_corpus,
    compute_knn_category_purity,
    create_training_pairs,
    train_embedding_model,
)


def test_synthetic_corpus_and_vocabulary():
    sentences, word2idx, idx2word = build_synthetic_corpus(n_sentences=500, seed=42)
    assert len(sentences) == 500
    assert len(word2idx) == len(idx2word)
    assert "cat" in word2idx
    assert "apple" in word2idx


def test_embedding_clustering_purity():
    sentences, word2idx, idx2word = build_synthetic_corpus(n_sentences=2500, seed=42)
    inputs, targets = create_training_pairs(sentences, word2idx)

    model = NextTokenEmbeddingModel(vocab_size=len(word2idx), embed_dim=16)
    final_loss = train_embedding_model(model, inputs, targets, epochs=140, lr=0.03, seed=42)

    weights = model.embedding.weight.detach().cpu().numpy()
    purity = compute_knn_category_purity(weights, word2idx, k=2)

    # Cross-entropy loss must decrease
    assert final_loss < 2.0
    # Emergent clustering purity should exceed 80% (random baseline is ~33%)
    assert purity >= 0.80, f"Purity {purity} did not reach expected threshold"
