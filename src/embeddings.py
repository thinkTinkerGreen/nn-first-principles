"""Part 3: Embeddings learn similarity from nothing but next-token.

Claim: Trained only to predict the next token in a tiny synthetic grammar,
the embedding table clusters related tokens, though similarity was never supplied.
"""

import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

# Synthetic vocabulary categorized semantically
CATEGORIES = {
    "animal": ["cat", "dog", "cow", "horse", "rabbit"],
    "fruit": ["apple", "mango", "banana", "peach", "orange"],
    "verb": ["eats", "chases", "likes", "bites", "watches"],
}


def build_synthetic_corpus(n_sentences: int = 4000, seed: int = 42) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Build synthetic sentences following grammar templates:
    Template 1: <animal> <verb> <fruit>
    Template 2: the <animal> will <verb>
    Template 3: the <fruit> was good
    """
    random.seed(seed)
    sentences = []

    for _ in range(n_sentences):
        r = random.random()
        a = random.choice(CATEGORIES["animal"])
        v = random.choice(CATEGORIES["verb"])
        f = random.choice(CATEGORIES["fruit"])

        if r < 0.40:
            sentences.append(f"{a} {v} {f}")
        elif r < 0.70:
            sentences.append(f"the {a} will {v}")
        else:
            sentences.append(f"the {f} was good")

    # Vocabulary
    all_tokens = sorted(list(set(
        [t for words in CATEGORIES.values() for t in words] + ["the", "will", "was", "good"]
    )))
    word2idx = {w: i for i, w in enumerate(all_tokens)}
    idx2word = {i: w for w, i in word2idx.items()}
    return sentences, word2idx, idx2word


def create_training_pairs(sentences: List[str], word2idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract consecutive (token_t, token_{t+1}) next-token pairs."""
    inputs, targets = [], []
    for s in sentences:
        tokens = s.split()
        for i in range(len(tokens) - 1):
            inputs.append(word2idx[tokens[i]])
            targets.append(word2idx[tokens[i + 1]])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class NextTokenEmbeddingModel(nn.Module):
    """Predicts next token from current token via an embedding bottleneck."""

    def __init__(self, vocab_size: int, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)
        return self.head(h)


def train_embedding_model(
    model: NextTokenEmbeddingModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 150,
    lr: float = 0.05,
    seed: int = 42,
) -> float:
    """Train next-token model with CrossEntropyLoss."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)


def compute_knn_category_purity(
    embeddings: np.ndarray, word2idx: Dict[str, int], k: int = 2
) -> float:
    """Compute the fraction of k-nearest neighbors that share the exact same category."""
    token_to_cat = {}
    for cat, words in CATEGORIES.items():
        for w in words:
            token_to_cat[w] = cat

    hits, total = 0, 0
    words_evaluated = [w for w in word2idx if w in token_to_cat]

    for w in words_evaluated:
        idx = word2idx[w]
        vec = embeddings[idx]
        diffs = embeddings - vec
        dists = np.linalg.norm(diffs, axis=1)
        # Exclude self
        dists[idx] = np.inf
        # Top-k nearest indices
        nn_indices = np.argsort(dists)[:k]
        for nn_idx in nn_indices:
            neighbor_word = [word for word, i in word2idx.items() if i == nn_idx][0]
            if neighbor_word in token_to_cat:
                if token_to_cat[neighbor_word] == token_to_cat[w]:
                    hits += 1
                total += 1

    return hits / total if total > 0 else 0.0


def run_experiment_3(save_path: str = "assets/03_emergent_embeddings.png") -> Tuple[float, float]:
    """Run Part 3 experiment and generate 2D projection plot."""
    sentences, word2idx, idx2word = build_synthetic_corpus(n_sentences=3000, seed=42)
    inputs, targets = create_training_pairs(sentences, word2idx)

    model = NextTokenEmbeddingModel(vocab_size=len(word2idx), embed_dim=16)
    final_loss = train_embedding_model(model, inputs, targets, epochs=160, lr=0.03, seed=42)

    # Extract learned embeddings
    weights = model.embedding.weight.detach().cpu().numpy()
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(weights)

    # Compute k-NN purity
    purity = compute_knn_category_purity(weights, word2idx, k=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    category_colors = {
        "animal": "#1f77b4",
        "fruit": "#2ca02c",
        "verb": "#ff7f0e",
        "location": "#9467bd",
        "other": "#7f7f7f",
    }

    token_to_cat = {}
    for cat, words in CATEGORIES.items():
        for w in words:
            token_to_cat[w] = cat

    for word, idx in word2idx.items():
        cat = token_to_cat.get(word, "other")
        color = category_colors[cat]
        x, y = coords[idx]

        ax.scatter(x, y, color=color, s=120, edgecolors="black", linewidth=1.2, zorder=3)
        ax.annotate(
            word,
            (x, y),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold" if cat != "other" else "normal",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=4,
        )

    # Category legend
    for cat, color in category_colors.items():
        if cat in CATEGORIES:
            ax.scatter([], [], color=color, label=f"Category: {cat.capitalize()}", s=90, edgecolors="black")

    ax.set_title(
        f"Proof 3: Emergent Semantic Clustering from Next-Token Prediction\n"
        f"2D PCA Projection of 16-D Embeddings | 2-NN Semantic Purity: {purity * 100:.1f}%\n"
        f"(No semantic similarity labels were provided during training)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel(f"PCA Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PCA Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.9)

    plt.savefig(save_path, dpi=300)
    plt.close()
    return final_loss, purity
