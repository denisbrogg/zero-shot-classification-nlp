import numpy as np
from typing import List, Protocol
from numpy.typing import NDArray
import scipy


class Embedder(Protocol):
    def encode(self, text: List[str], **kwargs) -> NDArray:
        ...


def compute_embeddings(X: str | List[str], model: Embedder):
    if isinstance(X, str):
        X = [X]
    elif not isinstance(X, List):
        raise ValueError(f"Input must be either a string or a list of strings.")
    embeddings = model.encode(X, show_progress_bar=False)
    return embeddings


def compute_similarity(A: NDArray, B: NDArray) -> NDArray:
    return 1 - scipy.spatial.distance.cdist(A, B, "cosine")


def zero_shot_predict(X: str | List[str], labels: List[str], model: Embedder) -> List:
    # Compute embeddings

    X_emb = compute_embeddings(X, model)
    Y_emb = compute_embeddings(labels, model)

    # Compute similarities among them

    similarities = compute_similarity(X_emb, Y_emb)

    # Collect results

    most_similar_labels = np.argmax(similarities, axis=1)
    return [dict(label=labels[idx]) for idx in most_similar_labels]
