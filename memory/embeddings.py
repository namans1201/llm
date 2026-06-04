"""Local sentence-transformers embedding singleton.

Groq does not currently expose an embeddings endpoint, so we run a small
local model. `all-MiniLM-L6-v2` is 384-d, fast on CPU, and good enough for a
prototype's semantic similarity needs.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np


_DEFAULT_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model_name: str):
    # Imported lazily so that test collection doesn't pay the torch import cost.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(text: str | list[str], model_name: str | None = None) -> np.ndarray:
    """Return a 2-D ndarray of shape (n, dim)."""
    with _lock:
        model = _load(model_name or _DEFAULT_MODEL)
    arr = model.encode(
        [text] if isinstance(text, str) else list(text),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(arr, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalized vectors -> just a dot product.

    Falls back to a safe normalization if vectors are not unit-length.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dim() -> int:
    return int(_load(_DEFAULT_MODEL).get_sentence_embedding_dimension())
