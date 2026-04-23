"""
Pure-Python sparse TF-IDF engine — zero native dependencies.

Used by both semantic_index and memory so embeddings work on any
Python version without compiling Rust/C++ extensions.

Why TF-IDF works well for code search:
  Code vocabulary is highly distinctive (function names, module paths,
  keywords). Even unigram TF-IDF yields strong retrieval signal.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """
    Extract meaningful tokens from code or natural language.

    Splits on non-alphanumeric boundaries, lowercases, and keeps
    tokens of at least 2 characters to reduce stop-word noise.
    """
    raw = re.findall(r"[A-Za-z][a-z0-9]*|[A-Z]{2,}(?=[A-Z][a-z]|\d|\b)|[A-Z]{2,}|\d+", text)
    return [tok.lower() for tok in raw if len(tok) >= 2]


def tf_vector(tokens: list[str]) -> dict[str, float]:
    """Compute term-frequency vector (normalised by document length)."""
    if not tokens:
        return {}
    counts: Counter[str] = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def cosine_sparse(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse TF vectors."""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a.get(k, 0.0) * v for k, v in vec_b.items())
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed(text: str) -> dict[str, float]:
    """Return a sparse TF vector for *text*."""
    return tf_vector(tokenize(text))
