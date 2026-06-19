"""Cross-encoder reranking.

A cross-encoder reads the query and each candidate chunk *together* and emits a
single relevance score, which is far more discriminative than the independent
similarity scores from the initial hybrid retrieval. We over-fetch with hybrid
search (e.g. top-10) and then narrow to a small set (e.g. top-3) here, so only
the strongest passages ever reach the LLM.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

from sentence_transformers import CrossEncoder

from .config import settings


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


def rerank(query: str, points, top_k: int | None = None) -> List[Tuple[object, float]]:
    """Score each candidate against the query and return the top_k (point, score)."""
    top_k = top_k or settings.rerank_top_k
    if not points:
        return []

    pairs = [(query, p.payload["text"]) for p in points]
    scores = reranker().predict(pairs)

    ranked = sorted(zip(points, scores), key=lambda x: float(x[1]), reverse=True)
    return [(p, float(s)) for p, s in ranked[:top_k]]
