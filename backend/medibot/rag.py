"""Hybrid RAG pipeline: RBAC-scoped retrieval -> rerank -> grounded answer.

Flow:
  1. hybrid_search  — fused dense + BM25 over a *broad* candidate set (top-10),
                      already filtered by the user's role inside Qdrant.
  2. rerank         — cross-encoder narrows to the strongest few (top-3).
  3. generate       — only the reranked chunks are placed in the LLM prompt,
                      which is instructed to answer strictly from context and
                      cite sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from . import llm
from .config import settings
from .rerank import rerank
from .vectorstore import get_store

SYSTEM_PROMPT = (
    "You are MediBot, MediAssist Health Network's internal clinical and "
    "operations assistant. Answer strictly and only from the provided context. "
    "Never use outside knowledge. If the answer is not in the context, say "
    "exactly: 'I could not find this information in the approved knowledge base.' "
    "When you answer, be precise with clinical terms, drug names, dosages and codes."
)


@dataclass
class Source:
    source_document: str
    section_title: str
    collection: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "source_document": self.source_document,
            "section_title": self.section_title,
            "collection": self.collection,
        }


@dataclass
class RagResult:
    answer: str
    sources: List[Source]
    retrieval_type: str = "hybrid_rag"
    reranker_scores: List[float] = None  # type: ignore[assignment]


def _build_context(reranked) -> str:
    parts = []
    for i, (point, _score) in enumerate(reranked, start=1):
        p = point.payload
        parts.append(
            f"[Source {i}] document={p['source_document']} | "
            f"section={p['section_title']} | collection={p['collection']}\n"
            f"{p['text']}"
        )
    return "\n\n".join(parts)


def answer_question(question: str, role: str) -> RagResult:
    """Run the full hybrid + rerank RAG pipeline for an authorised role."""
    candidates = get_store().hybrid_search(
        question, role=role, top_k=settings.retrieve_top_k
    )

    if not candidates:
        return RagResult(
            answer=(
                "I could not find this information in the approved knowledge base "
                "you have access to."
            ),
            sources=[],
            reranker_scores=[],
        )

    reranked = rerank(question, candidates, top_k=settings.rerank_top_k)
    context = _build_context(reranked)

    prompt = (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite the documents you used."
    )
    answer = llm.complete(prompt, system=SYSTEM_PROMPT, temperature=0.0)

    # De-duplicate sources while preserving rerank order.
    sources: List[Source] = []
    seen = set()
    scores: List[float] = []
    for point, score in reranked:
        p = point.payload
        scores.append(round(score, 4))
        key = (p["source_document"], p["section_title"])
        if key not in seen:
            seen.add(key)
            sources.append(
                Source(p["source_document"], p["section_title"], p["collection"])
            )

    return RagResult(
        answer=answer,
        sources=sources,
        retrieval_type="hybrid_rag",
        reranker_scores=scores,
    )
