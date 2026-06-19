"""MediBot — Advanced RAG backend for MediAssist Health Network.

On import we inject ``truststore`` so that all outbound TLS (HuggingFace model
downloads, etc.) trusts the operating-system certificate store. This is what
lets model downloads succeed behind corporate TLS-inspection proxies whose root
CA lives in the Windows/macOS trust store but not in certifi's bundle.

Components:
    config      — settings, RBAC matrix, demo users
    auth        — login + signed session tokens
    llm         — cloud LLM (Groq) wrapper
    ingestion   — Docling structural parsing + hierarchical chunking
    vectorstore — Qdrant hybrid (dense + BM25) store with RBAC filtering
    rerank      — cross-encoder reranking
    rag         — hybrid retrieval + reranking + grounded answer generation
    sql_rag     — natural-language-to-SQL analytical answering
    router      — analytical vs document query routing
    api         — FastAPI application
"""

try:  # use the OS trust store for TLS so HF downloads work behind proxies
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 - never block import on this best-effort step
    pass

__version__ = "1.0.0"
