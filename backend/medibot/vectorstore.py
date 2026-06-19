"""Qdrant hybrid vector store: dense + BM25 sparse, with RBAC metadata filter.

Both a dense (semantic) vector and a sparse (BM25 keyword) vector are stored
for every chunk at index time and queried together in a single Qdrant request
using server-side Reciprocal Rank Fusion (RRF). RBAC is enforced as a
``query_filter`` on ``access_roles`` — restricted chunks are filtered *inside*
Qdrant and never returned to the application, let alone the LLM.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from .config import settings
from .ingestion import Chunk

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


# --------------------------------------------------------------------------- #
# Cached models (loaded once per process)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def dense_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.dense_model)


@lru_cache(maxsize=1)
def sparse_embedder() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.sparse_model)


def embed_dense(texts: List[str]) -> List[List[float]]:
    vectors = dense_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 64
    )
    return [v.tolist() for v in vectors]


def embed_sparse(texts: List[str]) -> List[SparseVector]:
    out = []
    for emb in sparse_embedder().embed(texts):
        out.append(
            SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
        )
    return out


# --------------------------------------------------------------------------- #
# RBAC filter
# --------------------------------------------------------------------------- #
def role_filter(role: str) -> Filter:
    """Qdrant filter that only matches chunks whose access_roles include role."""
    return Filter(
        must=[FieldCondition(key="access_roles", match=MatchValue(value=role))]
    )


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
class VectorStore:
    def __init__(self, client: Optional[QdrantClient] = None):
        self.client = client or QdrantClient(path=str(settings.qdrant_path))
        self.collection = settings.collection_name

    # ---- indexing ---------------------------------------------------------- #
    def recreate_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: VectorParams(
                    size=settings.dense_dim, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams()},
        )
        # Payload index makes the RBAC filter efficient.
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="access_roles",
            field_schema="keyword",
        )

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        texts = [c.text for c in chunks]
        dense = embed_dense(texts)
        sparse = embed_sparse(texts)

        points: List[PointStruct] = []
        for idx, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        DENSE_VECTOR: dense[idx],
                        SPARSE_VECTOR: sparse[idx],
                    },
                    payload=chunk.as_payload(),
                )
            )

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
            )
        return len(points)

    def count(self) -> int:
        try:
            return self.client.count(self.collection).count
        except Exception:  # noqa: BLE001
            return 0

    # ---- retrieval --------------------------------------------------------- #
    def hybrid_search(self, query: str, role: str, top_k: Optional[int] = None):
        """Run a single fused dense+sparse query, scoped by the RBAC filter."""
        top_k = top_k or settings.retrieve_top_k
        dense_q = embed_dense([query])[0]
        sparse_q = embed_sparse([query])[0]
        rbac = role_filter(role)

        # The RBAC filter is applied INSIDE each prefetch (not only at the top
        # level) so restricted chunks are excluded *before* fusion — otherwise a
        # fusion query can surface candidates the role may not see.
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=dense_q, using=DENSE_VECTOR, limit=top_k, filter=rbac
                ),
                Prefetch(
                    query=sparse_q, using=SPARSE_VECTOR, limit=top_k, filter=rbac
                ),
            ],
            query=FusionQuery(fusion="rrf"),
            query_filter=rbac,
            limit=top_k,
            with_payload=True,
        )
        return result.points


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Process-wide singleton (the on-disk Qdrant client allows one handle)."""
    return VectorStore()
