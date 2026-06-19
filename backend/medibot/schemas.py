"""Pydantic request/response models for the FastAPI layer."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str
    display_name: str
    collections: List[str]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceModel(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceModel]
    retrieval_type: str
    role: str
    sql: Optional[str] = None
    reranker_scores: Optional[List[float]] = None
    access_blocked: bool = False


class CollectionsResponse(BaseModel):
    role: str
    collections: List[str]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
