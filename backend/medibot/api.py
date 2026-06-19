"""FastAPI application exposing the MediBot backend.

Endpoints:
  POST /login               -> role-tagged session token
  POST /chat                -> RBAC-scoped hybrid RAG or SQL RAG answer + sources
  GET  /collections/{role}  -> collections accessible to a role
  GET  /health              -> health check
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import rag, router
from .auth import AuthError, authenticate, create_token, parse_bearer
from .config import ALL_ROLES, SQL_RAG_ROLES, collections_for_role
from .schemas import (
    ChatRequest,
    ChatResponse,
    CollectionsResponse,
    HealthResponse,
    LoginRequest,
    LoginResponse,
    SourceModel,
)
from .sql_rag import SqlRagError, sql_rag_chain
from .vectorstore import get_store

app = FastAPI(title="MediBot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _refusal(role: str, collection: str) -> ChatResponse:
    accessible = ", ".join(collections_for_role(role))
    msg = (
        f"As a {role.replace('_', ' ')}, you don't have access to {collection} "
        f"documents. I can only answer questions from the {accessible} collections."
    )
    return ChatResponse(
        answer=msg,
        sources=[],
        retrieval_type="rbac_blocked",
        role=role,
        access_blocked=True,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", indexed_chunks=get_store().count())


@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest) -> LoginResponse:
    try:
        identity = authenticate(req.username, req.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return LoginResponse(
        token=create_token(identity),
        role=identity.role,
        display_name=identity.display_name,
        collections=collections_for_role(identity.role),
    )


@app.get("/collections/{role}", response_model=CollectionsResponse)
def collections(role: str) -> CollectionsResponse:
    if role not in ALL_ROLES:
        raise HTTPException(status_code=404, detail=f"Unknown role: {role}")
    return CollectionsResponse(role=role, collections=collections_for_role(role))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, authorization: Optional[str] = Header(default=None)) -> ChatResponse:
    # 1. Authenticate and resolve the role from the signed token.
    try:
        identity = parse_bearer(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    role = identity.role

    # 2. Route the question.
    intended = router.route(req.question)

    # 3. SQL RAG branch (analytical questions).
    if intended == "sql_rag":
        if role not in SQL_RAG_ROLES:
            accessible = ", ".join(collections_for_role(role))
            return ChatResponse(
                answer=(
                    f"As a {role.replace('_', ' ')}, you don't have access to the "
                    "analytical claims/maintenance database. Analytical reporting is "
                    "restricted to billing executives and admins. I can still answer "
                    f"document questions from the {accessible} collections."
                ),
                sources=[],
                retrieval_type="rbac_blocked",
                role=role,
                access_blocked=True,
            )
        try:
            result = sql_rag_chain(req.question)
        except SqlRagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatResponse(
            answer=result.answer,
            sources=[],
            retrieval_type="sql_rag",
            role=role,
            sql=result.sql,
        )

    # 4. Document (hybrid) RAG branch — RBAC enforced inside Qdrant.
    blocked = router.blocked_collection(req.question, role)
    if blocked:
        return _refusal(role, blocked)

    result = rag.answer_question(req.question, role=role)
    return ChatResponse(
        answer=result.answer,
        sources=[SourceModel(**s.as_dict()) for s in result.sources],
        retrieval_type=result.retrieval_type,
        role=role,
        reranker_scores=result.reranker_scores,
    )
