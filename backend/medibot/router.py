"""Route an incoming question to either SQL RAG or document (hybrid) RAG.

Analytical questions ("how many", "total", "average", counts over claims or
maintenance tickets) are answered from the relational database; everything else
is answered from the document knowledge base.
"""
from __future__ import annotations

import re
from typing import Optional

from .config import collections_for_role

_ANALYTICAL_PATTERNS = [
    r"\bhow many\b",
    r"\bnumber of\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\bsum\b",
    r"\baverage\b",
    r"\bavg\b",
    r"\bmost\b",
    r"\bfewest\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\bper (department|category|campus|insurer|status)\b",
    r"\bhow much\b",
    r"\bbreakdown of\b",
]

# Data-domain words that, combined with an analytical cue, indicate SQL.
_DATA_TERMS = [
    "claim",
    "claims",
    "ticket",
    "tickets",
    "maintenance",
    "escalated",
    "approved",
    "rejected",
    "pending",
    "submitted",
    "insurer",
    "department",
    "equipment",
]


def is_analytical(question: str) -> bool:
    q = question.lower()
    has_analytical = any(re.search(p, q) for p in _ANALYTICAL_PATTERNS)
    has_data_term = any(re.search(rf"\b{t}\b", q) for t in _DATA_TERMS)
    return has_analytical and has_data_term


def route(question: str) -> str:
    """Return 'sql_rag' or 'hybrid_rag'."""
    return "sql_rag" if is_analytical(question) else "hybrid_rag"


# --------------------------------------------------------------------------- #
# Optional RBAC intent guard (UX nicety on top of the retrieval-level filter)
# --------------------------------------------------------------------------- #
# Keywords that strongly indicate a question is *aimed at* a restricted
# collection. The real security guarantee is the Qdrant metadata filter; this
# guard only exists so the frontend can show a clear, role-aware refusal
# instead of a vague "I couldn't find that".
_COLLECTION_INTENT = {
    "billing": [
        "billing code",
        "insurance",
        "insurer",
        "claim",
        "pre-auth",
        "preauth",
        "reimbursement",
        "icd-10",
        "icd code",
        "tpa",
        "co-pay",
        "premium",
    ],
    "equipment": [
        "calibration",
        "equipment manual",
        "maintenance schedule",
        "ventilator",
        "fault code",
        "service the",
        "calibrate",
    ],
    "clinical": [
        "drug formulary",
        "dosage",
        "treatment protocol",
        "diagnostic reference",
        "prescrib",
    ],
    "nursing": [
        "icu nursing",
        "infection control",
        "ppe",
        "cannula",
        "nursing procedure",
    ],
}


def blocked_collection(question: str, role: str) -> Optional[str]:
    """If the question clearly targets a collection the role can't access,
    return that collection's name; otherwise None."""
    q = question.lower()
    allowed = set(collections_for_role(role))
    for collection, keywords in _COLLECTION_INTENT.items():
        if collection in allowed:
            continue
        if any(kw in q for kw in keywords):
            return collection
    return None
