"""Central configuration: paths, models, RBAC matrix and demo users.

Everything that varies between environments is read from environment
variables (loaded from a local ``.env`` file via python-dotenv) so the same
code runs unchanged in development and on a deployment host.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# Repo root = parent of the ``backend`` directory that contains this package.
BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


def _resolve(path_str: str) -> Path:
    """Resolve configured paths from the most likely project roots.

    Some deployments store the sample dataset one directory above the repo,
    so we try the backend directory, repo root, and the parent of the repo
    root before falling back to the first candidate.
    """
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p.resolve()

    candidates = [BACKEND_DIR / p, REPO_ROOT / p, REPO_ROOT.parent / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


# --------------------------------------------------------------------------- #
# Role-Based Access Control (RBAC)
# --------------------------------------------------------------------------- #
# Single source of truth mapping a *collection* to the roles allowed to read
# it. This drives both ingestion (stamped onto every chunk) and retrieval
# (translated into a Qdrant metadata filter on every query).
COLLECTION_ROLES: Dict[str, List[str]] = {
    "general": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    "clinical": ["doctor", "admin"],
    "nursing": ["nurse", "doctor", "admin"],
    "billing": ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
}

ALL_ROLES = ["doctor", "nurse", "billing_executive", "technician", "admin"]

# Roles permitted to use the analytical SQL RAG chain.
SQL_RAG_ROLES = ["billing_executive", "admin"]


def collections_for_role(role: str) -> List[str]:
    """Return the list of collections a role is allowed to query."""
    return [c for c, roles in COLLECTION_ROLES.items() if role in roles]


def access_roles_for_collection(collection: str) -> List[str]:
    """Return the roles that may read a given collection (defaults to admin)."""
    return COLLECTION_ROLES.get(collection.lower(), ["admin"])


# --------------------------------------------------------------------------- #
# Demo users (one per role). Passwords are demo-only.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DemoUser:
    username: str
    password: str
    role: str
    display_name: str


DEMO_USERS: Dict[str, DemoUser] = {
    "dr.mehta": DemoUser("dr.mehta", "medibot123", "doctor", "Dr. Mehta"),
    "nurse.priya": DemoUser("nurse.priya", "medibot123", "nurse", "Nurse Priya"),
    "billing.ravi": DemoUser(
        "billing.ravi", "medibot123", "billing_executive", "Ravi (Billing)"
    ),
    "tech.anand": DemoUser("tech.anand", "medibot123", "technician", "Anand (Tech)"),
    "admin.sys": DemoUser("admin.sys", "medibot123", "admin", "System Admin"),
}


# --------------------------------------------------------------------------- #
# Runtime settings
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    # LLM
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )

    # Auth
    jwt_secret: str = field(
        default_factory=lambda: os.getenv("MEDIBOT_JWT_SECRET", "dev-secret-change-me")
    )
    token_ttl_minutes: int = field(
        default_factory=lambda: int(os.getenv("MEDIBOT_TOKEN_TTL_MINUTES", "120"))
    )

    # Data
    data_dir: Path = field(
        default_factory=lambda: _resolve(
            os.getenv("MEDIBOT_DATA_DIR", "../data")
        )
    )
    db_path: Path = field(
        default_factory=lambda: _resolve(
            os.getenv(
                "MEDIBOT_DB_PATH",
                "../data/db/mediassist.db",
            )
        )
    )
    qdrant_path: Path = field(
        default_factory=lambda: _resolve(
            os.getenv("MEDIBOT_QDRANT_PATH", "./qdrant_storage")
        )
    )
    collection_name: str = "mediassist"

    # Models
    dense_model: str = field(
        default_factory=lambda: os.getenv(
            "MEDIBOT_DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    sparse_model: str = field(
        default_factory=lambda: os.getenv("MEDIBOT_SPARSE_MODEL", "Qdrant/bm25")
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv(
            "MEDIBOT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    dense_dim: int = 384

    # Retrieval
    retrieve_top_k: int = field(
        default_factory=lambda: int(os.getenv("MEDIBOT_RETRIEVE_TOP_K", "10"))
    )
    rerank_top_k: int = field(
        default_factory=lambda: int(os.getenv("MEDIBOT_RERANK_TOP_K", "3"))
    )


settings = Settings()
