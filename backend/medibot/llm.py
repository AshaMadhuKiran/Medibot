"""Thin wrapper around the cloud-hosted Groq chat completion API.

Centralising the client keeps the API key and model name in one place and lets
the rest of the codebase call a simple ``complete(prompt)`` helper.
"""
from __future__ import annotations

from functools import lru_cache

from groq import Groq

from .config import settings


@lru_cache(maxsize=1)
def _client() -> Groq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy backend/.env.example to backend/.env "
            "and add your Groq API key."
        )
    return Groq(api_key=settings.groq_api_key)


def complete(prompt: str, *, temperature: float = 0.0, system: str | None = None) -> str:
    """Run a single-turn chat completion and return the text content."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client().chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()
