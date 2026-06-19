"""Standalone ingestion pipeline.

Run this once before starting the API (Docling downloads its parsing models on
first run, so it can take a few minutes):

    cd backend
    python -m scripts.ingest

It parses every document under MEDIBOT_DATA_DIR, hierarchically chunks them,
embeds dense + sparse vectors, and writes them into the on-disk Qdrant store at
MEDIBOT_QDRANT_PATH. The API then reads from that same store.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Allow `python -m scripts.ingest` and `python scripts/ingest.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medibot.ingestion import load_and_chunk  # noqa: E402
from medibot.vectorstore import VectorStore  # noqa: E402


def main() -> None:
    print("=== MediBot ingestion ===")
    chunks = load_and_chunk()

    if not chunks:
        print("No chunks produced — check MEDIBOT_DATA_DIR.")
        sys.exit(1)

    by_collection = Counter(c.collection for c in chunks)
    by_type = Counter(c.chunk_type for c in chunks)
    print("\nChunks per collection:")
    for coll, n in sorted(by_collection.items()):
        print(f"  {coll:12s} {n}")
    print("Chunks per type:")
    for t, n in sorted(by_type.items()):
        print(f"  {t:12s} {n}")

    print("\nBuilding Qdrant collection (dense + BM25)...")
    store = VectorStore()
    store.recreate_collection()
    count = store.index_chunks(chunks)
    print(f"\nIndexed {count} chunks into '{store.collection}'. Done.")


if __name__ == "__main__":
    main()
