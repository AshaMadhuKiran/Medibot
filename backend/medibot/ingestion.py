"""Document ingestion: structural parsing (Docling) + hierarchical chunking.

The pipeline:
  1. Parse each PDF/Markdown into a structured Docling document — headings,
     tables and code blocks are recognised, not flattened to plain text.
  2. Chunk with Docling's ``HybridChunker``, which first splits along the
     document's natural hierarchy (section -> subsection -> paragraph/table)
     and then applies a token-aware second pass so chunks fit the embedding
     model's context window.
  3. Each chunk's embedded text is *contextualised* — its parent section
     headings are prepended so a fragment like "25mg twice daily" still carries
     the heading it belongs under.
  4. Every chunk is stamped with the full metadata schema required by the
     assignment, including the ``access_roles`` list that powers RBAC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from .config import access_roles_for_collection, settings

SUPPORTED_DOC_EXT = {".pdf", ".docx", ".pptx", ".html"}


@dataclass
class Chunk:
    """A single retrievable unit with its full metadata schema."""

    text: str  # contextualised text (section heading + body) used for embedding
    source_document: str
    collection: str
    access_roles: List[str]
    section_title: str
    chunk_type: str  # one of: text | table | heading | code

    def as_payload(self) -> Dict:
        return {
            "text": self.text,
            "source_document": self.source_document,
            "collection": self.collection,
            "access_roles": self.access_roles,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
        }


@dataclass
class _Document:
    name: str
    collection: str
    kind: str  # "docling" | "markdown"
    docling_doc: Optional[DoclingDocument] = None
    markdown_text: str = ""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _converter() -> DocumentConverter:
    # These are born-digital (text) PDFs, so OCR is unnecessary; we keep table
    # structure recognition so dosage/billing tables stay intact.
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = False
    pdf_opts.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


def _load_document(file_path: Path, converter: DocumentConverter) -> Optional[_Document]:
    ext = file_path.suffix.lower()
    collection = file_path.parent.name.lower()

    try:
        if ext == ".md":
            # Convert Markdown through Docling too, so headings/tables/code are
            # structurally recognised and chunked the same way as PDFs.
            result = converter.convert(str(file_path))
            return _Document(
                name=file_path.name,
                collection=collection,
                kind="docling",
                docling_doc=result.document,
            )
        if ext in SUPPORTED_DOC_EXT:
            result = converter.convert(str(file_path))
            return _Document(
                name=file_path.name,
                collection=collection,
                kind="docling",
                docling_doc=result.document,
            )
        print(f"  ! skipping unsupported file: {file_path.name}")
        return None
    except Exception as exc:  # noqa: BLE001 - report and continue with other docs
        print(f"  ! failed to parse {file_path.name}: {exc}")
        return None


def discover_files(data_dir: Path) -> List[Path]:
    """Return every ingestible file under the data directory (recursively)."""
    files: List[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in (SUPPORTED_DOC_EXT | {".md"}):
            files.append(path)
    return files


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _section_title(chunk) -> str:
    headings = getattr(chunk.meta, "headings", None)
    if headings:
        return " > ".join(str(h) for h in headings)
    return "Document"


def _chunk_type(chunk) -> str:
    """Derive chunk_type from the underlying Docling doc items."""
    labels = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        label = getattr(item, "label", None)
        if label is not None:
            labels.add(label)

    if DocItemLabel.TABLE in labels:
        return "table"
    if DocItemLabel.CODE in labels:
        return "code"
    if labels and labels <= {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}:
        return "heading"
    return "text"


def build_chunks(documents: List[_Document]) -> List[Chunk]:
    chunker = HybridChunker(tokenizer=settings.dense_model)
    all_chunks: List[Chunk] = []

    for doc in documents:
        if doc.kind != "docling" or doc.docling_doc is None:
            continue

        print(f"  chunking: {doc.name}")
        roles = access_roles_for_collection(doc.collection)

        for chunk in chunker.chunk(dl_doc=doc.docling_doc):
            # serialize() prepends the parent section headings to the body, so
            # the embedded text always carries its hierarchical context.
            embedded_text = chunker.serialize(chunk=chunk)
            if not embedded_text.strip():
                continue

            all_chunks.append(
                Chunk(
                    text=embedded_text,
                    source_document=doc.name,
                    collection=doc.collection,
                    access_roles=roles,
                    section_title=_section_title(chunk),
                    chunk_type=_chunk_type(chunk),
                )
            )

    return all_chunks


def load_and_chunk(data_dir: Optional[Path] = None) -> List[Chunk]:
    """Parse and chunk every document under ``data_dir`` (defaults to config)."""
    data_dir = data_dir or settings.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    converter = _converter()
    files = discover_files(data_dir)
    print(f"Found {len(files)} files under {data_dir}")

    documents: List[_Document] = []
    for file_path in files:
        print(f"  loading: {file_path.relative_to(data_dir)}")
        doc = _load_document(file_path, converter)
        if doc is not None:
            documents.append(doc)

    chunks = build_chunks(documents)
    print(f"Generated {len(chunks)} chunks from {len(documents)} documents")
    return chunks
