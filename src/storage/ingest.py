"""Chunk and index source files, updating existing records without duplicates."""

import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..models import Source
from ..tracing import trace_ingestion

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32
PAGE_MARKER = re.compile(r"^## Page (\d+)[ \t]*$", re.MULTILINE)
REPORT_DATA = "report_data"


def split_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            # try to split at a newline boundary if possible
            boundary = text.rfind("\n", start + CHUNK_SIZE // 2, end)
            if boundary != -1:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def document_sources(path: Path, data_dir: Path, source_kind: str) -> list[Source]:
    text = path.read_text(encoding="utf-8")
    parts = PAGE_MARKER.split(text) if source_kind == "brochure" else [text]
    pages = (
        [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]
        if len(parts) > 1
        else [(None, text)]
    )
    # Link brochure citations to the original PDF when it is available.
    citation_path = path
    if source_kind == "brochure" and path.with_suffix(".pdf").is_file():
        citation_path = path.with_suffix(".pdf")
    document_id = f"{source_kind}:{path.stem}"
    sources = []
    for page, page_text in pages:
        for chunk in split_text(page_text):
            chunk_index = len(sources)
            metadata = {
                "document_id": document_id,
                "source_kind": source_kind,
                "relative_path": citation_path.relative_to(data_dir).as_posix(),
                "chunk_index": chunk_index,
            }
            if page is not None:
                metadata["page"] = page
            sources.append(
                Source(id=f"{document_id}:{chunk_index}", text=chunk, metadata=metadata)
            )
    return sources


def stored_chunks(collection, document_id):
    records = collection.get(
        where={"document_id": document_id}, include=["documents", "metadatas"]
    )
    return {
        record_id: (
            text,
            {key: value for key, value in metadata.items() if key != REPORT_DATA},
        )
        for record_id, text, metadata in zip(
            records["ids"], records["documents"], records["metadatas"], strict=True
        )
    }


def store_chunks(collection, sources, gemini):
    embeddings = []
    for start in range(0, len(sources), BATCH_SIZE):
        batch = sources[start : start + BATCH_SIZE]
        embeddings.extend(gemini.embed_documents([source.text for source in batch]))
    collection.upsert(
        ids=[source.id for source in sources],
        documents=[source.text for source in sources],
        metadatas=[source.metadata for source in sources],
        embeddings=embeddings,
    )


def ingest_document(collection, path, data_dir, source_kind, gemini):
    sources = document_sources(path, data_dir, source_kind)
    stored = stored_chunks(collection, f"{source_kind}:{path.stem}")
    changed = [
        source
        for source in sources
        if stored.get(source.id) != (source.text, source.metadata)
    ]
    # if new or changed chunks exist
    if changed:
        store_chunks(collection, changed, gemini)
    # Remove old IDs only after all replacement chunks are stored.
    stale_ids = sorted(set(stored) - {source.id for source in sources})
    if stale_ids:
        collection.delete(ids=stale_ids)
    if source_kind == "service" and sources and (changed or stale_ids):
        collection.update(ids=[sources[0].id], metadatas=[{REPORT_DATA: ""}])
    return len(sources)


@trace_ingestion
def ingest_documents(settings, gemini):
    """Yield (relative file path, chunk count) after each document is stored."""
    groups = [
        (
            settings.chroma_product_collection,
            "brochure",
            sorted((settings.data_dir / "brochures").glob("*.md")),
        ),
        (
            settings.chroma_service_collection,
            "service",
            sorted((settings.data_dir / "HistoricalServices").glob("*.txt")),
        ),
    ]
    for _, source_kind, paths in groups:
        if not paths:
            raise ValueError(
                f"No {source_kind} source files found under {settings.data_dir}."
            )
    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    for name, source_kind, paths in groups:
        collection = client.get_or_create_collection(name, embedding_function=None)
        for path in paths:
            count = ingest_document(
                collection, path, settings.data_dir, source_kind, gemini
            )
            yield path.relative_to(settings.data_dir).as_posix(), count
    print("Ingestion completed successfully.")
