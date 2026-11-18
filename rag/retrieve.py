from __future__ import annotations

from pathlib import Path

from ingestion.embed_store import DocumentStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent

document_store = DocumentStore(index_name="documents")
_ingested = False


def ingest_documents(directory: str | Path | None = None) -> int:
    global _ingested
    source = Path(directory) if directory else PROJECT_ROOT / "data"
    added = 0

    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(
            file
            for file in source.iterdir()
            if file.is_file() and file.suffix.lower() in {".txt", ".md", ".text"}
        )
    else:
        files = []

    for file in files:
        try:
            added += document_store.add_text(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue

    _ingested = True
    return added


def retrieve_context(query: str, top_k: int = 5) -> str:
    if not _ingested:
        ingest_documents()
    results = document_store.search(query.strip(), top_k=top_k)
    return "\n".join(results)
