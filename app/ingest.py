"""Ingest documents from `documents/` into the ChromaDB collection.

Usage:
    python -m app.ingest            # incremental (diff against what's on disk)
    python -m app.ingest --full     # drop and rebuild the whole collection
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.chunking import chunk_text
from app.config import settings
from app.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileTypeError, load_sections
from app.vectorstore import delete_source, get_client, get_collection, known_sources, replace_source


@dataclass
class IngestSummary:
    files_ingested: list[str] = field(default_factory=list)
    files_skipped: list[tuple[str, str]] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    chunks_written: int = 0

    def __str__(self) -> str:
        lines = [
            f"Ingested {len(self.files_ingested)} file(s), "
            f"{self.chunks_written} chunk(s) written.",
        ]
        if self.files_removed:
            lines.append(f"Removed {len(self.files_removed)} deleted file(s) from the index.")
        if self.files_skipped:
            lines.append(f"Skipped {len(self.files_skipped)} file(s):")
            lines.extend(f"  - {name}: {reason}" for name, reason in self.files_skipped)
        return "\n".join(lines)


def _relative_source(path: Path) -> str:
    return str(path.relative_to(settings.documents_dir))


def _chunk_id(source: str, index: int) -> str:
    return hashlib.sha256(f"{source}:{index}".encode()).hexdigest()


def _discover_files() -> list[Path]:
    if not settings.documents_dir.exists():
        return []
    return sorted(
        p
        for p in settings.documents_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ingest_documents(full: bool = False) -> IngestSummary:
    summary = IngestSummary()

    if full:
        client = get_client()
        client.delete_collection(settings.collection_name)
        get_collection.cache_clear()

    collection = get_collection()
    files = _discover_files()
    current_sources = {_relative_source(p) for p in files}

    # Remove chunks for files that no longer exist on disk.
    for stale_source in known_sources(collection) - current_sources:
        delete_source(collection, stale_source)
        summary.files_removed.append(stale_source)

    for path in files:
        source = _relative_source(path)
        try:
            sections = load_sections(path)
        except UnsupportedFileTypeError as e:
            summary.files_skipped.append((source, str(e)))
            continue
        except Exception as e:  # noqa: BLE001 — one bad file must not abort the run
            summary.files_skipped.append((source, f"failed to load: {e}"))
            continue

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        chunk_index = 0
        for section in sections:
            for chunk in chunk_text(section.text, settings.chunk_size, settings.chunk_overlap):
                ids.append(_chunk_id(source, chunk_index))
                documents.append(chunk)
                metadata = {"source": source, "chunk_index": chunk_index}
                if section.page is not None:
                    metadata["page"] = section.page
                metadatas.append(metadata)
                chunk_index += 1

        if not documents:
            summary.files_skipped.append((source, "no extractable text"))
            continue

        replace_source(collection, source, ids, documents, metadatas)
        summary.files_ingested.append(source)
        summary.chunks_written += len(documents)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Drop and rebuild the entire collection instead of an incremental diff.",
    )
    args = parser.parse_args()
    print(ingest_documents(full=args.full))


if __name__ == "__main__":
    main()
