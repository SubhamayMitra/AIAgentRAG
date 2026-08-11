"""Thin wrapper around a local, persistent ChromaDB collection."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from app.config import settings


@lru_cache
def get_client() -> chromadb.ClientAPI:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))


@lru_cache
def get_collection() -> Collection:
    """Get (or create) the collection, using ChromaDB's bundled local embedding
    function (ONNX MiniLM — no external API key, runs on-device).

    To upgrade to Voyage AI embeddings later, swap the embedding_function here
    for `embedding_functions.VoyageAIEmbeddingFunction(api_key=..., model=...)`
    — but note this REQUIRES a full re-ingestion (`python -m app.ingest --full`).
    Embeddings from different models live in different vector spaces and are
    not comparable within one collection; a partial swap silently corrupts
    retrieval rather than raising an error.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )


def known_sources(collection: Collection) -> set[str]:
    """Distinct `source` metadata values currently stored in the collection."""
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    return {m["source"] for m in metadatas if m and "source" in m}


def replace_source(
    collection: Collection,
    source: str,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    """Delete any existing chunks for `source`, then insert the new ones.

    Delete-then-insert (rather than a naive upsert) avoids stale chunks
    lingering when a file's content changes and its chunk boundaries shift.
    """
    collection.delete(where={"source": source})
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)


def delete_source(collection: Collection, source: str) -> None:
    collection.delete(where={"source": source})


def query(collection: Collection, query_text: str, n_results: int) -> dict[str, Any]:
    return collection.query(query_texts=[query_text], n_results=n_results)
