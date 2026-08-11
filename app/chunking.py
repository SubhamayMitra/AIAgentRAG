"""Recursive character-based text chunking with overlap.

Kept deliberately simple (no external chunking library): tries to split on
the largest separator available first, falling back to smaller ones, so
chunks break on paragraph/sentence/word boundaries rather than mid-word.
"""

from __future__ import annotations

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    pieces = _split(text, _SEPARATORS, chunk_size)
    return _merge_with_overlap(pieces, chunk_size, chunk_overlap)


def _split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text on the first separator that actually reduces piece size."""
    if len(text) <= chunk_size:
        return [text]

    sep, *rest = separators
    if sep == "":
        # Last resort: hard character split.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    if len(parts) == 1:
        # Separator not present — try the next, smaller one.
        return _split(text, rest, chunk_size) if rest else [text]

    pieces: list[str] = []
    for part in parts:
        if len(part) > chunk_size:
            pieces.extend(_split(part, rest, chunk_size) if rest else [part])
        else:
            pieces.append(part)
    return pieces


def _merge_with_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Greedily pack small pieces back together up to chunk_size, with overlap between chunks."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = _overlap_tail(current, chunk_overlap) + piece if chunk_overlap else piece
            # If a single piece is itself larger than chunk_size (rare, hard-split fallback
            # already caps this), just keep it as its own chunk.
            if len(current) > chunk_size:
                current = piece

    if current:
        chunks.append(current)

    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or len(text) <= overlap:
        return ""
    return text[-overlap:] + " "
