"""Extract raw text (plus page metadata, where applicable) from source documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class UnsupportedFileTypeError(Exception):
    """Raised when a file's extension isn't one we know how to load."""


@dataclass
class Section:
    text: str
    page: int | None = None  # 1-indexed page number, when known (PDFs)


def load_sections(path: Path) -> list[Section]:
    """Extract text sections from a single document.

    Each Section roughly corresponds to a page (PDF) or the whole file
    (txt/md/docx), and is later split further by the chunker.
    """
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return _load_text(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")


def _load_text(path: Path) -> list[Section]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Section(text=text)] if text.strip() else []


def _load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    sections: list[Section] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append(Section(text=text, page=i))
    return sections


def _load_docx(path: Path) -> list[Section]:
    import docx

    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [Section(text=text)] if text.strip() else []
