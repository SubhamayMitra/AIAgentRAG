"""FastAPI app: chat (streamed), document ingestion/upload, and the static web UI."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import stream_reply
from app.config import settings
from app.ingest import ingest_documents

app = FastAPI(title="RAG Agent API")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    messages = [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})

    def event_stream():
        for text in stream_reply(messages):
            yield json.dumps({"type": "delta", "text": text}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    # Newline-delimited JSON over a chunked response — not native EventSource
    # SSE, since this endpoint needs a POST body (message + history). Consumed
    # client-side via fetch() + a ReadableStream reader (see static/app.js).
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/ingest")
def ingest():
    summary = ingest_documents()
    return {
        "files_ingested": summary.files_ingested,
        "files_removed": summary.files_removed,
        "files_skipped": summary.files_skipped,
        "chunks_written": summary.chunks_written,
    }


@app.post("/api/upload")
async def upload(file: UploadFile):
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.documents_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    summary = ingest_documents()
    return {
        "saved_as": dest.name,
        "files_ingested": summary.files_ingested,
        "files_skipped": summary.files_skipped,
        "chunks_written": summary.chunks_written,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
