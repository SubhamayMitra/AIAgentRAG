# AI Agent RAG

A simple, agentic RAG (Retrieval-Augmented Generation) chatbot. It ingests your documents
(PDF, Word, text/Markdown) into a local vector store and answers questions using OpenAI, which
decides for itself when a question needs a document search versus when it can answer directly.

Built directly on the OpenAI SDK and ChromaDB — no LangChain/LlamaIndex.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

## Add documents and ingest

Drop `.pdf`, `.docx`, `.txt`, or `.md` files into `documents/`, then run:

```bash
python -m app.ingest
```

This is incremental: re-running it picks up new/changed/deleted files. Add `--full` to drop and
rebuild the whole index from scratch:

```bash
python -m app.ingest --full
```

You can also ingest/upload via the API (see below).

## Run

**CLI chat:**

```bash
python chat.py
```

**Web app (API + browser UI):**

```bash
uvicorn app.api:app --reload
```

Then open http://127.0.0.1:8000 — you can upload documents and chat from the browser. The same
server also exposes:

- `POST /api/chat` — `{"message": "...", "history": [...]}`, streams the response as
  newline-delimited JSON (`{"type": "delta", "text": "..."}` lines, ending with `{"type": "done"}`).
- `POST /api/ingest` — re-runs ingestion over `documents/`.
- `POST /api/upload` — multipart file upload; saves to `documents/` and ingests it immediately.

## How it works

1. **Ingestion** (`app/ingest.py`): loads each document (`app/loaders.py`), splits it into
   overlapping chunks (`app/chunking.py`), and stores them in a local ChromaDB collection
   (`app/vectorstore.py`) under `data/chroma_db/`.
2. **Retrieval tool** (`app/tools.py`): a `search_documents(query, n_results)` tool backed by a
   ChromaDB similarity query, returning cited passages (source filename + page, where known).
3. **Agent** (`app/agent.py`): OpenAI (`gpt-4o` by default) is given the tool and a system
   prompt (`app/prompts.py`) instructing it to search when relevant, cite sources, and say when it
   doesn't know — rather than always retrieving on every turn. Unlike Anthropic's SDK, OpenAI's
   client doesn't execute tool calls automatically, so `stream_reply()` runs that loop by hand:
   stream a turn, run any requested tool calls, feed the results back, repeat until the model
   stops asking for tools.
4. **Interfaces**: `chat.py` (CLI REPL) and `app/api.py` (FastAPI + the static web UI in
   `static/`) are both thin layers over `app/agent.py`.

Conversation history is client-owned and sent in full on every request (the OpenAI API is
stateless) — there's no server-side session store in this version.

## Notes and tradeoffs

- **Embeddings**: uses ChromaDB's bundled local embedding model (runs on-device, no extra API key).
  It has a 256-token max input length, which is why the default chunk size (800 characters, ~150
  overlap) is kept well under that ceiling — text beyond the limit is silently truncated before
  embedding, not rejected. Tune `CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env` if retrieval quality is off
  for your documents.
- **Upgrading to Voyage AI embeddings** (better quality, requires a separate Voyage API key): swap
  the embedding function in `app/vectorstore.py`'s `get_collection()` for
  `embedding_functions.VoyageAIEmbeddingFunction(api_key=..., model=...)`. You **must** run
  `python -m app.ingest --full` afterward — embeddings from different models aren't comparable
  within one collection, so mixing them silently degrades retrieval rather than erroring.
- **Ingestion strategy**: each ingestion run deletes and re-inserts chunks per changed file, and
  removes chunks for files deleted from `documents/`. Unchanged files are still re-embedded on
  every run (a possible future optimization: skip files by content hash).
- **Prompt caching**: OpenAI caches repeated prompt prefixes (system prompt, tool definitions)
  automatically server-side once they're long enough — no explicit opt-in needed, unlike
  Anthropic's `cache_control` markers.
