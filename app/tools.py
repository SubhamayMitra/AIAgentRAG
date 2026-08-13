"""The retrieval tool exposed to the model via OpenAI function calling."""

from __future__ import annotations

from app.config import settings
from app.vectorstore import get_collection, query as query_collection

# OpenAI has no equivalent of Anthropic's `@beta_tool` (which derives the JSON
# schema from the function signature + docstring), so the schema is written
# out by hand here and paired with the plain function below.
SEARCH_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the ingested document knowledge base for relevant passages. "
            "Use this whenever the user's question might be answered by the "
            "documents that have been ingested into the knowledge base — for "
            "example, questions about specific facts, policies, or content that "
            "would plausibly appear in an uploaded document. Do not use it for "
            "general knowledge questions, greetings, or requests unrelated to "
            "the document collection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language question or topic to search for.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of relevant passages to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def search_documents(query: str, n_results: int = 5) -> str:
    n_results = n_results or settings.n_results_default
    collection = get_collection()
    result = query_collection(collection, query, n_results)

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]

    if not documents:
        return "No relevant passages were found in the document knowledge base."

    blocks = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page")
        location = f"{source} (page {page})" if page is not None else source
        blocks.append(f"[{i}] Source: {location}\n{doc}")

    return "\n\n".join(blocks)
