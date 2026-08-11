"""Anthropic client + tool-runner wiring for the RAG agent."""

from __future__ import annotations

from collections.abc import Iterator

import anthropic

from app.config import settings
from app.prompts import SYSTEM_BLOCKS
from app.tools import search_documents

client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)


def _build_runner(messages: list[dict], stream: bool = False):
    return client.beta.messages.tool_runner(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_BLOCKS,
        tools=[search_documents],
        messages=messages,
        stream=stream,
    )


def stream_reply(messages: list[dict]) -> Iterator[str]:
    """Yield response text deltas as they arrive, running the tool loop as needed."""
    runner = _build_runner(messages, stream=True)
    for stream in runner:
        yield from stream.text_stream
        final_message = stream.get_final_message()
        # With only the one custom tool, pause_turn shouldn't occur here, but
        # this is where a future server tool (e.g. web_search) would need
        # explicit resume handling — see shared/tool-use-concepts.md.
        if final_message.stop_reason == "pause_turn":
            break


def get_reply(messages: list[dict]) -> str:
    """Non-streaming convenience wrapper — mainly for smoke tests."""
    runner = _build_runner(messages)
    final_message = runner.until_done()
    return next((b.text for b in final_message.content if b.type == "text"), "")
