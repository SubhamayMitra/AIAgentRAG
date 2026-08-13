"""OpenAI client + manual tool-calling loop for the RAG agent.

OpenAI's chat completions API has no equivalent of Anthropic's `tool_runner`
(which streams, executes tool calls, and re-prompts automatically), so that
loop is implemented by hand here: stream a turn, accumulate any tool-call
deltas by index, execute recognized tools, append the results, and repeat
until the model stops requesting tools.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from openai import OpenAI

from app.config import settings
from app.prompts import SYSTEM_PROMPT
from app.tools import SEARCH_DOCUMENTS_TOOL, search_documents

client = OpenAI(api_key=settings.openai_api_key or None)

TOOLS = [SEARCH_DOCUMENTS_TOOL]
TOOL_FUNCTIONS = {"search_documents": search_documents}


def _stream_turn(messages: list[dict]) -> Iterator[str]:
    """Stream one assistant turn, yielding text deltas.

    Returns via a final `dict` describing any tool calls requested and the
    finish reason, stashed on the generator's return value.
    """
    stream = client.chat.completions.create(
        model=settings.openai_model,
        max_tokens=settings.max_tokens,
        messages=messages,
        tools=TOOLS,
        stream=True,
    )

    text_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None

    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            text_parts.append(delta.content)
            yield delta.content

        for tc in delta.tool_calls or []:
            slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    return {
        "text": "".join(text_parts),
        "tool_calls": list(tool_calls.values()),
        "finish_reason": finish_reason,
    }


def stream_reply(messages: list[dict]) -> Iterator[str]:
    """Yield response text deltas as they arrive, running the tool loop as needed."""
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

    while True:
        turn = yield from _stream_turn(full_messages)

        if turn["finish_reason"] != "tool_calls":
            break

        full_messages.append(
            {
                "role": "assistant",
                "content": turn["text"] or None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in turn["tool_calls"]
                ],
            }
        )

        for call in turn["tool_calls"]:
            fn = TOOL_FUNCTIONS[call["name"]]
            arguments = json.loads(call["arguments"] or "{}")
            result = fn(**arguments)
            full_messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": result}
            )


def get_reply(messages: list[dict]) -> str:
    """Non-streaming convenience wrapper — mainly for smoke tests."""
    return "".join(stream_reply(messages))
