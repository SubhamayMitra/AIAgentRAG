#!/usr/bin/env python3
"""Simple CLI chat REPL for the RAG agent.

Usage:
    python chat.py
"""

from app.agent import stream_reply


def main() -> None:
    print("RAG chat — type your question, or 'exit' / Ctrl-D to quit.\n")
    messages: list[dict] = []

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        print("claude> ", end="", flush=True)
        reply_parts: list[str] = []
        for text in stream_reply(messages):
            print(text, end="", flush=True)
            reply_parts.append(text)
        print("\n")

        messages.append({"role": "assistant", "content": "".join(reply_parts)})


if __name__ == "__main__":
    main()
