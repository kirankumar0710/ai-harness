"""Demo: a bare harness with one harmless tool.

    export ODYSSEUS_PROVIDER=gemini        # or openai / openrouter / anthropic
    export GEMINI_API_KEY=...              # the matching vendor key
    python demos/dice.py

Expected transcript: user -> assistant (tool call) -> tool result ->
assistant answer. Then a request the dice tool cannot help with, which
the model can only describe: judgment without hands.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odysseus import create_provider, run_loop  # noqa: E402


class RollDice:
    name = "roll_dice"
    spec = {"schema": {
        "name": "roll_dice",
        "description": "Roll some six-sided dice and return each value.",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer",
                                     "description": "How many dice"}},
            "required": ["count"],
        },
    }}

    def run(self, count: int) -> str:
        count = int(count)
        if not 1 <= count <= 20:
            raise ValueError("count must be between 1 and 20")
        return json.dumps([random.randint(1, 6) for _ in range(count)])


def print_event(kind: str, payload: Mapping[str, Any]) -> None:
    if kind == "assistant":
        if payload["text"]:
            print(f"assistant: {payload['text']}")
        usage = payload["usage"]
        print(f"  [tokens in {usage['input']}, out {usage['output']}]")
    elif kind == "tool_start":
        print(f"  -> {payload['name']}({payload['args']})")
    elif kind == "tool_end":
        print(f"  <- {payload['result']}")


def main() -> int:
    provider = create_provider()
    print(f"provider: {provider.config.PROVIDER_NAME} - {provider.model}\n")
    tools = {"roll_dice": RollDice()}
    system = "You are a helpful agent. Use tools when they help."

    for task in ("Roll 3 dice and tell me whether the total beats 10.",
                 "Build a landing page for a coffee shop."):
        print(f"user: {task}")
        messages = [{"role": "user", "text": task}]
        run_loop(provider, system, messages, tools, on_event=print_event,
                 max_turns=10)
        print(f"({len(messages)} messages in the array)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
