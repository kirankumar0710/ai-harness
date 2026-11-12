"""Demo: a bare harness with one harmless tool.

    0. Install the package once, from the repo root:
         pip install -e .

    1. Choose a provider (required, there is no default):
         export ODYSSEUS_PROVIDER=gemini   # or openai / openrouter / anthropic

    2. Export the API key for that provider. The variable name to use is
       the ``api_key_env`` entry of the provider's section in
       odysseus_v2/providers/providers.ini:
         gemini      GEMINI_API_KEY (or GOOGLE_API_KEY)
         openai      OPENAI_API_KEY
         openrouter  OPENROUTER_API_KEY
         anthropic   ANTHROPIC_API_KEY
       e.g.  export GEMINI_API_KEY=...
       ODYSSEUS_API_KEY works for any provider as a fallback.

    3. Run:
         python demos/dice/dice.py

See dice.html next to this file for the goal, a sample run and an
analysis of its output.

Base URL and model come from providers.ini; set ODYSSEUS_CONFIG to your
own .ini to override them (openai has no default model). If the provider
or its key is missing, the demo prints which variable to set and exits 1.

Expected transcript: user -> assistant (tool call) -> tool result ->
assistant answer. Then a request the dice tool cannot help with, which
the model can only describe: judgment without hands.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Mapping
from typing import Any

from odysseus_v2 import create_provider, run_loop
from odysseus_v2.providers import ConfigError


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
    try:
        provider = create_provider()
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
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
