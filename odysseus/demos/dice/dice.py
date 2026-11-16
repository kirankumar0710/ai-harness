"""Demo: a bare harness with one harmless tool.

    0. Install the package once, from the odysseus project folder:
         pip install -e .

    Offline (no provider, no API key): a scripted model calls the real
    roll_dice tool and answers from the values it gets back:
         python3 demos/dice/dice.py --offline

    1. Choose a provider (required, there is no default):
         export ODYSSEUS_PROVIDER=gemini   # or openai / openrouter / anthropic

    2. Export the API key for that provider. The variable name to use is
       the ``api_key_env`` entry of the provider's section in
       src/odysseus_v2/providers/providers.ini:
         gemini      GEMINI_API_KEY (or GOOGLE_API_KEY)
         openai      OPENAI_API_KEY
         openrouter  OPENROUTER_API_KEY
         anthropic   ANTHROPIC_API_KEY
       e.g.  export GEMINI_API_KEY=...
       ODYSSEUS_API_KEY works for any provider as a fallback.

    3. Run, from the odysseus project folder:
         python demos/dice/dice.py

See dice_walkthrough.html next to this file for the goal, a sample run and an
analysis of its output.

Base URL and model come from providers.ini; set ODYSSEUS_CONFIG to your
own .ini to override them (openai has no default model). If the provider
or its key is missing, the demo prints which variable to set and exits 1.

Expected transcript: user -> assistant (tool call) -> tool result ->
assistant answer. Then a request the dice tool cannot help with, which
the model can only describe: judgment without hands.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from odysseus_v2 import create_provider, run_loop
from odysseus_v2.providers import Completion, ConfigError, Provider


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


class ScriptedProvider(Provider):
    """A stand-in model for --offline: rolls, then answers from the roll.

    Task 1: asks for roll_dice(count=3), then reads the tool result in
    the conversation and reports the real total. Task 2: answers in text
    with no tool call, as a model with only a dice tool should.
    """

    model = "scripted replies"

    def __init__(self) -> None:
        pass

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        task = messages[0]["text"]
        last = messages[-1]
        if "dice" not in task:
            return _reply("I can only roll dice here, so I can't build a "
                          "page. I can describe one: a hero with the shop's "
                          "name, a menu, opening hours and a map.")
        if last["role"] == "user":
            return _reply("", {"id": "s1", "name": "roll_dice",
                               "args": {"count": 3}})
        values = json.loads(last["text"])
        total = sum(values)
        verdict = "beats 10" if total > 10 else "does not beat 10"
        return _reply(f"I rolled {', '.join(map(str, values))}: the total "
                      f"is {total}, which {verdict}.")

    def _build_request(self, *args):   # never called: complete is replaced
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


def _reply(text: str, *calls: Mapping[str, Any]) -> Completion:
    return Completion(text=text, tool_calls=[dict(c) for c in calls],
                      usage={"input": 0, "output": 0})


def print_event(kind: str, payload: Mapping[str, Any]) -> None:
    if kind == "assistant":
        if payload["text"]:
            print(f"assistant: {payload['text']}")
        usage = payload["usage"]
        if usage["input"] or usage["output"]:
            print(f"  [tokens in {usage['input']}, out {usage['output']}]")
    elif kind == "tool_start":
        print(f"  -> {payload['name']}({payload['args']})")
    elif kind == "tool_end":
        print(f"  <- {payload['result']}")
    elif kind == "truncated":
        print(f"  !! reply cut off at the output-token limit "
              f"({payload['output']} tokens): it is incomplete")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use a scripted model instead of a real one")
    args = parser.parse_args(argv)

    if args.offline:
        provider: Provider = ScriptedProvider()
        print("provider: offline - scripted replies\n")
    else:
        try:
            provider = create_provider()
        except ConfigError as error:
            print(f"error: {error}\n(or run with --offline)",
                  file=sys.stderr)
            return 1
        print(f"provider: {provider.config.PROVIDER_NAME} - "
              f"{provider.model}\n")
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
