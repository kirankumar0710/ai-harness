"""Demo: context compaction - the same long task, before and after.

    0. Install the package once, from the odysseus project folder:
         pip install -e .

    1. Run offline first - no provider, no API key, no network:
         python3 demos/context/context.py --offline

    2. Run against a real model (provider and key as in the dice demo):
         python3 demos/context/context.py

See context_walkthrough.html next to this file for the goal, a sample
run and an analysis of its output.

The throwaway project holds a chain of 8 files in data/. Each has a lot
of filler text, one VALUE, and the name of the NEXT file, so the agent
must read them one at a time. Four decoy files sit in the same folder
with large values that must not be counted.

The demo runs the same task twice, each in a fresh project:

    run 1  without compaction   an unlimited budget: every file read
                                stays in the conversation, so each
                                request is bigger than the last
    run 2  with compaction      a small budget: when the conversation
                                outgrows it, old turns are folded into a
                                summary the model writes

and ends with a before/after table: requests, compactions, the largest
request, total tokens sent, and whether TOTAL.txt is still correct.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from odysseus_v2 import create_provider, run_task
from odysseus_v2.context import (CHARS_PER_TOKEN, SUMMARY_HEADER,
                                 SUMMARY_PROMPT, estimate_tokens)
from odysseus_v2.providers import Completion, ConfigError, Provider

CHAIN = [("start", 17), ("amber", 42), ("birch", 8), ("cedar", 23),
         ("delta", 15), ("ember", 4), ("fjord", 16), ("grove", 31)]
DECOYS = [("hazel", 100), ("ivory", 200), ("jade", 300), ("koala", 400)]
EXPECTED_TOTAL = sum(value for _, value in CHAIN)          # 156

FILLER_LINES = 40
COMPACT_BUDGET = 2_500           # tokens, for the "with compaction" run
NO_LIMIT = 10 ** 9

TASK = ("Start at data/start.txt. Each file in the chain has a VALUE and "
        "names the NEXT file to read. Follow the chain to the end, add up "
        "the VALUEs of the files in the chain only, and write the total to "
        "TOTAL.txt as a single number.")


# --------------------------------------------------------------------------
# Sandbox, approver and measuring
# --------------------------------------------------------------------------


def chain_file(position: int, value: int, next_name: str | None) -> str:
    filler = "".join(
        f"Log line {i:03}: routine entry, nothing to act on here.\n"
        for i in range(1, FILLER_LINES + 1))
    following = (f"NEXT: data/{next_name}.txt" if next_name
                 else "NEXT: none - this is the last file in the chain")
    return (f"Chain file {position}.\n{filler}VALUE: {value}\n"
            f"{following}\n")


def make_sandbox() -> str:
    project = tempfile.mkdtemp(prefix="odysseus-context-")
    data = os.path.join(project, "data")
    os.makedirs(data)
    for i, (name, value) in enumerate(CHAIN):
        following = CHAIN[i + 1][0] if i + 1 < len(CHAIN) else None
        with open(os.path.join(data, f"{name}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(chain_file(i + 1, value, following))
    for name, value in DECOYS:
        with open(os.path.join(data, f"{name}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(chain_file(0, value, None).replace(
                "Chain file 0.", "Not part of the chain."))
    return project


def read_total(project: str) -> str:
    path = os.path.join(project, "TOTAL.txt")
    if not os.path.isfile(path):
        return "(missing)"
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def approver(call: Mapping[str, Any], reason: str) -> bool:
    approved = call["name"] in {"write_file", "edit_file"}
    print(f"  ? approver: {call['name']} -> {'yes' if approved else 'no'}")
    return approved


class Meter(Provider):
    """Wraps a provider; records the size of every request it sends."""

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.agent_sizes: list[int] = []
        self.summary_sizes: list[int] = []
        self.input_tokens = 0

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        size = estimate_tokens(messages) + len(system) // CHARS_PER_TOKEN
        if system == SUMMARY_PROMPT:
            self.summary_sizes.append(size)
        else:
            self.agent_sizes.append(size)
        reply = self.inner.complete(system, messages, tools,
                                    allow_tool_calls=allow_tool_calls)
        self.input_tokens += reply["usage"]["input"]
        return reply

    def _build_request(self, *args):   # never called: complete is replaced
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


def print_event(kind: str, payload: Mapping[str, Any]) -> None:
    if kind == "assistant":
        if payload["text"]:
            print(f"assistant: {payload['text']}")
        usage = payload["usage"]
        if usage["input"] or usage["output"]:
            print(f"  [tokens in {usage['input']}, out {usage['output']}]")
    elif kind == "tool_start":
        args = ", ".join(f"{k}={v!r}" for k, v in payload["args"].items())
        print(f"  -> {payload['name']}({args[:80]})")
    elif kind == "tool_end":
        result = payload["result"]
        value = re.search(r"VALUE: \d+", result)
        following = re.search(r"NEXT: \S+", result)
        if value:
            shown = (f"{len(result.splitlines())} lines; {value.group(0)}; "
                     f"{following.group(0) if following else 'NEXT: none'}")
        else:
            shown = (result.strip().splitlines() or ["(empty)"])[0][:100]
        print(f"  {'!!' if payload['is_error'] else '<-'} {shown}")
    elif kind == "compact":
        print(f"  ** compacted: {payload['summarised']} messages folded, "
              f"~{payload['before']} -> ~{payload['after']} tokens")


# --------------------------------------------------------------------------
# Offline mode: a scripted model
# --------------------------------------------------------------------------


def _reply(text: str = "", *calls: tuple[str, dict[str, Any]]) -> Completion:
    tool_calls = [{"id": f"s{i}", "name": name, "args": args}
                  for i, (name, args) in enumerate(calls, start=1)]
    return Completion(text=text, tool_calls=tool_calls,
                      usage={"input": 0, "output": 0})


class ScriptedModel(Provider):
    """Follows the chain by script, and writes summaries when asked.

    It knows only what is in the conversation it is sent, like a real
    model: the total it writes comes from the VALUEs it can still see
    (in kept tool results and in the summary), and a summary it writes
    lists only the VALUEs visible in the transcript it was given.
    """

    model = "scripted replies"

    def __init__(self) -> None:
        self._step = 0

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        if system == SUMMARY_PROMPT:
            return _reply(self._summarise(messages[0]["text"]))
        self._step += 1
        if self._step <= len(CHAIN):
            name = CHAIN[self._step - 1][0]
            return _reply("", ("read_file", {"path": f"data/{name}.txt"}))
        if self._step == len(CHAIN) + 1:
            total = sum(_visible_values(messages).values())
            return _reply(f"Total of the VALUEs I can see: {total}.",
                          ("write_file", {"path": "TOTAL.txt",
                                          "content": f"{total}\n"}))
        return _reply("Done: the total is in TOTAL.txt.")

    @staticmethod
    def _summarise(transcript: str) -> str:
        earlier = re.findall(r"Files read so far: (.*)", transcript)
        pairs = dict(p.split("=") for p in
                     (earlier[-1].split(", ") if earlier else []) if "=" in p)
        for block in transcript.split("\n\n[")[1:]:
            path = re.search(r"called read_file\(\{'path': '([^']+)'", block)
            if path:
                last_path = path.group(1)
            value = re.search(r"VALUE: (\d+)", block)
            if block.startswith("tool") and value:
                pairs[last_path] = value.group(1)
        listed = ", ".join(f"{k}={v}" for k, v in pairs.items())
        total = sum(int(v) for v in pairs.values())
        return ("Task: follow the chain from data/start.txt, sum the "
                "VALUEs of chain files only, write TOTAL.txt.\n"
                f"Files read so far: {listed}\n"
                f"Running total: {total}")

    def _build_request(self, *args):
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


def _visible_values(messages: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """VALUEs a model could see: from the summary and kept results."""
    seen: dict[str, int] = {}
    paths = {}
    for message in messages:
        text = message.get("text") or ""
        for listed in re.findall(r"Files read so far: (.*)", text):
            for pair in listed.split(", "):
                if "=" in pair:
                    path, value = pair.split("=")
                    seen[path] = int(value)
        for call in message.get("tool_calls", []):
            paths[call["id"]] = call["args"].get("path", "")
        value = re.search(r"VALUE: (\d+)", text)
        if message.get("role") == "tool" and value:
            seen[paths.get(message["tool_call_id"], "?")] = int(value.group(1))
    return seen


# --------------------------------------------------------------------------
# Running before and after
# --------------------------------------------------------------------------


def run(label: str, provider: Provider, budget: int) -> dict[str, Any]:
    print(f"=== {label} ===")
    project = make_sandbox()
    meter = Meter(provider)
    messages: list = []
    print(f"user: {TASK}")
    try:
        run_task(meter, project, TASK, mode="safe", approver=approver,
                 budget_tokens=budget, on_event=print_event,
                 max_turns=25, messages=messages,
                 extra_prompt="Always reply in English.")
        total = read_total(project)
    finally:
        shutil.rmtree(project, ignore_errors=True)

    summary = next((m["text"] for m in messages
                    if m.get("text", "").startswith(SUMMARY_HEADER)), None)
    if summary:
        print("\nthe summary the conversation now starts with:")
        print("  " + summary.replace("\n", "\n  "))
    print("\nrequest sizes (estimated tokens), in order:")
    print("  " + " ".join(str(size) for size in meter.agent_sizes))
    print()
    all_sizes = meter.agent_sizes + meter.summary_sizes
    return {"agent requests": len(meter.agent_sizes),
            "compactions": len(meter.summary_sizes),
            "largest request": max(all_sizes),
            "messages at the end": len(messages),
            "tokens sent (estimate)": sum(all_sizes),
            "tokens sent (reported)": meter.input_tokens or "-",
            "TOTAL.txt": total,
            "correct": "yes" if total == str(EXPECTED_TOTAL) else "no"}


def print_comparison(before: dict[str, Any], after: dict[str, Any]) -> None:
    print("=== before / after ===")
    print(f"{'':26}{'without compaction':>20}{'with compaction':>18}")
    for row in before:
        print(f"{row:26}{before[row]!s:>20}{after[row]!s:>18}")
    print(f"(compaction budget: {COMPACT_BUDGET} tokens; "
          f"expected total: {EXPECTED_TOTAL})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use scripted replies instead of a real model")
    args = parser.parse_args(argv)

    if args.offline:
        def provider_for() -> Provider:
            return ScriptedModel()
        print("provider: offline - scripted replies\n")
    else:
        try:
            live = create_provider()
        except ConfigError as error:
            print(f"error: {error}\n(or run with --offline)",
                  file=sys.stderr)
            return 1

        def provider_for() -> Provider:
            return live
        print(f"provider: {live.config.PROVIDER_NAME} - {live.model}\n")

    before = run("run 1: without compaction", provider_for(), NO_LIMIT)
    after = run("run 2: with compaction", provider_for(), COMPACT_BUDGET)
    print_comparison(before, after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
