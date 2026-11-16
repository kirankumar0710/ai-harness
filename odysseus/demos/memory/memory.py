"""Demo: memory - the same task, before and after the agent remembers.

    0. Install the package once, from the odysseus project folder:
         pip install -e .

    1. Run offline first - no provider, no API key, no network:
         python3 demos/memory/memory.py --offline

    2. Run against a real model (provider and key as in the dice demo):
         python3 demos/memory/memory.py

See memory_walkthrough.html next to this file for the goal, a sample run
and an analysis of its output.

The throwaway project hides one fact you cannot guess: its checks live
in checks/check_*.py, so plain "python3 -m unittest" finds no tests at
all. The right command is

    python3 -m unittest discover -s checks -p 'check_*.py' -t .

The demo runs the same task twice, each in a fresh conversation:

    run 1  without memory   no AGENT_MEMORY.md; the agent must discover
                            the command, and is asked to save what a
                            future session needs with remember
    run 2  with memory      calc.py is reset to the buggy version, but
                            AGENT_MEMORY.md from run 1 is kept and loaded
                            into the system prompt

and ends with a before/after table: model calls, tool calls, wasted
calls, tokens sent and whether the checks pass.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from odysseus_v2 import create_provider, run_task
from odysseus_v2.context import estimate_tokens
from odysseus_v2.memory import MEMORY_FILE, load_memory
from odysseus_v2.providers import Completion, ConfigError, Provider

CHECK_COMMAND = "python3 -m unittest discover -s checks -p 'check_*.py' -t ."

TASK = ("The checks for this project fail. Fix the bug in calc.py and "
        "confirm the checks pass. Before you finish, use the remember tool "
        "to save anything a future session would need to know about "
        "working in this project.")

BUGGY_CALC = '''\
def average(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    return sum(values) / (len(values) - 1)
'''

CHECK_CALC = '''\
import unittest

from calc import average


class AverageCheck(unittest.TestCase):
    def test_average(self):
        self.assertEqual(average([2, 4, 6]), 4)
'''

PROJECT_FILES = {
    "calc.py": BUGGY_CALC,
    "checks/__init__.py": "",
    "checks/check_calc.py": CHECK_CALC,
}

# The approver allows file changes, remember, and plain unittest runs.
UNITTEST = re.compile(r"\s*python3?\s+-m\s+unittest\b[^;&|`$<>()\n]*")

# A tool result that means the call achieved nothing.
WASTED = re.compile(r"NO TESTS RAN|Ran 0 tests|^ERROR|^BLOCKED", re.M)


# --------------------------------------------------------------------------
# Sandbox, approver and measuring
# --------------------------------------------------------------------------


def make_sandbox() -> str:
    project = tempfile.mkdtemp(prefix="odysseus-memory-")
    for relative, content in PROJECT_FILES.items():
        path = os.path.join(project, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return project


def reset_code(project: str) -> None:
    """Put the bug back; leave AGENT_MEMORY.md alone."""
    with open(os.path.join(project, "calc.py"), "w", encoding="utf-8") as f:
        f.write(BUGGY_CALC)


def checks_pass(project: str) -> bool:
    done = subprocess.run(CHECK_COMMAND, shell=True, cwd=project,
                          capture_output=True, text=True)
    return done.returncode == 0


def approver(call: Mapping[str, Any], reason: str) -> bool:
    name = call["name"]
    command = str(call["args"].get("command", ""))
    approved = (name in {"write_file", "edit_file", "remember"}
                or (name == "bash" and bool(UNITTEST.fullmatch(command))))
    print(f"  ? approver: {name} -> {'yes' if approved else 'no'}")
    return approved


class Meter(Provider):
    """Wraps a provider and records every request it sends."""

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.systems: list[str] = []
        self.sizes: list[int] = []
        self.input_tokens = 0

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        self.systems.append(system)
        self.sizes.append(estimate_tokens(messages)
                          + len(system) // 4)
        reply = self.inner.complete(system, messages, tools,
                                    allow_tool_calls=allow_tool_calls)
        self.input_tokens += reply["usage"]["input"]
        return reply

    def _build_request(self, *args):   # never called: complete is replaced
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


class Recorder:
    """Prints the run and counts tool calls and wasted ones."""

    def __init__(self) -> None:
        self.tool_calls = 0
        self.wasted: list[str] = []

    def __call__(self, kind: str, payload: Mapping[str, Any]) -> None:
        if kind == "assistant":
            if payload["text"]:
                print(f"assistant: {payload['text']}")
            usage = payload["usage"]
            if usage["input"] or usage["output"]:
                print(f"  [tokens in {usage['input']}, "
                      f"out {usage['output']}]")
        elif kind == "tool_start":
            self.tool_calls += 1
            print(f"  -> {payload['name']}({_short(payload['args'])})")
        elif kind == "tool_end":
            result = payload["result"]
            why = WASTED.search(result)
            if payload["is_error"] or why:
                reason = why.group(0) if why else _last_line(result)
                self.wasted.append(f"{payload['name']}: {reason}")
            lines = [ln for ln in result.splitlines() if ln.strip()]
            lines = lines or ["(empty)"]
            shown = lines[0][:100]
            if len(lines) > 1:
                shown += f"  ...  {lines[-1][:60]}  ({len(lines)} lines)"
            mark = "!!" if payload["is_error"] else "<-"
            print(f"  {mark} {shown}")


def _short(args: Mapping[str, Any]) -> str:
    parts = []
    for key, value in args.items():
        text = repr(value)
        if len(text) > 70:
            text = text[:67].replace("\\n", " ") + "...'"
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1][:60] if lines else "(empty)"


# --------------------------------------------------------------------------
# Offline mode: a scripted model
# --------------------------------------------------------------------------


def _reply(text: str = "", *calls: tuple[str, dict[str, Any]]) -> Completion:
    tool_calls = [{"id": f"s{i}", "name": name, "args": args}
                  for i, (name, args) in enumerate(calls, start=1)]
    return Completion(text=text, tool_calls=tool_calls,
                      usage={"input": 0, "output": 0})


FIX = ("edit_file", {"path": "calc.py", "old": "(len(values) - 1)",
                     "new": "len(values)"})
NOTE = ("Checks live in checks/check_*.py, so plain 'python3 -m unittest' "
        f"finds none; run them with: {CHECK_COMMAND}")

OFFLINE_SCRIPTS = {
    "without memory": [
        _reply("", ("list_files", {})),
        _reply("", ("read_file", {"path": "calc.py"})),
        _reply("", FIX),
        _reply("", ("bash", {"command": "python3 -m unittest"})),
        _reply("No tests ran; the checks are not named test_*.py.",
               ("bash", {"command": CHECK_COMMAND})),
        _reply("", ("remember", {"note": NOTE})),
        _reply("Fixed average() and the check passes."),
    ],
    "with memory": [
        _reply("", ("read_file", {"path": "calc.py"})),
        _reply("", FIX),
        _reply("", ("bash", {"command": CHECK_COMMAND})),
        _reply("Fixed average() and the check passes, using the command "
               "from project memory."),
    ],
}


class ScriptedProvider(Provider):
    """Replays fixed replies; ignores what the tools return."""

    model = "scripted replies"

    def __init__(self, replies: Sequence[Completion]) -> None:
        self._replies = list(replies)

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        if not self._replies:
            return _reply("(script finished)")
        return self._replies.pop(0)

    def _build_request(self, *args):
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Running before and after
# --------------------------------------------------------------------------


def memory_section(system: str) -> str:
    """The part of the system prompt that carries memory, if any."""
    marker = f"Project memory ({MEMORY_FILE}):"
    if marker not in system:
        return "(none)"
    return system[system.index(marker):].split("\n\n")[0]


def run(label: str, provider: Provider, project: str) -> dict[str, Any]:
    print(f"=== {label} ===")
    meter, recorder = Meter(provider), Recorder()
    print(f"user: {TASK}")
    run_task(meter, project, TASK, mode="safe", approver=approver,
             on_event=recorder, max_turns=15,
             extra_prompt="Always reply in English.")
    print("\nmemory section the model was given:")
    print("  " + memory_section(meter.systems[0]).replace("\n", "\n  "))
    print(f"\n{MEMORY_FILE} after this run:")
    print("  " + (load_memory(project) or "(no file)").replace("\n", "\n  "))
    print()
    return {"model calls": len(meter.sizes),
            "tool calls": recorder.tool_calls,
            "wasted calls": len(recorder.wasted),
            "wasted detail": recorder.wasted,
            "tokens sent (estimate)": sum(meter.sizes),
            "tokens sent (reported)": meter.input_tokens or "-",
            "checks pass": "yes" if checks_pass(project) else "no"}


def print_comparison(before: dict[str, Any], after: dict[str, Any]) -> None:
    print("=== before / after ===")
    rows = ["model calls", "tool calls", "wasted calls",
            "tokens sent (estimate)", "tokens sent (reported)",
            "checks pass"]
    print(f"{'':26}{'without memory':>16}{'with memory':>14}")
    for row in rows:
        print(f"{row:26}{before[row]!s:>16}{after[row]!s:>14}")
    for label, result in (("without memory", before),
                          ("with memory", after)):
        for detail in result["wasted detail"]:
            print(f"  wasted ({label}): {detail}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use scripted replies instead of a real model")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary project folder")
    args = parser.parse_args(argv)

    if args.offline:
        def provider_for(label: str) -> Provider:
            return ScriptedProvider(OFFLINE_SCRIPTS[label])
        print("provider: offline - scripted replies\n")
    else:
        try:
            live = create_provider()
        except ConfigError as error:
            print(f"error: {error}\n(or run with --offline)",
                  file=sys.stderr)
            return 1

        def provider_for(label: str) -> Provider:
            return live
        print(f"provider: {live.config.PROVIDER_NAME} - {live.model}\n")

    project = make_sandbox()
    print(f"sandbox: {project}\n")
    try:
        before = run("run 1: without memory",
                     provider_for("without memory"), project)
        reset_code(project)
        after = run("run 2: with memory",
                    provider_for("with memory"), project)
        print_comparison(before, after)
    finally:
        if args.keep:
            print(f"kept: {project}")
        else:
            shutil.rmtree(project, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
