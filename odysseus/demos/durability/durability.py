"""Demo: durability, recovery and sub-agents.

    0. Install the package once, from the odysseus project folder:
         pip install -e .

    1. Run offline first - no provider, no API key, no network:
         python3 demos/durability/durability.py --offline

    2. Run against a real model (provider and key as in the dice demo):
         python3 demos/durability/durability.py

See durability_walkthrough.html next to this file for the goal, a sample
run and an analysis of its output.

Part 1 - crash and resume. The task: apply setup steps 1, 2 and 3 with
the apply_step tool, each exactly once. A separate Python process runs
it with a session log. While step 2 runs, after it has written to
applied.log, that process kills itself with SIGKILL: no cleanup, no
exception handler, no chance to record the result. A new process then
resumes from the log alone. The check: is every step in applied.log
exactly once?

Part 2 - a sub-agent. The same question about six long note files is
answered twice: once by an agent that reads the files itself, once by an
agent that delegates the reading to a sub-agent with spawn_agent. The
table compares how big the parent's own conversation gets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from odysseus_v2 import (core_tools, create_provider, open_session,
                         run_task, tool)
from odysseus_v2.context import estimate_overhead, estimate_tokens
from odysseus_v2.providers import Completion, ConfigError, Provider
from odysseus_v2.session import INTERRUPTED
from odysseus_v2.subagent import SUBAGENT_PROMPT

EXTRA_PROMPT = "Always reply in English."

# --------------------------------------------------------------------------
# Part 1: crash and resume
# --------------------------------------------------------------------------

STEPS = 3
CRASH_STEP = 2
APPLIED_LOG = "applied.log"

STEPS_TASK = (f"Apply setup steps 1 to {STEPS} in order with the apply_step "
              "tool. Each step must be applied exactly once: applying one "
              f"twice breaks the setup. {APPLIED_LOG} records every step "
              "applied so far.")
RESUME_TASK = "Continue the task."


def steps_tools(project: str, crash: bool) -> list:
    """read_file and list_files, plus apply_step for this project."""

    @tool("Apply one setup step. Steps must be applied in order, each "
          "exactly once.", number=f"The step number, 1 to {STEPS}")
    def apply_step(number: int) -> str:
        if not 1 <= number <= STEPS:
            return f"ERROR: there is no step {number}"
        with open(os.path.join(project, APPLIED_LOG), "a",
                  encoding="utf-8") as f:
            f.write(f"step {number} applied\n")
        if crash and number == CRASH_STEP:
            print(f"  !! the process is killed while step {number} runs "
                  f"(after it wrote to {APPLIED_LOG})", flush=True)
            os.kill(os.getpid(), signal.SIGKILL)
        return f"Applied step {number}."

    reading = [t for t in core_tools(project)
               if t.name in {"read_file", "list_files"}]
    return [*reading, apply_step]


def first_process(project: str, session: str, offline: bool) -> int:
    """Run in a child process: start the task, and die at CRASH_STEP."""
    provider = ScriptedModel() if offline else create_provider()
    run_task(provider, project, STEPS_TASK, mode="yolo",
             tools=steps_tools(project, crash=True), session=session,
             on_event=print_event, extra_prompt=EXTRA_PROMPT, max_turns=12)
    print("  (the task finished before the crash step)")
    return 0


def describe(message: Mapping[str, Any]) -> str:
    """One line for a logged message."""
    role = message.get("role")
    if role == "assistant":
        calls = [f"{c['name']}({_args(c['args'])})"
                 for c in message.get("tool_calls") or []]
        text = (message.get("text") or "").strip().splitlines()
        shown = text[0][:70] if text else ""
        return "assistant " + (" + ".join(calls) if calls else shown)
    if role == "tool":
        first = (message["text"].strip().splitlines() or [""])[0]
        return f"tool      {message['name']}: {first[:70]}"
    return f"user      {message.get('text', '')[:70]}"


def show_log(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    for number, line in enumerate(lines, start=1):
        print(f"  {number:2}  {describe(json.loads(line))}")
    return len(lines)


def applied_steps(project: str) -> list[int]:
    path = os.path.join(project, APPLIED_LOG)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [int(n) for n in re.findall(r"step (\d+) applied", f.read())]


def part_one(offline: bool) -> dict[str, Any]:
    project = tempfile.mkdtemp(prefix="odysseus-durability-")
    try:
        session, _ = open_session(project, "setup steps")
        print("=== part 1a: the first process ===")
        print(f"session log: {os.path.relpath(session, project)}")
        print(f"user: {STEPS_TASK}")
        sys.stdout.flush()
        command = [sys.executable, os.path.abspath(__file__),
                   "--first-process", project, session]
        if offline:
            command.append("--offline")
        code = subprocess.run(command).returncode
        died = (f"killed by signal {-code}" if code < 0
                else f"exit code {code}")
        print(f"\nthe first process ended: {died}")
        print(f"{APPLIED_LOG}: steps {applied_steps(project)}")
        print("\nthe session log it left behind:")
        logged = show_log(session)

        print("\n=== part 1b: a new process resumes from the log ===")
        path, messages = open_session(project, resume=True)
        repaired = [m for m in messages if m.get("text") == INTERRUPTED]
        print(f"resumed {os.path.relpath(path, project)}: "
              f"{logged} messages logged, "
              f"{len(repaired)} unanswered call(s) answered:")
        for message in repaired:
            print(f"  {message['name']}: {message['text']}")
        print(f"user: {RESUME_TASK}")
        provider = ScriptedModel() if offline else create_provider()
        run_task(provider, project, RESUME_TASK, mode="yolo",
                 tools=steps_tools(project, crash=False), messages=messages,
                 session=path, on_event=print_event,
                 extra_prompt=EXTRA_PROMPT, max_turns=12)
        steps = applied_steps(project)
        with open(path, encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())
        print(f"\n{APPLIED_LOG}: steps {steps}")
        return {
            "first process": died,
            "messages logged before the kill": logged,
            "calls answered as interrupted": len(repaired),
            "messages in the log at the end": total,
            "steps applied": ", ".join(map(str, steps)) or "none",
            "every step exactly once": ("yes" if steps == list(
                range(1, STEPS + 1)) else "no"),
        }
    finally:
        shutil.rmtree(project, ignore_errors=True)


# --------------------------------------------------------------------------
# Part 2: a sub-agent
# --------------------------------------------------------------------------

NOTES = {
    "auth": None,
    "billing": "DEADLINE: 2026-12-01 - switch invoices to the new tax rates",
    "docs": None,
    "metrics": None,
    "mobile": None,
    "search": "DEADLINE: 2027-01-15 - retire the old search index",
}
EXPECTED_DATES = ("2026-12-01", "2027-01-15")
FILLER_LINES = 40
# No grep: with it, one call finds every DEADLINE line and nothing needs
# reading. The parent and the sub-agent get the same two tools.
NOTE_TOOLS = {"read_file", "list_files"}

NOTES_TASK = ("Which areas in notes/ have a deadline, and when is each "
              "due? Answer with one line per area.")
DELEGATE = (" Delegate the reading to one sub-agent with spawn_agent and "
            "answer from its report; do not read the notes yourself.")


def make_notes() -> str:
    project = tempfile.mkdtemp(prefix="odysseus-durability-")
    os.makedirs(os.path.join(project, "notes"))
    for area, deadline in NOTES.items():
        filler = "".join(
            f"Meeting note {i:03}: {area} discussed, no decision needed.\n"
            for i in range(1, FILLER_LINES + 1))
        last = deadline or "No deadline set for this area."
        with open(os.path.join(project, "notes", f"{area}.md"), "w",
                  encoding="utf-8") as f:
            f.write(f"# {area} notes\n{filler}{last}\n")
    return project


class Meter(Provider):
    """Wraps a provider; measures parent and sub-agent requests apart.

    A request belongs to a sub-agent when its system prompt carries
    SUBAGENT_PROMPT. Sizes are kept estimated (four characters per
    token) and as the vendor reported them (offline: nothing).
    """

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.sizes: dict[str, list[int]] = {"parent": [], "sub-agent": []}
        self.reported: dict[str, list[int]] = {"parent": [], "sub-agent": []}

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        who = "sub-agent" if SUBAGENT_PROMPT in system else "parent"
        self.sizes[who].append(estimate_tokens(messages)
                               + estimate_overhead(system, tools))
        reply = self.inner.complete(system, messages, tools,
                                    allow_tool_calls=allow_tool_calls)
        self.reported[who].append(reply["usage"]["input"])
        return reply

    def _build_request(self, *args):   # never called: complete is replaced
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


def run_notes(label: str, provider: Provider, delegate: bool
              ) -> dict[str, Any]:
    print(f"=== {label} ===")
    project = make_notes()
    meter = Meter(provider)
    messages: list = []
    task = NOTES_TASK + (DELEGATE if delegate else "")
    print(f"user: {task}")
    try:
        tools = [t for t in core_tools(project) if t.name in NOTE_TOOLS]
        answer = run_task(meter, project, task, mode="read-only",
                          tools=tools, subagent_depth=1 if delegate else 0,
                          messages=messages, on_event=print_event,
                          extra_prompt=EXTRA_PROMPT, max_turns=20)
    finally:
        shutil.rmtree(project, ignore_errors=True)

    vendor_reports = any(meter.reported["parent"])

    def reported(who: str) -> Any:
        return sum(meter.reported[who]) if vendor_reports else "-"

    print(f"\nparent request sizes (estimated tokens): "
          f"{' '.join(map(str, meter.sizes['parent']))}")
    if meter.sizes["sub-agent"]:
        print(f"sub-agent request sizes (estimated tokens): "
              f"{' '.join(map(str, meter.sizes['sub-agent']))}")
    if any(meter.reported["parent"]):
        print(f"parent request sizes (reported): "
              f"{' '.join(map(str, meter.reported['parent']))}")
        if meter.reported["sub-agent"]:
            print(f"sub-agent request sizes (reported): "
                  f"{' '.join(map(str, meter.reported['sub-agent']))}")
    print()
    every = meter.sizes["parent"] + meter.sizes["sub-agent"]
    return {
        "parent requests": len(meter.sizes["parent"]),
        "sub-agent requests": len(meter.sizes["sub-agent"]),
        "largest parent request (est.)": max(meter.sizes["parent"]),
        "largest parent request (rep.)": (max(meter.reported["parent"])
                                          or "-"),
        "parent messages at the end": len(messages),
        "tokens sent by parent (est.)": sum(meter.sizes["parent"]),
        "tokens sent by sub-agent (est.)": sum(meter.sizes["sub-agent"]),
        "tokens sent in all (est.)": sum(every),
        "tokens sent by parent (rep.)": reported("parent"),
        "tokens sent by sub-agent (rep.)": reported("sub-agent"),
        "both deadlines in the answer": ("yes" if all(
            date in answer for date in EXPECTED_DATES) else "no"),
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def _args(args: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def print_event(kind: str, payload: Mapping[str, Any]) -> None:
    pad = "      [sub-agent] " if payload.get("subagent") else "  "
    if kind == "assistant":
        if payload["text"]:
            lines = payload["text"].strip().splitlines()
            prefix = pad.rstrip() + " " if payload.get("subagent") else ""
            print(f"{prefix}assistant: " + f"\n{prefix}  ".join(lines))
        usage = payload["usage"]
        if usage["input"] or usage["output"]:
            print(f"{pad}[tokens in {usage['input']}, "
                  f"out {usage['output']}]")
    elif kind == "tool_start":
        print(f"{pad}-> {payload['name']}({_args(payload['args'])[:110]})")
    elif kind == "tool_end":
        lines = [ln for ln in payload["result"].splitlines() if ln.strip()]
        lines = lines or ["(empty)"]
        shown = lines[0][:100]
        if len(lines) > 1:
            shown += f"  ...  ({len(lines)} lines)"
        print(f"{pad}{'!!' if payload['is_error'] else '<-'} {shown}")


def print_table(title: str, columns: Sequence[str],
                rows: Sequence[Mapping[str, Any]]) -> None:
    print(f"=== {title} ===")
    width = max(len(key) for key in rows[0]) + 2
    print(f"{'':{width}}" + "".join(f"{c:>22}" for c in columns))
    for key in rows[0]:
        print(f"{key:{width}}" + "".join(f"{row[key]!s:>22}"
                                         for row in rows))


# --------------------------------------------------------------------------
# Offline mode: a scripted model
# --------------------------------------------------------------------------


def _reply(text: str = "", *calls: tuple[str, dict[str, Any]]) -> Completion:
    tool_calls = [{"id": f"s{i}", "name": name, "args": args}
                  for i, (name, args) in enumerate(calls, start=1)]
    return Completion(text=text, tool_calls=tool_calls,
                      usage={"input": 0, "output": 0})


class ScriptedModel(Provider):
    """Decides from the conversation it is sent, like a real model.

    Part 1: applies the next step; after an interrupted call it reads
    applied.log and continues from what it finds there. Part 2: with
    spawn_agent it delegates; without, it lists and reads the notes and
    answers from the DEADLINE lines it read.
    """

    model = "scripted replies"

    def __init__(self) -> None:     # no config: nothing is sent anywhere
        pass

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        names = {t["name"] for t in tools}
        results = [m for m in messages if m.get("role") == "tool"]
        if "apply_step" in names:
            return self._steps(results)
        if "spawn_agent" in names:
            if not results:
                return _reply("", ("spawn_agent", {"task": (
                    "Read every file in notes/. Report each area whose "
                    "file has a DEADLINE line: the area, the date and "
                    "what is due. Skip areas without one.")}))
            return _reply(results[-1]["text"])
        return self._read_notes(results)

    @staticmethod
    def _steps(results: list[Mapping[str, Any]]) -> Completion:
        if not results:
            return _reply("", ("apply_step", {"number": 1}))
        last = results[-1]
        if last["text"] == INTERRUPTED:
            return _reply("That step was interrupted; checking what was "
                          "applied before repeating anything.",
                          ("read_file", {"path": APPLIED_LOG}))
        if last["name"] == "read_file":
            done = [int(n) for n in re.findall(r"step (\d+) applied",
                                               last["text"])]
            following = max(done, default=0) + 1
        else:
            following = int(re.search(r"\d+", last["text"]).group()) + 1
        if following > STEPS:
            return _reply(f"All {STEPS} steps are applied, each once.")
        return _reply("", ("apply_step", {"number": following}))

    @staticmethod
    def _read_notes(results: list[Mapping[str, Any]]) -> Completion:
        if not results:
            return _reply("", ("list_files", {"pattern": "notes/*"}))
        if len(results) == 1:
            paths = results[0]["text"].split()
            return _reply("", *[("read_file", {"path": p}) for p in paths])
        found = []
        for result in results[1:]:
            area = re.search(r"# (\w+) notes", result["text"])
            deadline = re.search(r"DEADLINE: (.*)", result["text"])
            if area and deadline:
                found.append(f"{area.group(1)}: {deadline.group(1)}")
        return _reply("\n".join(found) or "No area has a deadline.")

    def _build_request(self, *args):
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use scripted replies instead of a real model")
    parser.add_argument("--first-process", nargs=2,
                        metavar=("PROJECT", "SESSION"),
                        help=argparse.SUPPRESS)   # part 1's child process
    args = parser.parse_args(argv)

    if args.first_process:
        return first_process(*args.first_process, offline=args.offline)

    if args.offline:
        provider: Provider = ScriptedModel()
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

    resumed = part_one(args.offline)
    print()
    print_table("part 1: crash and resume", ["result"], [resumed])
    print()

    alone = run_notes("part 2a: the agent reads the notes itself",
                      provider, delegate=False)
    helped = run_notes("part 2b: the agent delegates to a sub-agent",
                       provider, delegate=True)
    print_table("part 2: before / after", ["reads itself",
                                           "with a sub-agent"],
                [alone, helped])
    return 0


if __name__ == "__main__":
    sys.exit(main())
