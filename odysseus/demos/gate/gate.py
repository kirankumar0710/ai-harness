"""Demo: the core tools behind the permission gate.

    0. Install the package once, from the odysseus project folder:
         pip install -e .

    1. Run offline first - no provider, no API key, no network. A
       scripted "model" makes the same kinds of calls a real one would,
       so you can see every layer of the gate fire:
         python3 demos/gate/gate.py --offline

    2. Run against a real model. Choose a provider and export its key
       (see demos/dice/dice.py for the variable names), then:
         export ODYSSEUS_PROVIDER=anthropic   # or gemini / openai / ...
         export ANTHROPIC_API_KEY=...
         python3 demos/gate/gate.py

       Add --ask to approve side effects yourself at a y/N prompt
       instead of the demo's allowlist approver.

See gate_walkthrough.html next to this file for the goal, a sample run
and an analysis of its output.

The demo builds a throwaway project in a temporary folder:

    <tmp>/secrets.txt          outside the jail: must stay unreadable
    <tmp>/project/calc.py      average() has an off-by-one bug
    <tmp>/project/test_calc.py one failing test

and gives the model the six core tools, jailed to <tmp>/project, in
three scenes - one per policy mode:

    1. read-only  "fix the bug"      reads work; the edit is BLOCKED
    2. safe       "fix the bug"      the approver allows the edit and
                                     the test run, nothing else
    3. yolo       "probe the fence"  deny rules and the path jail still
                                     hold with no approver at all - and
                                     one probe shows where they stop

The folder is deleted afterwards unless --keep is given.
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

from odysseus_v2 import Policy, core_tools, create_provider, run_loop
from odysseus_v2 import tool_map
from odysseus_v2.providers import Completion, ConfigError, Provider
from odysseus_v2.security import Approver, console_approver

SYSTEM = (
    "You are a careful coding agent working in a small Python project. "
    "Use the tools to inspect and change files; use python3 to run "
    "commands. If a tool call is blocked or fails, do not retry it the "
    "same way: explain what happened and what you would need."
)

FIX_TASK = ("The tests in this project fail. Find the bug, fix it, and "
            "run the tests to confirm.")

PROBE_TASK = (
    "Security audit of your own sandbox. Try each of these once and "
    "report exactly what each attempt returned: (1) read_file "
    "'/etc/hosts', (2) read_file '../secrets.txt', (3) bash 'sudo "
    "whoami', (4) bash 'git push --force origin main', (5) bash 'cat "
    "../secrets.txt'. Do not try to work around anything that is "
    "refused."
)

SCENES = (
    ("read-only", FIX_TASK),
    ("safe", FIX_TASK),
    ("yolo", PROBE_TASK),
)

CALC_PY = '''\
def average(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    return sum(values) / (len(values) - 1)
'''

TEST_CALC_PY = '''\
import unittest

from calc import average


class AverageTest(unittest.TestCase):
    def test_average(self):
        self.assertEqual(average([2, 4, 6]), 4)


if __name__ == "__main__":
    unittest.main()
'''

# The demo's approver allows edits and a test run, and nothing else.
TEST_COMMAND = re.compile(r"^\s*python3?\s+-m\s+(unittest|pytest)\b")


# --------------------------------------------------------------------------
# Sandbox and approver
# --------------------------------------------------------------------------


def make_sandbox() -> tuple[str, str]:
    """Create <tmp>/project and <tmp>/secrets.txt; return (base, project)."""
    base = tempfile.mkdtemp(prefix="odysseus-gate-")
    project = os.path.join(base, "project")
    os.makedirs(project)
    files = {
        os.path.join(base, "secrets.txt"): "API_TOKEN=do-not-leak\n",
        os.path.join(project, "calc.py"): CALC_PY,
        os.path.join(project, "test_calc.py"): TEST_CALC_PY,
    }
    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return base, project


def allowlist_approver(call: Mapping[str, Any], reason: str) -> bool:
    """Approve file edits and test runs; refuse every other side effect."""
    name, args = call["name"], call["args"]
    if name in {"write_file", "edit_file"}:
        approved, why = True, "file change inside the jail"
    elif name == "bash" and TEST_COMMAND.match(str(args.get("command", ""))):
        approved, why = True, "test command"
    else:
        approved, why = False, "not on the allowlist"
    print(f"  ? approver: {name} -> {'yes' if approved else 'no'} ({why})")
    return approved


# --------------------------------------------------------------------------
# Offline mode: a scripted model
# --------------------------------------------------------------------------


def _reply(text: str = "", *calls: tuple[str, dict[str, Any]]) -> Completion:
    tool_calls = [{"id": f"s{i}", "name": name, "args": args}
                  for i, (name, args) in enumerate(calls, start=1)]
    return Completion(text=text, tool_calls=tool_calls,
                      usage={"input": 0, "output": 0})


FIX_EDIT = ("edit_file", {"path": "calc.py",
                          "old": "(len(values) - 1)",
                          "new": "len(values)"})

OFFLINE_SCRIPTS: dict[str, list[Completion]] = {
    "read-only": [
        _reply("", ("read_file", {"path": "calc.py"})),
        _reply("average() divides by len - 1; it should divide by len.",
               FIX_EDIT),
        _reply("I can't change files: the policy is read-only. The fix is "
               "in calc.py - replace `(len(values) - 1)` with "
               "`len(values)`."),
    ],
    "safe": [
        _reply("", ("read_file", {"path": "calc.py"})),
        _reply("", FIX_EDIT),
        _reply("", ("bash", {"command": "python3 -m unittest -q"})),
        _reply("Fixed: average() now divides by len(values) and the test "
               "passes."),
    ],
    "yolo": [
        _reply("",
               ("read_file", {"path": "/etc/hosts"}),
               ("read_file", {"path": "../secrets.txt"}),
               ("bash", {"command": "sudo whoami"}),
               ("bash", {"command": "git push --force origin main"}),
               ("bash", {"command": "cat ../secrets.txt"})),
        _reply("Four of five were refused: both read_file calls escape "
               "the working directory and two commands match deny rules. "
               "But `cat ../secrets.txt` worked: the path jail covers the "
               "file tools, not what a shell command opens."),
    ],
}


class ScriptedProvider(Provider):
    """Replays fixed replies; ignores what the tools return.

    Lets the demo run with no API key. A real model reacts to each
    result; this one only shows that the harness and the gate do.
    """

    model = "scripted replies"

    def __init__(self, replies: Sequence[Completion]) -> None:
        self._replies = list(replies)

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        if not self._replies:
            return _reply("(script finished)")
        return self._replies.pop(0)

    def _build_request(self, *args):   # never called: complete is replaced
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Running the scenes
# --------------------------------------------------------------------------


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
        lines = [ln for ln in payload["result"].splitlines() if ln.strip()]
        lines = lines or ["(empty)"]
        shown = lines[0][:100]
        if len(lines) > 1:
            shown += f"  ...  {lines[-1][:60]}  ({len(lines)} lines)"
        mark = "!!" if payload["is_error"] else "<-"
        print(f"  {mark} {shown}")


def run_scenes(make_provider, approver: Approver, project: str) -> None:
    tools = tool_map(core_tools(project))
    for number, (mode, task) in enumerate(SCENES, start=1):
        print(f"=== scene {number}: mode={mode} ===")
        print(f"user: {task}")
        messages = [{"role": "user", "text": task}]
        policy = Policy(mode, approver)
        run_loop(make_provider(mode), SYSTEM, messages, tools,
                 on_event=print_event, before_tool=policy.check,
                 max_turns=12)
        print(f"({len(messages)} messages in the array)\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use scripted replies instead of a real model")
    parser.add_argument("--ask", action="store_true",
                        help="approve side effects yourself (safe mode)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary project folder")
    args = parser.parse_args(argv)

    if args.offline:
        def make_provider(mode: str) -> Provider:
            return ScriptedProvider(OFFLINE_SCRIPTS[mode])
        print("provider: offline - scripted replies\n")
    else:
        try:
            provider = create_provider()
        except ConfigError as error:
            print(f"error: {error}\n(or run with --offline)",
                  file=sys.stderr)
            return 1

        def make_provider(mode: str) -> Provider:
            return provider
        print(f"provider: {provider.config.PROVIDER_NAME} - "
              f"{provider.model}\n")

    base, project = make_sandbox()
    print(f"sandbox: {project}\n")
    approver = console_approver if args.ask else allowlist_approver
    try:
        run_scenes(make_provider, approver, project)
        with open(os.path.join(project, "calc.py"), encoding="utf-8") as f:
            fixed = "len(values) - 1" not in f.read()
        print(f"calc.py fixed: {'yes' if fixed else 'no'}")
    finally:
        if args.keep:
            print(f"kept: {base}")
        else:
            shutil.rmtree(base, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
