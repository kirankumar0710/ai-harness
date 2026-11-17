"""Demo: a fleet - three agents building three small projects in parallel.

    0. Install the package once, from the odysseus project folder:
         pip install -e .

    1. Run offline first - no provider, no API key, no network:
         python3 demos/fleet/fleet.py --offline

    2. Run against a real model (provider and key as in the dice demo):
         python3 demos/fleet/fleet.py [--workers 3]

See fleet_walkthrough.html next to this file for the goal, a sample run
and an analysis of its output.

Each project starts as a folder holding one test file, the spec: build
slugify.py, roman.py or wordfreq.py so that its tests pass. run_fleet()
gives each project its own agent and runs them at the same time, each
jailed to its own folder with its own session log.

The agents work in safe mode. The approver allows writing files and
running the project's tests, nothing else. When every job is done the
demo runs each project's tests itself: it does not take an agent's word
for it. The summary compares wall-clock time with the sum of the jobs'
times.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

from odysseus_v2 import Job, create_provider, run_fleet
from odysseus_v2.providers import Completion, ConfigError, Provider

TEST_COMMAND = "python3 -m unittest -v"
EXTRA_PROMPT = (f"Run the tests with: {TEST_COMMAND}. Always reply in "
                "English.")

PROJECTS: dict[str, dict[str, str]] = {
    "slugify": {
        "task": "Build slugify.py so that the tests in test_slugify.py "
                "pass. Do not change the tests.",
        "tests": '''\
import unittest

from slugify import slugify


class SlugifyTest(unittest.TestCase):
    def test_words_become_lowercase_and_hyphenated(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation_is_dropped(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_runs_of_separators_collapse(self):
        self.assertEqual(slugify("  a -- b__c  "), "a-b-c")

    def test_empty(self):
        self.assertEqual(slugify("!!!"), "")


if __name__ == "__main__":
    unittest.main()
''',
        "solution": '''\
import re


def slugify(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words)
''',
    },
    "roman": {
        "task": "Build roman.py so that the tests in test_roman.py pass. "
                "Do not change the tests.",
        "tests": '''\
import unittest

from roman import from_roman, to_roman


class RomanTest(unittest.TestCase):
    def test_to_roman(self):
        self.assertEqual(to_roman(4), "IV")
        self.assertEqual(to_roman(1994), "MCMXCIV")
        self.assertEqual(to_roman(3999), "MMMCMXCIX")

    def test_from_roman(self):
        self.assertEqual(from_roman("XLII"), 42)
        self.assertEqual(from_roman("MCMXCIV"), 1994)

    def test_round_trip(self):
        for n in range(1, 4000):
            self.assertEqual(from_roman(to_roman(n)), n)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            to_roman(0)
        with self.assertRaises(ValueError):
            to_roman(4000)


if __name__ == "__main__":
    unittest.main()
''',
        "solution": '''\
NUMERALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(number: int) -> str:
    if not 1 <= number <= 3999:
        raise ValueError(f"out of range: {number}")
    out = ""
    for value, numeral in NUMERALS:
        while number >= value:
            out += numeral
            number -= value
    return out


def from_roman(text: str) -> int:
    total, i = 0, 0
    for value, numeral in NUMERALS:
        while text.startswith(numeral, i):
            total += value
            i += len(numeral)
    return total
''',
    },
    "wordfreq": {
        "task": "Build wordfreq.py so that the tests in test_wordfreq.py "
                "pass. Do not change the tests.",
        "tests": '''\
import unittest

from wordfreq import top_words


class TopWordsTest(unittest.TestCase):
    def test_counts_case_insensitively(self):
        self.assertEqual(top_words("The the THE cat", 1), [("the", 3)])

    def test_ties_are_alphabetical(self):
        self.assertEqual(top_words("b a c b a", 2), [("a", 2), ("b", 2)])

    def test_apostrophes_stay_in_words(self):
        self.assertEqual(top_words("don't stop, don't", 1),
                         [("don't", 2)])

    def test_n_larger_than_vocabulary(self):
        self.assertEqual(top_words("one", 5), [("one", 1)])


if __name__ == "__main__":
    unittest.main()
''',
        "solution": '''\
import re
from collections import Counter


def top_words(text: str, n: int) -> list[tuple[str, int]]:
    counts = Counter(re.findall(r"[a-z']+", text.lower()))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:n]
''',
    },
}

# The approver: writing files is fine (the jail keeps each agent in its
# own folder); bash only for the test command, optionally with output
# trimmed. The same rule as the gate demo.
TEST_ONLY = re.compile(
    r"\s*python3?\s+-m\s+unittest\b[^;&|`$<>()\n]*"
    r"(2>&1)?\s*(\|\s*(tail|head)((\s+-n)?\s+-?\d+)?)?\s*")
WRITE_TOOLS = {"write_file", "edit_file"}


def approver(call: Mapping[str, Any], reason: str) -> bool | str:
    if call["name"] in WRITE_TOOLS:
        return True
    if call["name"] == "bash" and TEST_ONLY.fullmatch(
            call["args"].get("command", "")):
        return True
    return ("only writing files and running the tests "
            f"({TEST_COMMAND}) are approved")


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


class Printer:
    """Prints each job's events with its name; counts tokens per job."""

    def __init__(self) -> None:
        self.tokens: dict[str, list[int]] = {}
        self.started = time.monotonic()

    def __call__(self, kind: str, payload: Mapping[str, Any]) -> None:
        job = payload["job"]
        stamp = f"{time.monotonic() - self.started:5.1f}s [{job:8}]"
        if kind == "assistant":
            usage = payload["usage"]
            totals = self.tokens.setdefault(job, [0, 0, 0])
            totals[0] += usage["input"]
            totals[1] += usage["output"]
            totals[2] += 1
            text = (payload["text"] or "").strip()
            if text:
                print(f"{stamp} assistant: {text.splitlines()[0][:100]}")
        elif kind == "tool_start":
            args = payload["args"]
            shown = args.get("command") or args.get("path") or ""
            print(f"{stamp} -> {payload['name']}({shown})")
        elif kind == "tool_end":
            lines = [ln for ln in payload["result"].splitlines()
                     if ln.strip()] or ["(empty)"]
            last = lines[-1][:80]
            mark = "!!" if payload["is_error"] else "<-"
            print(f"{stamp} {mark} {last}"
                  + (f"  ({len(lines)} lines)" if len(lines) > 1 else ""))


def check_project(folder: str, name: str) -> tuple[bool, str]:
    """Run the project's tests ourselves; return (passed, summary)."""
    done = subprocess.run(
        [sys.executable, "-m", "unittest", "-q"], cwd=folder,
        capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE":
             "1"})
    last = (done.stderr.strip().splitlines() or ["(no output)"])[-1]
    tests_intact = _read(os.path.join(folder, f"test_{name}.py")) == \
        PROJECTS[name]["tests"]
    return done.returncode == 0 and tests_intact, (
        last if tests_intact else "the test file was changed")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Offline mode: a scripted model
# --------------------------------------------------------------------------


def _reply(text: str = "", *calls: tuple[str, dict[str, Any]]) -> Completion:
    tool_calls = [{"id": f"s{i}", "name": name, "args": args}
                  for i, (name, args) in enumerate(calls, start=1)]
    return Completion(text=text, tool_calls=tool_calls,
                      usage={"input": 0, "output": 0})


class ScriptedModel(Provider):
    """Reads the tests, writes the module, runs the tests, reports.

    Stateless, so one instance serves all three jobs. Each request
    sleeps OFFLINE_LATENCY seconds to stand in for a real model's
    response time; without it there would be nothing to run in
    parallel.
    """

    model = "scripted replies"
    OFFLINE_LATENCY = 0.5

    def __init__(self) -> None:     # no config: nothing is sent anywhere
        pass

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True) -> Completion:
        time.sleep(self.OFFLINE_LATENCY)
        name = re.search(r"Build (\w+)\.py", messages[0]["text"]).group(1)
        results = [m for m in messages if m.get("role") == "tool"]
        if not results:
            return _reply("", ("read_file", {"path": f"test_{name}.py"}))
        if len(results) == 1:
            return _reply("", ("write_file", {
                "path": f"{name}.py",
                "content": PROJECTS[name]["solution"]}))
        if len(results) == 2:
            return _reply("", ("bash", {"command": TEST_COMMAND}))
        return _reply(f"{name}.py is written and its tests pass.")

    def _build_request(self, *args):
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def make_projects() -> tuple[str, list[Job]]:
    base = tempfile.mkdtemp(prefix="odysseus-fleet-")
    jobs = []
    for name, project in PROJECTS.items():
        folder = os.path.join(base, name)
        os.makedirs(folder)
        with open(os.path.join(folder, f"test_{name}.py"), "w",
                  encoding="utf-8") as f:
            f.write(project["tests"])
        jobs.append(Job(name, folder, project["task"]))
    return base, jobs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true",
                        help="use scripted replies instead of a real model")
    parser.add_argument("--workers", type=int, default=len(PROJECTS),
                        help="jobs to run at once (default: all)")
    args = parser.parse_args(argv)

    if args.offline:
        provider: Provider = ScriptedModel()
        print("provider: offline - scripted replies "
              f"({ScriptedModel.OFFLINE_LATENCY} s per request)\n")
    else:
        try:
            provider = create_provider()
        except ConfigError as error:
            print(f"error: {error}\n(or run with --offline)",
                  file=sys.stderr)
            return 1
        print(f"provider: {provider.config.PROVIDER_NAME} - "
              f"{provider.model}\n")

    base, jobs = make_projects()
    print(f"projects: {', '.join(job.name for job in jobs)} "
          f"(workers: {args.workers})")
    for job in jobs:
        print(f"  {job.name:8} {job.task}")
    print()
    printer = Printer()
    started = time.monotonic()
    try:
        results = run_fleet(provider, jobs, mode="safe", approver=approver,
                            max_workers=args.workers, on_event=printer,
                            sessions=True, max_turns=20,
                            extra_prompt=EXTRA_PROMPT)
        wall = time.monotonic() - started
        checks = {job.name: check_project(job.workdir, job.name)
                  for job in jobs}
        logged = {r.name: sum(1 for _ in open(r.session, encoding="utf-8"))
                  for r in results if r.session}
    finally:
        shutil.rmtree(base, ignore_errors=True)

    print("\n=== results ===")
    print(f"{'job':10}{'agent':>8}{'seconds':>9}{'requests':>10}"
          f"{'tokens in':>11}{'logged':>8}  tests (run by the demo)")
    for result in results:
        tokens = printer.tokens.get(result.name, [0, 0, 0])
        passed, summary = checks[result.name]
        print(f"{result.name:10}{'ok' if result.ok else 'error':>8}"
              f"{result.seconds:>9.1f}{tokens[2]:>10}"
              f"{tokens[0] or '-':>11}{logged.get(result.name, 0):>8}  "
              f"{'pass' if passed else 'FAIL'}: {summary}")
    total = sum(r.seconds for r in results)
    print(f"\nwall-clock time: {wall:.1f} s; the jobs' times add up to "
          f"{total:.1f} s ({total / wall:.1f}x)")
    print(f"projects passing: {sum(p for p, _ in checks.values())} of "
          f"{len(checks)}")
    for result in results:
        if not result.ok:
            print(f"{result.name}: {result.answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
