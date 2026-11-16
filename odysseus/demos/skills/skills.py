"""Demo: skills - instructions for a kind of task, loaded only when needed.

    0. Install the package once, from the odysseus project folder:
         pip install -e .

    1. Run offline first - no provider, no API key, no network. A
       scripted "model" makes the calls a real one should:
         python3 demos/skills/skills.py --offline

    2. Run against a real model. Choose a provider and export its key
       (see demos/dice/dice.py for the variable names), then:
         export ODYSSEUS_PROVIDER=anthropic   # or gemini / openai / ...
         export ANTHROPIC_API_KEY=...
         python3 demos/skills/skills.py

See skills_walkthrough.html next to this file for the goal, a sample run
and an analysis of its output.

The two skills are ordinary files, checked in next to this script:

    example_skills/release-note/SKILL.md      how to write a release note
    example_skills/review-migration/SKILL.md  how to review a migration

The demo copies them into a throwaway project with a fixed calc.py, an
empty RELEASE_NOTES.md and one database migration, then runs three tasks
through run_task():

    1. "add a release note"   should load release-note only
    2. "a quick question"     should load no skill at all
    3. "review the migration" should load review-migration only

After each task the script records which skills the model loaded and
checks the result against the skill's format. The model only ever sees
the skills' one-line descriptions until it asks for one with use_skill.

The folder is deleted afterwards unless --keep is given.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from odysseus_v2 import create_provider, run_task
from odysseus_v2.providers import Completion, ConfigError, Provider
from odysseus_v2.skills import catalog, catalog_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_SKILLS = os.path.join(HERE, "example_skills")
TODAY = datetime.date.today().isoformat()

CALC_PY = '''\
def average(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    return sum(values) / len(values)
'''

MIGRATION_SQL = """\
-- 0002: give users an email address and link orders to users
ALTER TABLE users ADD COLUMN email TEXT NOT NULL;
ALTER TABLE orders ADD COLUMN user_id INTEGER REFERENCES users(id);
"""

PROJECT_FILES = {
    "calc.py": CALC_PY,
    "RELEASE_NOTES.md": "# Release notes\n",
    "migrations/0002_add_email.up.sql": MIGRATION_SQL,
}

TASKS = (
    ("release note",
     "We fixed a bug in calc.py: average() divided by len(values) - 1 "
     "instead of len(values), so every average came out too high. Add a "
     f"release note for version 1.0.1, dated {TODAY}."),
    ("question",
     "In one sentence: what does average() in calc.py return?"),
    ("migration review",
     "Review the migration migrations/0002_add_email.up.sql."),
)

# What each task should look like if the model followed the right skill.
RELEASE_ENTRY = re.compile(
    r"^## 1\.0\.1 - " + re.escape(TODAY) + r"\n### Fixed\n- .+\(calc\.py\)"
    r"\nUpgrade: .+", re.MULTILINE)
REVIEW_LINE = re.compile(
    r"^\s*\d\. [A-Za-z]+: (PASS|FAIL) - .+", re.MULTILINE)
VERDICT = re.compile(r"^\s*Verdict: (APPROVE|REQUEST CHANGES)\s*$",
                     re.MULTILINE)

EXPECTED_SKILLS = {
    "release note": ["release-note"],
    "question": [],
    "migration review": ["review-migration"],
}


# --------------------------------------------------------------------------
# Sandbox and approver
# --------------------------------------------------------------------------


def make_sandbox() -> str:
    """Create the throwaway project and copy the skills into it."""
    project = tempfile.mkdtemp(prefix="odysseus-skills-")
    for relative, content in PROJECT_FILES.items():
        path = os.path.join(project, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    shutil.copytree(EXAMPLE_SKILLS, os.path.join(project, "skills"))
    return project


def file_approver(call: Mapping[str, Any], reason: str) -> bool:
    """Approve file writes inside the jail; refuse everything else."""
    approved = call["name"] in {"write_file", "edit_file"}
    print(f"  ? approver: {call['name']} -> {'yes' if approved else 'no'}")
    return approved


# --------------------------------------------------------------------------
# Offline mode: a scripted model
# --------------------------------------------------------------------------


def _reply(text: str = "", *calls: tuple[str, dict[str, Any]]) -> Completion:
    tool_calls = [{"id": f"s{i}", "name": name, "args": args}
                  for i, (name, args) in enumerate(calls, start=1)]
    return Completion(text=text, tool_calls=tool_calls,
                      usage={"input": 0, "output": 0})


OFFLINE_NOTES = (
    "# Release notes\n\n"
    f"## 1.0.1 - {TODAY}\n"
    "### Fixed\n"
    "- Averages were too high because one value was left out of the "
    "count (calc.py)\n"
    "Upgrade: no action needed\n")

OFFLINE_REVIEW = (
    "1. Reversible: FAIL - there is no 0002_add_email.down.sql\n"
    "2. Locking: FAIL - users.email is NOT NULL with no DEFAULT\n"
    "3. Indexes: FAIL - orders.user_id is a foreign key with no index\n"
    "4. Data: PASS - no rows are deleted or rewritten\n"
    "Verdict: REQUEST CHANGES")

OFFLINE_SCRIPTS: dict[str, list[Completion]] = {
    "release note": [
        _reply("", ("use_skill", {"name": "release-note"})),
        _reply("", ("write_file", {"path": "RELEASE_NOTES.md",
                                   "content": OFFLINE_NOTES})),
        _reply("Added the 1.0.1 entry to RELEASE_NOTES.md."),
    ],
    "question": [
        _reply("", ("read_file", {"path": "calc.py"})),
        _reply("It returns the arithmetic mean: the sum of the values "
               "divided by how many there are."),
    ],
    "migration review": [
        _reply("", ("use_skill", {"name": "review-migration"}),
               ("read_file", {"path": "migrations/0002_add_email.up.sql"}),
               ("list_files", {"pattern": "migrations/*"})),
        _reply(OFFLINE_REVIEW),
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

    def _build_request(self, *args):   # never called: complete is replaced
        raise NotImplementedError

    def _parse_response(self, raw):
        raise NotImplementedError


# --------------------------------------------------------------------------
# Running the tasks
# --------------------------------------------------------------------------


class Recorder:
    """Prints the run and remembers which skills each task loaded."""

    def __init__(self) -> None:
        self.skills_loaded: list[str] = []

    def __call__(self, kind: str, payload: Mapping[str, Any]) -> None:
        if kind == "assistant":
            if payload["text"]:
                print(f"assistant: {payload['text']}")
            usage = payload["usage"]
            if usage["input"] or usage["output"]:
                print(f"  [tokens in {usage['input']}, "
                      f"out {usage['output']}]")
        elif kind == "tool_start":
            print(f"  -> {payload['name']}({_short(payload['args'])})")
            if payload["name"] == "use_skill":
                self.skills_loaded.append(str(payload["args"].get("name")))
        elif kind == "tool_end":
            lines = [ln for ln in payload["result"].splitlines()
                     if ln.strip()] or ["(empty)"]
            shown = lines[0][:100]
            if len(lines) > 1:
                shown += f"  ...  {lines[-1][:60]}  ({len(lines)} lines)"
            mark = "!!" if payload["is_error"] else "<-"
            print(f"  {mark} {shown}")


def _short(args: Mapping[str, Any]) -> str:
    """Arguments for display: long values are cut to one line."""
    parts = []
    for key, value in args.items():
        text = repr(value)
        if key != "command" and len(text) > 60:   # commands in full
            text = text[:57].replace("\\n", " ") + "...'"
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def check(label: str, answer: str, project: str) -> str:
    """Did the result follow the skill's format? Returns a short verdict."""
    if label == "release note":
        with open(os.path.join(project, "RELEASE_NOTES.md"),
                  encoding="utf-8") as f:
            notes = f.read()
        ok = bool(RELEASE_ENTRY.search(notes))
        return f"entry in the skill's format: {'yes' if ok else 'no'}"
    if label == "migration review":
        lines = len(REVIEW_LINE.findall(answer))
        verdict = VERDICT.search(answer)
        ok = lines == 4 and verdict is not None
        found = verdict.group(1) if verdict else "missing"
        return (f"4 checks + verdict: {'yes' if ok else 'no'} "
                f"({lines} check lines, verdict {found})")
    return "no format to check"


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
            provider = create_provider()
        except ConfigError as error:
            print(f"error: {error}\n(or run with --offline)",
                  file=sys.stderr)
            return 1

        def provider_for(label: str) -> Provider:
            return provider
        print(f"provider: {provider.config.PROVIDER_NAME} - "
              f"{provider.model}\n")

    project = make_sandbox()
    on_disk = sum(os.path.getsize(info["path"])
                  for info in catalog(project).values())
    in_prompt = len(catalog_prompt(project))
    print(f"sandbox: {project}")
    print(f"skills: {', '.join(catalog(project))}  "
          f"({on_disk} chars on disk, {in_prompt} chars in every "
          "system prompt)\n")

    summary = []
    try:
        for number, (label, task) in enumerate(TASKS, start=1):
            print(f"=== task {number}: {label} ===")
            print(f"user: {task}")
            recorder = Recorder()
            answer = run_task(provider_for(label), project, task,
                              mode="safe", approver=file_approver,
                              on_event=recorder, max_turns=12,
                              extra_prompt="Always reply in English.")
            verdict = check(label, answer, project)
            expected = EXPECTED_SKILLS[label]
            summary.append((number, label, recorder.skills_loaded,
                            recorder.skills_loaded == expected, verdict))
            print()
    finally:
        print("=== summary ===")
        for number, label, loaded, as_expected, verdict in summary:
            print(f"task {number} ({label}): skills loaded "
                  f"{loaded or 'none'} - "
                  f"{'as expected' if as_expected else 'NOT as expected'}"
                  f"; {verdict}")
        if args.keep:
            print(f"kept: {project}")
        else:
            shutil.rmtree(project, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
