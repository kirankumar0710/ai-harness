"""Memory and the system prompt.

The system prompt is infrastructure, not prose: identity, rules, and
whatever the project remembers about itself. Anything written to
AGENT_MEMORY.md in the working directory is loaded into every session. That
one convention is the whole memory system.

    system = build_system_prompt(workdir, catalog_prompt(workdir))
    tools = [*core_tools(workdir), *memory_tools(workdir)]

Design rules:

* Memory is a plain file the user can read, edit and commit.
* The model adds to it only through the ``remember`` tool, so the gate
  decides whether it may: blocked in read-only mode, approved in safe.
* AGENT_MEMORY.md is trusted: it goes into the system prompt as written.
  Treat it like code: review what a model writes there.
"""

from __future__ import annotations

import os
import platform

from .tools import Tool, tool

MEMORY_FILE = "AGENT_MEMORY.md"
MAX_MEMORY_CHARS = 20_000       # cap, so memory cannot eat the budget
MAX_NOTE_CHARS = 500

BASE_PROMPT = """\
You are Odysseus, a small, sharp coding agent. You work inside one \
working directory using the tools provided.

Rules:
- Act, don't narrate. Use tools to inspect before you assume.
- Prefer edit_file for small changes; write_file for new or rewritten files.
- After building something, verify it: run it, test it, or read it back.
- If a tool call fails, read the error and fix the cause; never repeat a \
failing call unchanged.
- When the task is complete, reply with a short plain-text summary of what \
you built and how you verified it, and stop calling tools.
"""


def build_system_prompt(workdir: str | os.PathLike[str],
                        *extra: str) -> str:
    """Assemble the system prompt: rules, environment, memory, extras.

    ``extra`` sections (such as the skills catalog) are appended in
    order; empty ones are skipped.
    """
    workdir = os.path.realpath(workdir)
    sections = [
        BASE_PROMPT,
        f"Environment: {platform.system()}, working directory: {workdir}",
    ]
    memory = load_memory(workdir)
    if memory:
        sections.append(f"Project memory ({MEMORY_FILE}):\n{memory}")
    sections.extend(section for section in extra if section)
    return "\n\n".join(sections)


def load_memory(workdir: str | os.PathLike[str]) -> str:
    """Return the project memory, capped at ``MAX_MEMORY_CHARS``."""
    path = os.path.join(workdir, MEMORY_FILE)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if len(text) > MAX_MEMORY_CHARS:
        # Keep the end: the newest notes are appended last.
        text = ("... (older notes truncated)\n"
                + text[-MAX_MEMORY_CHARS:])
    return text


def memory_tools(workdir: str | os.PathLike[str]) -> list[Tool]:
    """Return the ``remember`` tool, writing to ``workdir``/AGENT_MEMORY.md."""
    path = os.path.join(os.path.realpath(workdir), MEMORY_FILE)

    @tool("Save a short fact to project memory (AGENT_MEMORY.md). It is "
          "loaded into every future session, so save only what will "
          "still matter: conventions, decisions, how to run things.",
          note="One fact, in one line")
    def remember(note: str) -> str:
        note = " ".join(note.split())        # one line, tidy spacing
        if not note:
            return "ERROR: the note is empty"
        if len(note) > MAX_NOTE_CHARS:
            return (f"ERROR: the note is {len(note)} characters; keep it "
                    f"under {MAX_NOTE_CHARS}")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"- {note}\n")
        return f"Remembered in {MEMORY_FILE}"

    return [remember]
