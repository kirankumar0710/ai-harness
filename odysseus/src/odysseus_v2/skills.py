"""Skills: teachable expertise, no code changes.

A skill is a folder with a SKILL.md: a name (the folder), a one-line
description and instructions. The harness lists the catalog in the
system prompt; the model pulls a skill's full text into context with the
``use_skill`` tool when the task calls for it. This is how a bare-bones
core grows expertise per project.

    skills/
      testing/
        SKILL.md      ---
                      description: How to write and run this project's tests
                      ---
                      Use unittest, not pytest. ...

Design rules:

* Only the one-line descriptions sit in every prompt; a skill's full
  text costs context only when the model asks for it.
* The model names a skill; it never supplies a path. ``use_skill`` reads
  only files found by ``catalog()``.
* ``use_skill`` only reads, so the gate treats it as a read tool.
"""

from __future__ import annotations

import os

from .tools import Tool, tool

SKILLS_DIR = "skills"
SKILL_FILE = "SKILL.md"
MAX_SKILL_CHARS = 20_000


def catalog(workdir: str | os.PathLike[str]) -> dict[str, dict[str, str]]:
    """Return ``{name: {"description", "path"}}`` from skills/*/SKILL.md."""
    found: dict[str, dict[str, str]] = {}
    directory = os.path.join(workdir, SKILLS_DIR)
    if not os.path.isdir(directory):
        return found
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name, SKILL_FILE)
        if os.path.isfile(path):
            found[name] = {"description": _description(path) or name,
                           "path": path}
    return found


def catalog_prompt(workdir: str | os.PathLike[str]) -> str:
    """The catalog as a system-prompt section, or "" if there are none."""
    skills = catalog(workdir)
    if not skills:
        return ""
    lines = "\n".join(f"- {name}: {info['description']}"
                      for name, info in skills.items())
    return ("Skills available (load one with the use_skill tool when "
            f"relevant):\n{lines}")


def read_skill(workdir: str | os.PathLike[str], name: str) -> str:
    """Return a skill's full text, or an ``ERROR:`` naming the options."""
    skills = catalog(workdir)
    if name not in skills:
        available = ", ".join(skills) or "none"
        return f"ERROR: no skill named {name!r}. Available: {available}"
    with open(skills[name]["path"], encoding="utf-8",
              errors="replace") as f:
        text = f.read()
    if len(text) > MAX_SKILL_CHARS:
        text = text[:MAX_SKILL_CHARS] + "\n... (skill truncated)"
    return text


def skill_tools(workdir: str | os.PathLike[str]) -> list[Tool]:
    """Return the ``use_skill`` tool for the skills under ``workdir``."""
    project_dir = os.path.realpath(workdir)

    @tool("Load a skill's full instructions into the conversation. Use it "
          "when the task matches a skill in the catalog.",
          name="The skill's name, exactly as listed in the catalog")
    def use_skill(name: str) -> str:
        return read_skill(project_dir, name)

    return [use_skill]


def _description(path: str) -> str:
    """Read ``description:`` from the front matter at the top of SKILL.md.

    Front matter is the block between a first line of ``---`` and the
    next ``---``. Without one, only the lines before the first blank
    line are searched, so instructions further down are never mistaken
    for the description.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    if lines and lines[0].strip() == "---":
        header = []
        for line in lines[1:]:
            if line.strip() == "---":
                break
            header.append(line)
    else:
        header = []
        for line in lines:
            if not line.strip():
                break
            header.append(line)
    for line in header:
        key, _, value = line.partition(":")
        if key.strip().lower() == "description" and value.strip():
            return value.strip()
    return ""
