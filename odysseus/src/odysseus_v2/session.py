"""Durability: the session log.

Every message is appended to a JSONL file the moment it exists. If the
process dies (a crash, Ctrl-C, a power cut), ``load_session()`` replays
the log and the agent continues where it stopped. Append-only is the
whole trick: a write can only be torn at the end, never in the middle.

    path, messages = open_session(workdir, "fix tests")
    run_task(provider, workdir, task, messages=messages, session=path)
    # ... the process dies; later:
    path, messages = open_session(workdir, "fix tests", resume=True)
    run_task(provider, workdir, "Continue.", messages=messages, session=path)

Design rules:

* The log holds the raw conversation: the task, every reply, every tool
  result. Compaction is not logged; a resumed session is folded again
  on its first turn if it is over budget.
* A torn last line is dropped. A bad line anywhere else is corruption,
  and loading refuses rather than silently losing the rest.
* Resuming mends the file before anything is appended: the torn tail is
  cut, and the repair below is written to the log, so every later
  resume replays the same, valid history.
* A call whose result never reached the log is answered with an error
  saying it may or may not have run: the side effect (an edit, a
  command) may have happened even though its result was lost.
* Logs live in ``.odysseus/sessions/`` inside the project. The file
  tools refuse ``.odysseus/``, so the agent cannot rewrite the history
  it will be replayed from. ``bash`` is not jailed and still can.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from .providers import Message
from .tools import HARNESS_DIR

SESSION_DIR = os.path.join(HARNESS_DIR, "sessions")
SESSION_SUFFIX = ".jsonl"

INTERRUPTED = ("ERROR: interrupted - the process stopped before this "
               "call's result was recorded. It may or may not have run; "
               "check its effect before repeating it.")


class SessionCorrupt(ValueError):
    """A line before the last one in a session log is not valid JSON."""


def session_dir(workdir: str | os.PathLike[str]) -> str:
    return os.path.join(workdir, SESSION_DIR)


def new_session(workdir: str | os.PathLike[str],
                label: str = "session") -> str:
    """Create an empty session log and return its path.

    Names sort by time (``20261116-093012-fix-tests.jsonl``); a second
    session in the same second gets ``-2``, ``-3`` instead of sharing a
    file.
    """
    directory = session_dir(workdir)
    os.makedirs(directory, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "-"
                   for c in label)[:40] or "session"
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}"
    for attempt in range(1, 1000):
        suffix = "" if attempt == 1 else f"-{attempt}"
        path = os.path.join(directory, stem + suffix + SESSION_SUFFIX)
        try:
            with open(path, "x", encoding="utf-8"):
                return path
        except FileExistsError:
            continue
    raise FileExistsError(f"too many sessions named {stem!r}")


def append_message(path: str | os.PathLike[str], message: Message) -> None:
    """Append one message as one line, flushed to disk before returning."""
    line = json.dumps(message, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load_session(path: str | os.PathLike[str]) -> list[Message]:
    """Replay a session log, repaired so the model can continue it.

    Read-only: the file is not changed. To continue the session and
    keep logging to it, use ``open_session(..., resume=True)``, which
    also writes the repair into the log.
    """
    messages, _ = _read(path)
    return repair(messages)


def _read(path: str | os.PathLike[str]) -> tuple[list[Message], int]:
    """Return the messages and the byte length of the good part.

    A bad last line is a torn write and is left out of the good part; a
    bad line before it is corruption.
    """
    with open(path, "rb") as f:
        data = f.read()
    chunks = data.split(b"\n")
    messages: list[Message] = []
    offset = 0
    for index, chunk in enumerate(chunks):
        if chunk.strip():
            try:
                messages.append(json.loads(chunk.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if any(rest.strip() for rest in chunks[index + 1:]):
                    raise SessionCorrupt(
                        f"{path}: line {index + 1} is not valid JSON"
                    ) from None
                return messages, offset          # torn final write
        offset += len(chunk) + 1
    return messages, len(data)


def resume_session(path: str | os.PathLike[str]) -> list[Message]:
    """Load ``path`` and mend the file so appending can continue.

    A torn tail is cut off (otherwise the next line would be glued to
    it), a missing final newline is added, and the results ``repair``
    adds are written to the log, so the next resume sees them too.
    """
    messages, good = _read(path)
    with open(path, "rb+") as f:
        f.truncate(good)
        if good:
            f.seek(good - 1)
            if f.read(1) != b"\n":
                f.write(b"\n")
    logged = len(messages)
    messages = repair(messages)
    for message in messages[logged:]:
        append_message(path, message)
    return messages


def latest_session(workdir: str | os.PathLike[str]) -> str | None:
    """The newest session log in ``workdir``, or ``None``."""
    directory = session_dir(workdir)
    if not os.path.isdir(directory):
        return None
    names = sorted(name for name in os.listdir(directory)
                   if name.endswith(SESSION_SUFFIX))
    return os.path.join(directory, names[-1]) if names else None


def open_session(workdir: str | os.PathLike[str], label: str = "session",
                 *, resume: bool = False) -> tuple[str, list[Message]]:
    """Return ``(path, messages)``: the latest session when ``resume``
    and one exists, otherwise a new, empty one."""
    if resume:
        path = latest_session(workdir)
        if path is not None:
            return path, resume_session(path)
    return new_session(workdir, label), []


def repair(messages: list[Message]) -> list[Message]:
    """Answer every call of the last reply that has no result yet.

    Every provider rejects a history with an unanswered call, so a log
    cut off between a call and its result could not be continued.
    Results are matched to calls by id, not counted.
    """
    last = next((i for i in range(len(messages) - 1, -1, -1)
                 if messages[i].get("role") == "assistant"), None)
    if last is None:
        return messages
    answered = {m.get("tool_call_id") for m in messages[last + 1:]
                if m.get("role") == "tool"}
    for call in messages[last].get("tool_calls") or []:
        if call["id"] not in answered:
            messages.append(Message(
                role="tool", tool_call_id=call["id"], name=call["name"],
                text=INTERRUPTED, is_error=True))
    return messages


def session_recorder(
        path: str | os.PathLike[str],
) -> Callable[[str, Mapping[str, Any]], None]:
    """An event handler that appends each ``"message"`` event to ``path``."""
    def record(kind: str, payload: Mapping[str, Any]) -> None:
        if kind == "message":
            append_message(path, dict(payload))
    return record
