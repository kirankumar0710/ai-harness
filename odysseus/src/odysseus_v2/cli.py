"""The doorway: an interactive prompt and a headless mode.

    python3 -m odysseus_v2                      # interactive, safe mode
    python3 -m odysseus_v2 -p "fix the tests"   # headless: one task, exit
    python3 -m odysseus_v2 --resume             # continue the last session
    odysseus ...                                # the same, once installed

Interactive mode asks before every write or command (``safe`` mode with
``console_approver``). Headless mode has nobody to ask: it defaults to
``safe`` with ``deny_all``, so writes and commands are refused with a
reason, unless you choose ``--mode yolo``. The original defaulted
headless runs to yolo; running commands unattended should be a choice.

Every task is logged to a session in ``.odysseus/sessions/``. Ctrl-C
during a task stops it; the conversation is then reloaded from the log,
repaired, so the next task continues from a valid history.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .harness import run_task
from .providers import ConfigError, create_provider
from .security import MODES, console_approver, deny_all
from .session import new_session, open_session, resume_session


class Style:
    """ANSI styles, off when output is not a terminal or NO_COLOR is set."""

    def __init__(self, enabled: bool) -> None:
        on = enabled
        self.tool = "\033[33m" if on else ""
        self.dim = "\033[2m" if on else ""
        self.bold = "\033[1m" if on else ""
        self.reset = "\033[0m" if on else ""


STYLE = Style(sys.stdout.isatty() and "NO_COLOR" not in os.environ)


def print_event(kind: str, payload: Mapping[str, Any]) -> None:
    s = STYLE
    pad = "    " * payload.get("subagent", 0)
    if kind == "assistant" and payload.get("text"):
        print(f"\n{pad}{payload['text']}")
    elif kind == "tool_start":
        args = ", ".join(f"{k}={str(v)[:80]!r}"
                         for k, v in payload["args"].items())
        print(f"{pad}{s.tool}* {payload['name']}{s.reset}{s.dim}({args})"
              f"{s.reset}")
    elif kind == "tool_end":
        result = str(payload.get("result", ""))
        first = result.splitlines()[0][:120] if result.strip() else ""
        print(f"{pad}{s.dim}  -> {first}{s.reset}")
    elif kind == "compact":
        print(f"{s.dim}(compacted {payload['summarised']} messages){s.reset}")
    elif kind == "truncated":
        print(f"{s.dim}(the reply hit the output-token limit){s.reset}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odysseus",
        description="Odysseus - a small agent harness for any model vendor.")
    parser.add_argument("-p", "--prompt",
                        help="run one task headless and exit")
    parser.add_argument("-d", "--workdir", default=".",
                        help="working directory, the jail (default: .)")
    parser.add_argument("--provider", help="vendor (default: "
                        "$ODYSSEUS_PROVIDER)")
    parser.add_argument("-m", "--model", help="model (default: the "
                        "provider's)")
    parser.add_argument("--mode", choices=MODES, default="safe",
                        help="permission mode (default: safe)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the newest session in the workdir")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--budget", type=int, default=None,
                        help="compaction budget in tokens")
    parser.add_argument("--subagents", type=int, default=0, metavar="N",
                        help="offer spawn_agent, N levels deep "
                             "(default: 0, off)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = STYLE
    if not os.path.isdir(args.workdir):
        print(f"error: no such folder: {args.workdir}", file=sys.stderr)
        return 2
    try:
        overrides = {"model": args.model} if args.model else {}
        provider = create_provider(args.provider, **overrides)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    headless = args.prompt is not None
    approver = deny_all if headless else console_approver
    options: dict[str, Any] = {
        "mode": args.mode, "approver": approver,
        "max_turns": args.max_turns, "on_event": print_event,
        "subagent_depth": args.subagents}
    if args.budget:
        options["budget_tokens"] = args.budget

    session: str | None = None
    messages: list = []
    if args.resume:
        session, messages = open_session(args.workdir, resume=True)
        note = (f"resumed {len(messages)} messages from {session}"
                if messages else "no session to resume; starting a new one")
        print(f"{s.dim}{note}{s.reset}")

    def run(task: str) -> None:
        nonlocal session, messages
        if session is None:
            session = new_session(args.workdir, task[:32])
        try:
            run_task(provider, args.workdir, task, messages=messages,
                     session=session, **options)
        except KeyboardInterrupt:
            messages = resume_session(session)
            print(f"\n{s.dim}interrupted: the session log is safe; "
                  f"carry on, or run again with --resume{s.reset}")
            if headless:
                raise

    if headless:
        if args.mode == "safe":
            print(f"{s.dim}headless, safe mode: writes and commands are "
                  f"refused (use --mode yolo to allow them){s.reset}")
        try:
            run(args.prompt)
        except KeyboardInterrupt:
            return 130
        return 0

    print(f"{s.bold}{s.tool}ODYSSEUS{s.reset} - {provider.model} - mode "
          f"{args.mode} - jail {os.path.realpath(args.workdir)}\n"
          f"{s.dim}Ctrl-D to leave.{s.reset}")
    while True:
        try:
            task = input(f"\n{s.bold}> {s.reset}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if task:
            run(task)


if __name__ == "__main__":
    sys.exit(main())
