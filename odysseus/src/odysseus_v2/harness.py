"""The harness: every part, connected.

    from odysseus_v2 import create_provider, run_task

    answer = run_task(create_provider(), "path/to/project",
                      "Fix the failing test", mode="safe",
                      approver=console_approver)

One call wires the parts together:

    provider    the model, from any vendor            providers/
    tools       six core tools + remember + use_skill tools.py, memory.py,
                                                      skills.py
    gate        Policy(mode, approver).check          security.py
    system      rules + environment + AGENT_MEMORY.md memory.py, skills.py
                + skills catalog
    compactor   folds old turns when over budget      context.py
    session     appends every message to a JSONL log  session.py
    sub-agents  spawn_agent runs a fresh run_task     subagent.py
    loop        run_loop(...)                         loop.py

Pass ``messages`` to continue an earlier conversation: it is updated in
place, so the same list can be passed to the next ``run_task`` call.
Pass ``session`` (a path from ``open_session``) to log every message,
so a crashed run can be resumed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, MutableSequence
from typing import Any

from .context import DEFAULT_BUDGET_TOKENS, Compactor, estimate_overhead
from .loop import EventHandler, run_loop
from .memory import build_system_prompt, memory_tools
from .providers import Message, Provider
from .security import Approver, Mode, Policy, deny_all
from .session import append_message, session_recorder
from .skills import catalog_prompt, skill_tools
from .subagent import SUBAGENT_PROMPT, subagent_tool
from .tools import Tool, core_tools, tool_map


def _ignore_event(kind: str, payload: Mapping[str, Any]) -> None:
    return None


def harness_tools(workdir: str | os.PathLike[str]) -> dict[str, Tool]:
    """The full tool set: six core tools, ``remember`` and ``use_skill``."""
    return tool_map([*core_tools(workdir), *memory_tools(workdir),
                     *skill_tools(workdir)])


def run_task(
    provider: Provider,
    workdir: str | os.PathLike[str],
    task: str,
    *,
    mode: Mode = "safe",
    approver: Approver = deny_all,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    max_turns: int = 80,
    on_event: EventHandler = _ignore_event,
    messages: MutableSequence[Message] | None = None,
    extra_prompt: str = "",
    tools: Iterable[Tool] | None = None,
    session: str | os.PathLike[str] | None = None,
    subagent_depth: int = 0,
) -> str:
    """Run one task in ``workdir`` and return the model's final answer.

    The system prompt is rebuilt on every call, so notes saved with
    ``remember`` and skills added to ``skills/`` take effect on the next
    task. ``on_event`` receives the loop's events plus ``"compact"``.
    ``tools`` replaces the default set (``harness_tools``), for a task
    that should only have some of them.

    ``session`` is a log file: the task and every message after it are
    appended to it as they happen. ``subagent_depth`` > 0 adds the
    ``spawn_agent`` tool; each child gets one level less, the same
    tools, mode and approver, a fresh conversation and no session log.
    Its events reach ``on_event`` with ``"subagent"`` set to its level.
    """
    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"working directory not found: {workdir}")
    if messages is None:
        messages = []
    user = Message(role="user", text=task)
    messages.append(user)
    record = (session_recorder(session) if session is not None
              else _ignore_event)
    if session is not None:
        append_message(session, user)

    system = build_system_prompt(workdir, catalog_prompt(workdir),
                                 extra_prompt)
    base_tools = (list(tools) if tools is not None
                  else list(harness_tools(workdir).values()))
    tool_set = tool_map(base_tools)
    if subagent_depth > 0:
        def child_events(kind: str, payload: Mapping[str, Any]) -> None:
            on_event(kind, {**payload,
                            "subagent": payload.get("subagent", 0) + 1})

        def run_child(child_task: str) -> str:
            return run_task(
                provider, workdir, child_task, mode=mode, approver=approver,
                budget_tokens=budget_tokens, max_turns=max_turns,
                on_event=child_events, extra_prompt=SUBAGENT_PROMPT,
                tools=base_tools, subagent_depth=subagent_depth - 1)

        spawn = subagent_tool(run_child)
        tool_set[spawn.name] = spawn
    overhead = estimate_overhead(
        system, [t.spec["schema"] for t in tool_set.values()])
    compactor = Compactor(provider, budget_tokens, on_event=on_event,
                          overhead_tokens=overhead)
    policy = Policy(mode, approver, available=tool_set)

    def events(kind: str, payload: Mapping[str, Any]) -> None:
        record(kind, payload)                # session log, if any
        compactor.observe(kind, payload)     # real request sizes
        on_event(kind, payload)

    return run_loop(provider, system, messages, tool_set,
                    on_event=events, before_tool=policy.check,
                    max_turns=max_turns, before_turn=compactor.compact)
