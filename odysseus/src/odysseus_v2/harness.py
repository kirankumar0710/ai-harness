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
    loop        run_loop(...)                         loop.py

Pass ``messages`` to continue an earlier conversation: it is updated in
place, so the same list can be passed to the next ``run_task`` call.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableSequence
from typing import Any

from .context import DEFAULT_BUDGET_TOKENS, Compactor
from .loop import EventHandler, run_loop
from .memory import build_system_prompt, memory_tools
from .providers import Message, Provider
from .security import Approver, Mode, Policy, deny_all
from .skills import catalog_prompt, skill_tools
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
) -> str:
    """Run one task in ``workdir`` and return the model's final answer.

    The system prompt is rebuilt on every call, so notes saved with
    ``remember`` and skills added to ``skills/`` take effect on the next
    task. ``on_event`` receives the loop's events plus ``"compact"``.
    """
    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"working directory not found: {workdir}")
    if messages is None:
        messages = []
    messages.append(Message(role="user", text=task))

    system = build_system_prompt(workdir, catalog_prompt(workdir),
                                 extra_prompt)
    compactor = Compactor(provider, budget_tokens, on_event=on_event)
    policy = Policy(mode, approver)
    return run_loop(provider, system, messages, harness_tools(workdir),
                    on_event=on_event, before_tool=policy.check,
                    max_turns=max_turns, before_turn=compactor.compact)
