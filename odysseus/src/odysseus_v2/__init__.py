"""Odysseus - the smallest agentic harness, vendor-agnostic.

    providers/   the doorway to any model: one abstract Provider,
                 one subclass per vendor (Gemini, OpenAI, OpenRouter,
                 Anthropic)
    loop.py      the agentic loop
    tools.py     the @tool decorator and the six core tools
    security.py  the permission gate (deny rules, modes, approver)
    context.py   the token budget and the compactor
    memory.py    the system prompt and project memory (AGENT_MEMORY.md)
    skills.py    the skills catalog and the use_skill tool
    session.py   the append-only session log: crash, then resume
    subagent.py  spawn_agent: a fresh agent for a self-contained task
    harness.py   run_task(): every part above, connected
    fleet.py     run_fleet(): many run_task calls in parallel
    cli.py       the command line: python3 -m odysseus_v2
"""

from .context import Compactor
from .fleet import Job, JobResult, run_fleet
from .harness import harness_tools, run_task
from .loop import run_loop
from .memory import build_system_prompt, memory_tools
from .providers import Provider, ProviderConfig, create_provider
from .security import Policy
from .session import (SessionCorrupt, latest_session, load_session,
                      new_session, open_session, resume_session)
from .skills import catalog, skill_tools
from .subagent import subagent_tool
from .tools import Tool, core_tools, tool, tool_map

__version__ = "0.4.0"
__all__ = [
    "Compactor",
    "Job",
    "JobResult",
    "Policy",
    "Provider",
    "ProviderConfig",
    "SessionCorrupt",
    "Tool",
    "build_system_prompt",
    "catalog",
    "core_tools",
    "create_provider",
    "harness_tools",
    "latest_session",
    "load_session",
    "memory_tools",
    "new_session",
    "open_session",
    "resume_session",
    "run_fleet",
    "run_loop",
    "run_task",
    "skill_tools",
    "subagent_tool",
    "tool",
    "tool_map",
]
