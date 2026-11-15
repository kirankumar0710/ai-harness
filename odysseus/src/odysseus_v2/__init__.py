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
    harness.py   run_task(): every part above, connected
"""

from .context import Compactor
from .harness import harness_tools, run_task
from .loop import run_loop
from .memory import build_system_prompt, memory_tools
from .providers import Provider, ProviderConfig, create_provider
from .security import Policy
from .skills import catalog, skill_tools
from .tools import Tool, core_tools, tool, tool_map

__version__ = "0.2.0"
__all__ = [
    "Compactor",
    "Policy",
    "Provider",
    "ProviderConfig",
    "Tool",
    "build_system_prompt",
    "catalog",
    "core_tools",
    "create_provider",
    "harness_tools",
    "memory_tools",
    "run_loop",
    "run_task",
    "skill_tools",
    "tool",
    "tool_map",
]
