"""Odysseus - the smallest agentic harness, vendor-agnostic.

    providers/   the doorway to any model: one abstract Provider,
                 one subclass per vendor (Gemini, OpenAI, OpenRouter,
                 Anthropic)
    loop.py      the agentic loop
    tools.py     the @tool decorator and the six core tools
    security.py  the permission gate (deny rules, modes, approver)
"""

from .loop import run_loop
from .providers import Provider, ProviderConfig, create_provider
from .security import Policy
from .tools import Tool, core_tools, tool, tool_map

__version__ = "0.2.0"
__all__ = [
    "Policy",
    "Provider",
    "ProviderConfig",
    "Tool",
    "core_tools",
    "create_provider",
    "run_loop",
    "tool",
    "tool_map",
]
