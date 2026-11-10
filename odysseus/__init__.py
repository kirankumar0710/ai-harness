"""Odysseus - the smallest agentic harness, vendor-agnostic.

Day 1 files:

    providers/   the doorway to any model: one abstract Provider,
                 one subclass per vendor (Gemini, OpenAI, OpenRouter,
                 Anthropic)
    loop.py      the agentic loop

Later days add tools, security, context, memory, sessions, sub-agents
and orchestration on top of these two pieces.
"""

from .loop import run_loop
from .providers import Provider, ProviderConfig, create_provider

__version__ = "0.2.0"
__all__ = ["Provider", "ProviderConfig", "create_provider", "run_loop"]
