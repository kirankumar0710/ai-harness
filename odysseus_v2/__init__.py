"""Odysseus - the smallest agentic harness, vendor-agnostic.

    providers/   the doorway to any model: one abstract Provider,
                 one subclass per vendor (Gemini, OpenAI, OpenRouter,
                 Anthropic)
    loop.py      the agentic loop
"""

from .loop import run_loop
from .providers import Provider, ProviderConfig, create_provider

__version__ = "0.2.0"
__all__ = ["Provider", "ProviderConfig", "create_provider", "run_loop"]
