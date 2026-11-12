"""Model providers: one abstract interface, one subclass per vendor.

    from odysseus.providers import create_provider

    provider = create_provider()                  # from the environment
    provider = create_provider("anthropic")       # pick a vendor
    provider = create_provider("openai", model="gpt-...",
                               base_url="http://localhost:8000/v1")

The vendor comes from the ``name`` argument, else ``ODYSSEUS_PROVIDER``;
if neither is set, ``ConfigError`` is raised (there is no default
vendor). Adding a vendor means one new module and one line in
``PROVIDERS``; nothing else in the harness changes.
"""

from __future__ import annotations

import os
from typing import Any

from .anthropic import AnthropicConfig, AnthropicProvider
from .base import (
    Completion,
    ConfigError,
    Message,
    Provider,
    ProviderConfig,
    ProviderError,
    ToolCall,
    ToolSchema,
)
from .gemini import GeminiConfig, GeminiProvider
from .openai_compatible import (
    OpenAIConfig,
    OpenAIProvider,
    OpenRouterConfig,
    OpenRouterProvider,
)

PROVIDERS: dict[str, type[Provider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "openrouter": OpenRouterProvider,
    "anthropic": AnthropicProvider,
}


def create_provider(name: str | None = None, **overrides: Any) -> Provider:
    """Build a provider from its name, the environment and overrides.

    ``overrides`` are any ``ProviderConfig`` fields (``api_key``,
    ``model``, ``base_url``, ``temperature``, ...).
    """
    known = ", ".join(sorted(PROVIDERS))
    name = (name or os.environ.get("ODYSSEUS_PROVIDER") or "").strip().lower()
    if not name:
        raise ConfigError(
            "No provider selected: pass a name to create_provider() or set "
            f"ODYSSEUS_PROVIDER to one of: {known}")
    try:
        provider_class = PROVIDERS[name]
    except KeyError:
        raise ConfigError(
            f"Unknown provider {name!r}; choose one of: {known}") from None
    config = provider_class.config_class.from_env(**overrides)
    return provider_class(config)


__all__ = [
    "PROVIDERS",
    "AnthropicConfig",
    "AnthropicProvider",
    "Completion",
    "ConfigError",
    "GeminiConfig",
    "GeminiProvider",
    "Message",
    "OpenAIConfig",
    "OpenAIProvider",
    "OpenRouterConfig",
    "OpenRouterProvider",
    "Provider",
    "ProviderConfig",
    "ProviderError",
    "ToolCall",
    "ToolSchema",
    "create_provider",
]
