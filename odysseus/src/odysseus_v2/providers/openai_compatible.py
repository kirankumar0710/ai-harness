"""OpenAI-compatible providers (Chat Completions wire format).

One implementation serves every endpoint that speaks the OpenAI Chat
Completions format: OpenAI itself, OpenRouter (300+ models behind one
key), and self-hosted servers such as vLLM or Ollama. Only the config
differs: base URL, key variable and a few headers.

Vendor quirks handled here, and only here:

* The system prompt is the first message, with role ``system``.
* Tool results use a dedicated ``tool`` role and quote ``tool_call_id``.
* Tool arguments arrive as a JSON *string*, which may be malformed; a
  bad string becomes an error the model sees, never a crash.
* OpenAI's newer models take ``max_completion_tokens``; most compatible
  servers still take ``max_tokens``. The field name is configurable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ProviderError,
    ToolCall,
    ToolSchema,
    new_call_id,
)


@dataclass(frozen=True)
class OpenAIConfig(ProviderConfig):
    """OpenAI settings. There is no default model: choose one explicitly
    with ``model=`` or ``ODYSSEUS_MODEL``."""

    max_tokens_field: str = "max_completion_tokens"

    PROVIDER_NAME = "openai"   # defaults: [openai] in providers.ini


@dataclass(frozen=True)
class OpenRouterConfig(OpenAIConfig):
    """OpenRouter settings. Model ids are namespaced, e.g.
    ``google/gemini-3.1-pro-preview``."""

    max_tokens_field: str = "max_tokens"
    extra_headers: dict[str, str] = field(
        default_factory=lambda: {"X-Title": "Odysseus"})

    PROVIDER_NAME = "openrouter"   # defaults: [openrouter] in providers.ini


class OpenAICompatibleProvider(Provider):
    config_class = OpenAIConfig
    config: OpenAIConfig

    def _build_request(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        allow_tool_calls: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self.config.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_wire(system, messages),
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            body[self.config.max_tokens_field] = self.config.max_output_tokens
        if tools:
            body["tools"] = [{"type": "function", "function": dict(t)}
                             for t in tools]
            if not allow_tool_calls:
                body["tool_choice"] = "none"
        return url, headers, body

    @staticmethod
    def _to_wire(system: str,
                 messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Neutral messages -> OpenAI ``messages``."""
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})
        for message in messages:
            role = message["role"]
            if role == "user":
                wire.append({"role": "user",
                             "content": message.get("text", "")})
            elif role == "assistant":
                calls = message.get("tool_calls", [])
                entry: dict[str, Any] = {
                    "role": "assistant",
                    # content may be null only when tool calls are present
                    "content": message.get("text") or (None if calls else ""),
                }
                if calls:
                    entry["tool_calls"] = [
                        {"id": call["id"], "type": "function",
                         "function": {"name": call["name"],
                                      "arguments": json.dumps(call["args"])}}
                        for call in calls]
                wire.append(entry)
            elif role == "tool":
                wire.append({"role": "tool",
                             "tool_call_id": message["tool_call_id"],
                             "content": message.get("text", "")})
        return wire

    def _parse_response(self, raw: Mapping[str, Any]) -> Completion:
        choices = raw.get("choices") or []
        if not choices:
            raise ProviderError(f"No choices in reply: {raw.get('error')}")
        reply = choices[0].get("message", {})

        tool_calls = []
        for call in reply.get("tool_calls") or []:
            function = call.get("function", {})
            parsed = ToolCall(id=call.get("id") or new_call_id(),
                              name=function.get("name", ""), args={})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
                parsed["args"] = arguments
            except ValueError as error:
                parsed["error"] = f"invalid tool arguments: {error}"
            tool_calls.append(parsed)

        usage = raw.get("usage") or {}
        return Completion(
            text=reply.get("content") or "",
            tool_calls=tool_calls,
            usage={"input": usage.get("prompt_tokens", 0),
                   "output": usage.get("completion_tokens", 0)},
            truncated=choices[0].get("finish_reason") == "length",
        )


class OpenAIProvider(OpenAICompatibleProvider):
    config_class = OpenAIConfig


class OpenRouterProvider(OpenAICompatibleProvider):
    config_class = OpenRouterConfig
