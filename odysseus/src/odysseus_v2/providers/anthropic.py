"""Anthropic provider (Claude Messages API).

Vendor quirks handled here, and only here:

* The system prompt is a top-level ``system`` field.
* Messages hold lists of content blocks; tool requests are ``tool_use``
  blocks and results are ``tool_result`` blocks in the next ``user``
  message. All results for one turn must share that single message.
* ``max_tokens`` is mandatory, so the config gives it a default.
* A request whose history contains tool blocks must still define the
  tools; to forbid further calls we send ``tool_choice: none``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    ToolCall,
    ToolSchema,
)


@dataclass(frozen=True)
class AnthropicConfig(ProviderConfig):
    max_output_tokens: int | None = 8192
    anthropic_version: str = "2023-06-01"

    PROVIDER_NAME = "anthropic"   # defaults: [anthropic] in providers.ini


class AnthropicProvider(Provider):
    config_class = AnthropicConfig
    config: AnthropicConfig

    def _build_request(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        allow_tool_calls: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self.config.base_url}/messages"
        headers = {"x-api-key": self.config.api_key,
                   "anthropic-version": self.config.anthropic_version}
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.config.max_output_tokens or 8192,
            "messages": self._to_wire(messages),
        }
        if system:
            body["system"] = system
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if tools:
            body["tools"] = [
                {"name": t["name"],
                 "description": t.get("description", ""),
                 "input_schema": t.get("parameters",
                                       {"type": "object", "properties": {}})}
                for t in tools]
            if not allow_tool_calls:
                body["tool_choice"] = {"type": "none"}
        return url, headers, body

    @staticmethod
    def _to_wire(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Neutral messages -> Anthropic ``messages``."""
        wire: list[dict[str, Any]] = []

        def add(role: str, blocks: list[dict[str, Any]]) -> None:
            # Consecutive same-role messages merge: every tool_result for
            # one turn lands in a single user message, results first.
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"].extend(blocks)
            else:
                wire.append({"role": role, "content": blocks})

        for message in messages:
            role = message["role"]
            if role == "user":
                add("user", [{"type": "text",
                              "text": message.get("text") or "(empty)"}])
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                if message.get("text"):
                    blocks.append({"type": "text", "text": message["text"]})
                for call in message.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": call["id"],
                                   "name": call["name"],
                                   "input": call["args"]})
                # The API rejects empty text blocks.
                add("assistant",
                    blocks or [{"type": "text", "text": "(no content)"}])
            elif role == "tool":
                add("user", [{"type": "tool_result",
                              "tool_use_id": message["tool_call_id"],
                              "content": message.get("text", ""),
                              "is_error": bool(message.get("is_error"))}])
        return wire

    def _parse_response(self, raw: Mapping[str, Any]) -> Completion:
        text, tool_calls = "", []
        for block in raw.get("content") or []:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"], name=block["name"],
                    args=dict(block.get("input") or {})))
            # "thinking" blocks are internal reasoning: skipped
        usage = raw.get("usage") or {}
        return Completion(
            text=text,
            tool_calls=tool_calls,
            usage={"input": usage.get("input_tokens", 0),
                   "output": usage.get("output_tokens", 0)},
            truncated=raw.get("stop_reason") == "max_tokens",
        )
