"""Google Gemini provider (Generative Language REST API).

Vendor quirks handled here, and only here:

* The system prompt goes in ``systemInstruction``, not in the messages.
* The model's role is called ``model``; tool results travel in the
  ``user`` role as ``functionResponse`` parts.
* Gemini 3 signs its reasoning. The ``thoughtSignature`` returned with a
  function call must be echoed back on that call in later requests, or
  the API answers HTTP 400.
* Parallel function responses are grouped into one ``user`` content.
* The API key is sent in the ``x-goog-api-key`` header, never in the URL,
  so it cannot leak into logs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .base import (
    GENERATED_ID_PREFIX,
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
class GeminiConfig(ProviderConfig):
    """Gemini settings. Leave ``temperature`` unset: Google recommends
    the default of 1.0 for Gemini 3, and lower values can cause looping.
    """

    PROVIDER_NAME = "gemini"   # defaults: [gemini] in providers.ini


class GeminiProvider(Provider):
    config_class = GeminiConfig

    def _build_request(
        self,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        allow_tool_calls: bool,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        url = f"{self.config.base_url}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.config.api_key}
        body: dict[str, Any] = {"contents": self._to_wire(messages)}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        generation: dict[str, Any] = {}
        if self.config.temperature is not None:
            generation["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            generation["maxOutputTokens"] = self.config.max_output_tokens
        if generation:
            body["generationConfig"] = generation

        if tools:
            declarations = [dict(t) for t in tools]
            body["tools"] = [{"functionDeclarations": declarations}]
            if not allow_tool_calls:
                body["toolConfig"] = {
                    "functionCallingConfig": {"mode": "NONE"}}
        return url, headers, body

    @staticmethod
    def _to_wire(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Neutral messages -> Gemini ``contents``."""
        contents: list[dict[str, Any]] = []

        def add(role: str, parts: list[dict[str, Any]]) -> None:
            # Consecutive same-role entries merge, which groups parallel
            # function responses into one turn as Gemini expects.
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})

        for message in messages:
            role = message["role"]
            if role == "user":
                add("user", [{"text": message.get("text", "")}])
            elif role == "assistant":
                parts: list[dict[str, Any]] = []
                if message.get("text"):
                    parts.append({"text": message["text"]})
                for call in message.get("tool_calls", []):
                    function_call: dict[str, Any] = {
                        "name": call["name"], "args": call["args"]}
                    if not call["id"].startswith(GENERATED_ID_PREFIX):
                        function_call["id"] = call["id"]
                    part: dict[str, Any] = {"functionCall": function_call}
                    if call.get("signature"):
                        part["thoughtSignature"] = call["signature"]
                    parts.append(part)
                add("model", parts or [{"text": ""}])
            elif role == "tool":
                response: dict[str, Any] = {
                    "name": message["name"],
                    "response": {"result": message.get("text", "")},
                }
                call_id = message.get("tool_call_id", "")
                if call_id and not call_id.startswith(GENERATED_ID_PREFIX):
                    response["id"] = call_id
                add("user", [{"functionResponse": response}])
        return contents

    def _parse_response(self, raw: Mapping[str, Any]) -> Completion:
        candidates = raw.get("candidates") or []
        if not candidates:
            raise ProviderError(
                f"Gemini returned no candidates: {raw.get('promptFeedback')}")

        text, tool_calls = "", []
        for part in candidates[0].get("content", {}).get("parts", []):
            if part.get("thought"):
                continue  # internal reasoning summary, not the answer
            if "text" in part:
                text += part["text"]
            if "functionCall" in part:
                call = part["functionCall"]
                tool_calls.append(ToolCall(
                    id=call.get("id") or new_call_id(),
                    name=call["name"],
                    args=dict(call.get("args") or {}),
                    signature=part.get("thoughtSignature"),
                ))

        usage = raw.get("usageMetadata", {})
        return Completion(
            text=text,
            tool_calls=tool_calls,
            usage={"input": usage.get("promptTokenCount", 0),
                   "output": usage.get("candidatesTokenCount", 0)},
        )
