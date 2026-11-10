"""Day 1 - The agentic loop.

The heart of every harness is small:

    while turns remain:
        reply = model(conversation)          # call
        append reply                         # remember what it said
        if reply has no tool calls: return   # the model owns the ending
        run each tool, append each result    # act, then remember
    wrap-up call with tools disabled         # never end without a report

The loop knows nothing about vendors: it talks to any ``Provider``.

Design rules:

* Every tool result quotes the id of the call it answers, and every call
  is answered exactly once, even when it is blocked or fails.
* A tool can never crash the loop: errors become results the model reads.
* The model decides when the work is done; ``max_turns`` is the hard
  stop the model cannot override.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence
from typing import Any, Protocol

from .providers import Completion, Message, Provider, ToolCall

EventHandler = Callable[[str, Mapping[str, Any]], None]
ToolGate = Callable[[ToolCall], "str | None"]   # block reason, or None
TurnHook = Callable[[list[Message]], list[Message]]

WRAP_UP_PROMPT = (
    "Turn limit reached. Do not call any more tools. Summarise what was "
    "done, what remains, and what someone resuming this work needs to know."
)


class Tool(Protocol):
    """What the loop needs from a tool. Day 2 provides real ones."""

    name: str
    spec: Mapping[str, Any]          # {"schema": {name, description, ...}}

    def run(self, **kwargs: Any) -> Any: ...


def _ignore_event(kind: str, payload: Mapping[str, Any]) -> None:
    return None


def _allow_all(call: ToolCall) -> str | None:
    return None


def run_loop(
    provider: Provider,
    system: str,
    messages: MutableSequence[Message],
    tools: Mapping[str, Tool],
    on_event: EventHandler = _ignore_event,
    before_tool: ToolGate = _allow_all,
    max_turns: int = 80,
    before_turn: TurnHook | None = None,
) -> str:
    """Run turns until the model stops calling tools; return its answer.

    ``messages`` is updated in place: it is the agent's entire memory.

    Events: ``"assistant"`` after each model reply (payload: the
    ``Completion``), ``"tool_start"`` and ``"tool_end"`` around each tool.
    ``before_tool`` is the permission gate (Day 2); ``before_turn`` may
    rewrite the history before each call (Day 3 compaction).
    """
    schemas = [tool.spec["schema"] for tool in tools.values()]

    for _turn in range(max_turns):
        if before_turn is not None:
            messages[:] = before_turn(list(messages))

        reply = provider.complete(system, messages, schemas)
        _append_reply(messages, reply)
        on_event("assistant", reply)

        if not reply["tool_calls"]:
            return reply["text"]            # no request: the model is done

        for call in reply["tool_calls"]:    # answer every call, in order
            on_event("tool_start", call)
            text, is_error = _execute(call, tools, before_tool)
            messages.append(Message(
                role="tool", tool_call_id=call["id"], name=call["name"],
                text=text, is_error=is_error))
            on_event("tool_end", {**call, "result": text,
                                  "is_error": is_error})

    # Out of turns: ask for a status report instead of stopping silently.
    messages.append(Message(role="user", text=WRAP_UP_PROMPT))
    reply = provider.complete(system, messages, schemas,
                              allow_tool_calls=False)
    # Drop any tool calls the model made anyway: they would never be
    # answered, which would leave the history invalid for the next call.
    _append_reply(messages, Completion(text=reply["text"], tool_calls=[],
                                       usage=reply["usage"]))
    on_event("assistant", reply)
    return reply["text"]


def _append_reply(messages: MutableSequence[Message],
                  reply: Completion) -> None:
    messages.append(Message(role="assistant", text=reply["text"],
                            tool_calls=list(reply["tool_calls"])))


def _execute(call: ToolCall, tools: Mapping[str, Tool],
             before_tool: ToolGate) -> tuple[str, bool]:
    """Run one tool call. Returns (result_text, is_error); never raises."""
    if call.get("error"):
        return f"ERROR: {call['error']}", True
    tool = tools.get(call["name"])
    if tool is None:
        known = ", ".join(sorted(tools)) or "none"
        return (f"ERROR: unknown tool {call['name']!r} "
                f"(available: {known})", True)
    blocked = before_tool(call)
    if blocked:
        return f"BLOCKED: {blocked}", True
    try:
        return str(tool.run(**call["args"])), False
    except Exception as error:   # tools are an isolation boundary
        return f"ERROR: {type(error).__name__}: {error}", True
