"""Context: the budget and the compactor.

A model's context window is finite; a real task is not. When the
conversation outgrows the budget, the oldest turns are folded into a
summary written by the model itself, and the work carries on. This one
trick is what lets a harness work for hours.

    compactor = Compactor(provider, budget_tokens=60_000)
    run_loop(..., before_turn=compactor.compact)

Design rules:

* The compactor talks to any ``Provider``; it never knows the vendor.
* The newest turns are never summarised: the model keeps working on
  exact text, not a paraphrase of it.
* The cut never separates a tool call from its result. Vendors reject a
  result whose call is missing, so the cut moves back to the call.
* The history stays valid after compaction: it starts with a user
  message, and two user messages are never left side by side.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .providers import Message, Provider

CHARS_PER_TOKEN = 4             # a rough but honest estimate
KEEP_RECENT = 6                 # never summarise the newest messages
DEFAULT_BUDGET_TOKENS = 60_000
MAX_CHARS_PER_MESSAGE = 2_000   # per message, in the transcript to summarise

SUMMARY_HEADER = "[Conversation so far, compacted]"

SUMMARY_PROMPT = (
    "You compress agent transcripts. Preserve: the original task, every "
    "file created or edited and its purpose, key decisions, unresolved "
    "errors, and what remains to be done. Be dense and factual."
)

EventHandler = Callable[[str, Mapping[str, Any]], None]


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Estimate the size of a conversation in tokens.

    Counts every character of every message, keys included, so it
    overestimates slightly: the safe direction for a budget.
    """
    return sum(len(str(m)) for m in messages) // CHARS_PER_TOKEN


class Compactor:
    """Fold old turns into a summary when the conversation is too big.

    ``compact`` has the shape of the loop's ``TurnHook``: it takes the
    history and returns the history to send. ``on_event`` receives
    ``("compact", {"before", "after", "summarised"})`` each time it
    folds, so a caller can show or log it.
    """

    def __init__(
        self,
        provider: Provider,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        keep_recent: int = KEEP_RECENT,
        on_event: EventHandler | None = None,
    ) -> None:
        if budget_tokens <= 0 or keep_recent < 1:
            raise ValueError("budget_tokens and keep_recent must be positive")
        self.provider = provider
        self.budget_tokens = budget_tokens
        self.keep_recent = keep_recent
        self.on_event = on_event
        self.compactions = 0

    def compact(self, messages: list[Message]) -> list[Message]:
        """Return ``messages``, or a shorter equivalent if over budget."""
        before = estimate_tokens(messages)
        if before <= self.budget_tokens:
            return messages

        cut = _safe_cut(messages, len(messages) - self.keep_recent)
        if cut < 2:          # too little old history to be worth folding
            return messages
        old, recent = messages[:cut], messages[cut:]

        summary = self.provider.complete(
            SUMMARY_PROMPT,
            [Message(role="user", text="Summarise this transcript:\n\n"
                                       + transcript(old))],
        )["text"].strip()
        compacted = _prepend_summary(f"{SUMMARY_HEADER}\n{summary}", recent)

        self.compactions += 1
        if self.on_event is not None:
            self.on_event("compact", {"before": before,
                                      "after": estimate_tokens(compacted),
                                      "summarised": len(old)})
        return compacted


def transcript(messages: Sequence[Message]) -> str:
    """Render messages as plain text for the summariser."""
    blocks = []
    for message in messages:
        role = message.get("role", "?")
        if role == "tool":
            status = " error" if message.get("is_error") else ""
            role = f"tool {message.get('name', '')}{status}".rstrip()
        text = _shorten(message.get("text") or "")
        calls = "".join(f"\n  called {c['name']}({c['args']})"
                        for c in message.get("tool_calls", []))
        blocks.append(f"[{role}] {text}{calls}")
    return "\n\n".join(blocks)


def _shorten(text: str) -> str:
    """Cut a long message to its start and end, marking what was cut.

    Long tool results usually end with what matters most - a test
    summary, an error, a final value - so keeping only the start would
    hide it from the summariser.
    """
    if len(text) <= MAX_CHARS_PER_MESSAGE:
        return text
    half = MAX_CHARS_PER_MESSAGE // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n... ({cut} chars cut) ...\n{text[-half:]}"


def _safe_cut(messages: Sequence[Message], cut: int) -> int:
    """Move ``cut`` back so the recent part starts on a call, not a result.

    If the first kept message were a tool result, its call would be in
    the summarised part and the vendor would reject the history.
    """
    while cut > 0 and messages[cut].get("role") == "tool":
        cut -= 1
    return max(cut, 0)


def _prepend_summary(summary: str,
                     recent: list[Message]) -> list[Message]:
    """Put the summary first, as a user message.

    If the recent part already starts with a user message, the summary
    is merged into it, so two user messages never sit side by side.
    """
    if recent and recent[0].get("role") == "user":
        first = dict(recent[0])
        first["text"] = f"{summary}\n\n{first.get('text', '')}"
        return [Message(**first), *recent[1:]]
    return [Message(role="user", text=summary), *recent]
