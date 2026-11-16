"""Context: the budget and the compactor.

A model's context window is finite; a real task is not. When the
conversation outgrows the budget, the oldest turns are folded into a
summary written by the model itself, and the work carries on. This one
trick is what lets a harness work for hours.

    overhead = estimate_overhead(system, schemas)
    compactor = Compactor(provider, budget_tokens=60_000,
                          overhead_tokens=overhead)
    run_loop(..., before_turn=compactor.compact,
             on_event=compactor.observe)

Design rules:

* The compactor talks to any ``Provider``; it never knows the vendor.
* The newest turns are never summarised: the model keeps working on
  exact text, not a paraphrase of it. Only as many are kept as fit in
  half the budget, so the budget is a cap, and several steps fit
  between one fold and the next.
* Sizes come from the vendor when it reports them; four characters per
  token is only the fallback, corrected by how far off it was.
* The cut never separates a tool call from its result. Vendors reject a
  result whose call is missing, so the cut moves back to the call.
* The history stays valid after compaction: it starts with a user
  message, and two user messages are never left side by side.
* The budget covers the whole request. The system prompt and tool
  schemas go out with every call, so they are counted as a fixed
  overhead on top of the messages.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .providers import Message, Provider

CHARS_PER_TOKEN = 4             # a rough but honest estimate
KEEP_RECENT = 6                 # never summarise the newest messages
DEFAULT_BUDGET_TOKENS = 60_000
MAX_CHARS_PER_MESSAGE = 2_000   # per message, in the transcript to summarise

FOLD_TO = 0.5                  # after a fold, keep at most half the budget
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


def estimate_overhead(system: str,
                      schemas: Sequence[Mapping[str, Any]]) -> int:
    """Estimate the fixed part of every request: system prompt and tools."""
    return (len(system) + len(json.dumps(list(schemas)))) // CHARS_PER_TOKEN


class Compactor:
    """Fold old turns into a summary when the conversation is too big.

    ``compact`` has the shape of the loop's ``TurnHook``: it takes the
    history and returns the history to send. ``overhead_tokens`` is the
    part of every request that is not history (see
    ``estimate_overhead``); it counts against the budget.

    Measuring: pass ``observe`` the loop's events. It reads the input
    size the vendor reports for each request, so the next measurement
    is that real size plus an estimate for the messages added since, and
    the estimate is scaled by how far it was off last time. Without
    reported sizes (or before the first one) it estimates.

    Folding: the newest ``keep_recent`` messages are kept, or fewer, so
    that what is kept fits in ``fold_to`` of the budget. Folding well
    below the budget leaves room for the summary and for several more
    steps before the next fold, instead of a summary call at every step.
    A tool call and its result are never separated.

    ``on_event`` receives ``("compact", {"before", "after", "target",
    "summarised", "kept"})`` each time it folds, and
    ``("budget_warning", {"reason", "after", "target", "budget"})`` the
    first time a fold cannot reach its target, or cannot even get under
    the budget. That means the budget is too small for this task: the
    fixed part of every request (system prompt, tool schemas) plus the
    newest step and the summary do not fit, so it would otherwise fold
    silently at every step.
    """

    def __init__(
        self,
        provider: Provider,
        budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        keep_recent: int = KEEP_RECENT,
        on_event: EventHandler | None = None,
        overhead_tokens: int = 0,
        fold_to: float = FOLD_TO,
    ) -> None:
        if not 0 < fold_to < 1:
            raise ValueError("fold_to must be between 0 and 1")
        if budget_tokens <= 0 or keep_recent < 1 or overhead_tokens < 0:
            raise ValueError("budget_tokens and keep_recent must be "
                             "positive, overhead_tokens not negative")
        if overhead_tokens >= budget_tokens:
            raise ValueError(
                f"budget_tokens ({budget_tokens}) must be larger than the "
                f"fixed overhead of every request ({overhead_tokens})")
        self.provider = provider
        self.budget_tokens = budget_tokens
        self.overhead_tokens = overhead_tokens
        self.keep_recent = keep_recent
        self.fold_to = fold_to
        self.on_event = on_event
        self.compactions = 0
        self.warnings: list[str] = []   # budget_warning reasons, once each
        self.scale = 1.0              # reported tokens per estimated token
        self._sent_count = 0          # messages in the last request sent
        self._sent_estimate = 0       # its estimated size
        self._reported = 0            # its size as the vendor reported it

    def observe(self, kind: str, payload: Mapping[str, Any]) -> None:
        """Event handler: note the reported size of the last request."""
        if kind == "assistant":
            usage = payload.get("usage") or {}
            self._reported = int(usage.get("input") or 0)

    def compact(self, messages: list[Message]) -> list[Message]:
        """Return ``messages``, or a shorter equivalent if over budget."""
        before = self.measure(messages)
        if before <= self.budget_tokens:
            self._sent(messages)
            return messages

        cut = self._choose_cut(messages)
        if cut < 2:          # too little old history to be worth folding
            self._sent(messages)
            return messages
        old, recent = messages[:cut], messages[cut:]

        summary = self.provider.complete(
            SUMMARY_PROMPT,
            [Message(role="user", text="Summarise this transcript:\n\n"
                                       + transcript(old))],
        )["text"].strip()
        compacted = _prepend_summary(f"{SUMMARY_HEADER}\n{summary}", recent)

        self.compactions += 1
        after = self._estimate(compacted)
        target = int(self.budget_tokens * self.fold_to)
        self._emit("compact", {"before": before, "after": after,
                               "target": target, "summarised": len(old),
                               "kept": len(recent)})
        if after > self.budget_tokens:
            self._warn("over budget even after folding", after, target)
        elif after > target:
            self._warn("cannot fold down to the target", after, target)
        self._sent(compacted)
        return compacted

    def _emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(kind, payload)

    def _warn(self, reason: str, after: int, target: int) -> None:
        """Report a budget problem, once per kind of problem."""
        if reason in self.warnings:
            return
        self.warnings.append(reason)
        self._emit("budget_warning", {
            "reason": f"{reason}: about {after} tokens after a fold, "
                      f"target {target}, budget {self.budget_tokens}; "
                      "raise the budget or give the task fewer tools",
            "after": after, "target": target,
            "budget": self.budget_tokens})

    def measure(self, messages: Sequence[Message]) -> int:
        """The size of the request ``messages`` would make, in tokens."""
        if self._reported and self._sent_count <= len(messages):
            self.scale = self._reported / max(self._sent_estimate, 1)
            added = estimate_tokens(messages[self._sent_count:])
            return self._reported + round(self.scale * added)
        return self._estimate(messages)

    def _estimate(self, messages: Sequence[Message]) -> int:
        return round(self.scale
                     * (estimate_tokens(messages) + self.overhead_tokens))

    def _sent(self, messages: Sequence[Message]) -> None:
        """Remember the request about to be sent, to pair with its usage."""
        self._sent_count = len(messages)
        self._sent_estimate = estimate_tokens(messages) + self.overhead_tokens
        self._reported = 0

    def _choose_cut(self, messages: Sequence[Message]) -> int:
        """Keep as many recent messages as fit in ``fold_to`` of the budget.

        At least the newest message (with its call, if it is a result)
        is kept.
        """
        room = int(self.budget_tokens * self.fold_to)
        for keep in range(self.keep_recent, 0, -1):
            cut = _safe_cut(messages, len(messages) - keep)
            if self._estimate(messages[cut:]) <= room:
                return cut
        return _safe_cut(messages, len(messages) - 1)


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
