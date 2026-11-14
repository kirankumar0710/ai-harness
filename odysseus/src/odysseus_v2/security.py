"""Security: the permission gate.

Every tool call passes through one checkpoint before it runs. Three
layers, in order:

  1. hard deny rules - catastrophic shell commands are never allowed,
     in any mode
  2. the mode        - read-only / safe / yolo decides what needs approval
  3. the approver    - a human (or an automatic answer in headless runs)

``Policy.check`` has the shape of the loop's ``ToolGate``: it takes a
neutral ``ToolCall`` and returns a block reason, or ``None`` to allow.
Wire it in with ``run_loop(..., before_tool=policy.check)``; the loop
answers a blocked call with ``BLOCKED: <reason>`` so the model can
adapt instead of retrying blindly.

The file-path jail lives in ``tools.py``; this gate covers behaviour.
Deny rules are a seat belt, not a sandbox: a determined model can
phrase a command they do not match. Run untrusted work in a container.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Literal, get_args

from .providers import ToolCall

Mode = Literal["read-only", "safe", "yolo"]
Approver = Callable[[ToolCall, str], bool]

MODES: tuple[str, ...] = get_args(Mode)

# Tools with no side effects: allowed in every mode without asking.
READ_TOOLS = frozenset({"read_file", "list_files", "grep"})

# Tools whose ``command`` argument is a shell command to screen.
SHELL_TOOLS = frozenset({"bash"})

DENY_PATTERNS: tuple[str, ...] = (
    # rm -rf on /, /*, ~ or $HOME
    r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*\s+)+(/\*?|~/?|\$HOME/?)(\s|;|&|$)",
    r"\bsudo\b",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+.*\bof=/dev/",
    r"\b(curl|wget)\b[^|;&]*\|\s*(ba|z)?sh\b",       # curl ... | sh
    r"\bgit\s+push\b.*(--force\b|\s-f\b)",
    r">\s*/dev/(sd|nvme|disk)",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",     # fork bomb
)


def deny_all(call: ToolCall, reason: str) -> bool:
    """The default approver: say no. Safe when nobody is watching."""
    return False


def approve_all(call: ToolCall, reason: str) -> bool:
    """Approve everything. For headless runs that rely on the jail
    and the deny rules alone."""
    return True


def console_approver(call: ToolCall, reason: str) -> bool:
    """Ask on the terminal: ``y`` approves, anything else refuses."""
    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
    try:
        answer = input(f"\n[{reason}] {call['name']}({args})  allow? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


class Policy:
    """Decide, per tool call, whether it may run.

    ``mode``:
      * ``"read-only"`` - only ``read_tools`` run; everything else is
        blocked without asking.
      * ``"safe"`` - read tools run; anything else asks the approver.
      * ``"yolo"`` - everything runs that the deny rules allow.

    ``approver(call, reason) -> bool`` is consulted in ``"safe"`` mode.
    It defaults to ``deny_all`` so an unattended run never acts by
    accident; the CLI wires it to ``console_approver``.
    """

    def __init__(
        self,
        mode: Mode = "safe",
        approver: Approver = deny_all,
        *,
        read_tools: Iterable[str] = READ_TOOLS,
        shell_tools: Iterable[str] = SHELL_TOOLS,
        deny_patterns: Iterable[str] = DENY_PATTERNS,
    ) -> None:
        if mode not in MODES:
            raise ValueError(
                f"unknown mode {mode!r}; choose one of: {', '.join(MODES)}")
        self.mode: Mode = mode
        self.approver = approver
        self.read_tools = frozenset(read_tools)
        self.shell_tools = frozenset(shell_tools)
        self._deny = [re.compile(p) for p in deny_patterns]

    def check(self, call: ToolCall) -> str | None:
        """Return a block reason, or ``None`` to allow the call."""
        denied = self._denied_by(call)
        if denied:
            return f"command matches deny rule: {denied}"

        if call["name"] in self.read_tools or self.mode == "yolo":
            return None
        if self.mode == "read-only":
            return ("policy is read-only; only "
                    f"{', '.join(sorted(self.read_tools))} are allowed")

        # mode == "safe": side effects need approval.
        try:
            approved = self.approver(call, "side-effecting tool call")
        except Exception as error:   # a broken prompt must not allow
            return f"approval failed: {type(error).__name__}: {error}"
        return None if approved else "not approved by user"

    __call__ = check

    def _denied_by(self, call: ToolCall) -> str | None:
        """Return the first deny pattern a shell command matches."""
        if call["name"] not in self.shell_tools:
            return None
        command = str(call["args"].get("command", ""))
        for pattern in self._deny:
            if pattern.search(command):
                return pattern.pattern
        return None
