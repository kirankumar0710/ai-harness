"""Sub-agents: the harness inside the harness.

A sub-agent is another ``run_task`` with a fresh conversation and one
task. The parent sees a single tool call; the child spends its own
context and returns only its final report. Much work in, a short
answer out: that asymmetry is why delegation scales.

Design rules:

* The child cannot see the parent's conversation, so the task must be a
  complete brief. The tool's description tells the model so.
* The child runs under the same mode and approver as the parent, so
  spawning one cannot do anything the parent could not. The gate
  therefore treats ``spawn_agent`` as a read tool; each of the child's
  own calls is gated as usual.
* Depth is limited: at the last level the tool is simply not offered,
  rather than offered and refused.
* The report is capped, so one child cannot flood the parent's context.
* A child has no session log. If the process dies, the parent's log
  shows the call as interrupted, and the parent decides what to redo.
"""

from __future__ import annotations

from collections.abc import Callable

from .tools import Tool, tool

MAX_REPORT_CHARS = 8_000

SUBAGENT_PROMPT = (
    "You are a sub-agent. Another agent gave you this task and cannot "
    "see your work: your final answer is your whole report to it. Make "
    "it complete and concise: what you found or changed, and anything "
    "left undone."
)


def subagent_tool(run_child: Callable[[str], str]) -> Tool:
    """Build ``spawn_agent``; ``run_child(task)`` runs a fresh agent."""

    @tool("Delegate a self-contained task to a fresh sub-agent with its "
          "own empty context. It cannot see this conversation, so give "
          "it a complete brief. Returns the sub-agent's final report.",
          task="Complete, self-contained instructions for the sub-agent")
    def spawn_agent(task: str) -> str:
        if not task.strip():
            return "ERROR: the task is empty - write a complete brief"
        report = run_child(task)
        if len(report) > MAX_REPORT_CHARS:
            report = (report[:MAX_REPORT_CHARS]
                      + f"\n... report cut at {MAX_REPORT_CHARS} of "
                        f"{len(report)} chars")
        return report

    return spawn_agent
