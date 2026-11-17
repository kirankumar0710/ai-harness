"""Orchestration: many agents, one goal.

Nothing clever: a thread pool of independent ``run_task`` calls, each in
its own working directory. The plan (which jobs, in which folders) is
plain code out here; the intelligence stays inside each agent.

    results = run_fleet(provider, [
        Job("slugify", "work/slugify", "Build slugify.py with tests"),
        Job("roman", "work/roman", "Build roman.py with tests"),
    ], mode="yolo", max_workers=2)

Design rules:

* Jobs are independent: a job never sees another job's conversation or
  folder. Two jobs may not share a working directory, because their
  edits would collide.
* A failing job is reported, never raised: one bad job cannot stop the
  others. Ctrl-C cancels the jobs that have not started.
* The fleet shares one approver and one event handler. Both are called
  under a lock, so a console approver asks one question at a time and
  output lines do not interleave. Each event carries ``"job"``.
* With ``sessions=True`` every job keeps a session log in its own
  folder, so a crashed fleet can be resumed job by job.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .context import DEFAULT_BUDGET_TOKENS
from .harness import run_task
from .loop import EventHandler
from .providers import Provider, ToolCall
from .security import Approver, Mode, deny_all
from .session import open_session


@dataclass(frozen=True)
class Job:
    name: str
    workdir: str
    task: str


@dataclass
class JobResult:
    name: str
    ok: bool
    answer: str                 # the final answer, or the error
    seconds: float
    session: str | None = None


def _ignore_event(kind: str, payload: Mapping[str, Any]) -> None:
    return None


def check_jobs(jobs: Iterable[Job]) -> list[Job]:
    """Return the jobs as a list; refuse duplicates and shared folders."""
    jobs = list(jobs)
    names: set[str] = set()
    folders: dict[str, str] = {}
    for job in jobs:
        if job.name in names:
            raise ValueError(f"two jobs are named {job.name!r}")
        names.add(job.name)
        folder = os.path.realpath(job.workdir)
        if folder in folders:
            raise ValueError(f"jobs {folders[folder]!r} and {job.name!r} "
                             f"share the working directory {job.workdir!r}")
        folders[folder] = job.name
    return jobs


def run_fleet(
    provider: Provider | Callable[[], Provider],
    jobs: Iterable[Job],
    *,
    mode: Mode = "safe",
    approver: Approver = deny_all,
    max_workers: int = 4,
    on_event: EventHandler = _ignore_event,
    sessions: bool = False,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    max_turns: int = 80,
    extra_prompt: str = "",
    subagent_depth: int = 0,
) -> list[JobResult]:
    """Run every job in parallel; return one result per job, in order.

    ``provider`` is shared by all jobs (providers keep no state between
    calls), or a function that makes one per job. Folders are created if
    missing.
    """
    jobs = check_jobs(jobs)
    lock = threading.Lock()

    def make_provider() -> Provider:
        return provider if isinstance(provider, Provider) else provider()

    def locked_approver(call: ToolCall, reason: str) -> bool | str:
        with lock:
            return approver(call, reason)

    def run_one(job: Job) -> JobResult:
        def events(kind: str, payload: Mapping[str, Any]) -> None:
            with lock:
                on_event(kind, {**payload, "job": job.name})

        started = time.monotonic()
        session = None
        try:
            os.makedirs(job.workdir, exist_ok=True)
            messages: list = []
            if sessions:
                session, messages = open_session(job.workdir, job.name)
            answer = run_task(
                make_provider(), job.workdir, job.task, mode=mode,
                approver=locked_approver, budget_tokens=budget_tokens,
                max_turns=max_turns, on_event=events, messages=messages,
                extra_prompt=extra_prompt, session=session,
                subagent_depth=subagent_depth)
            ok = True
        except Exception as error:      # one job's failure is its result
            answer, ok = f"{type(error).__name__}: {error}", False
        return JobResult(job.name, ok, answer,
                         round(time.monotonic() - started, 2), session)

    results: dict[str, JobResult] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, max_workers))
    try:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results[result.name] = result
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    return [results[job.name] for job in jobs]
