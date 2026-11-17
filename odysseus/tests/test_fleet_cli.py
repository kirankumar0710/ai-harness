"""Offline tests: the fleet (parallel jobs) and the command line."""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

from odysseus_v2 import Job, latest_session, load_session, run_fleet
from odysseus_v2 import cli
from odysseus_v2.providers import Completion, ConfigError, Provider


def reply(text="", calls=()):
    return Completion(text=text, tool_calls=list(calls),
                      usage={"input": 0, "output": 0})


class WriteOnce(Provider):
    """Writes NAME.txt holding the job's folder name, then answers.

    Stateless: it decides from the conversation, so one instance can be
    shared by every job of a fleet.
    """

    def __init__(self, delay=0.0, barrier=None) -> None:
        self.delay, self.barrier = delay, barrier
        self.seen: list[list[dict]] = []
        self.lock = threading.Lock()

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True):
        with self.lock:
            self.seen.append([dict(m) for m in messages])
        if self.barrier is not None:
            self.barrier.wait(timeout=5)     # breaks unless run in parallel
        time.sleep(self.delay)
        task = messages[0]["text"]
        if messages[-1].get("role") == "tool":
            return reply(f"done: {task}")
        return reply(calls=[{"id": "w1", "name": "write_file",
                             "args": {"path": "NAME.txt", "content": task}}])

    def _build_request(self, *args):   # pragma: no cover - unused
        raise NotImplementedError

    def _parse_response(self, raw):    # pragma: no cover - unused
        raise NotImplementedError


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def jobs(self, *names):
        return [Job(n, os.path.join(self.base, n), n) for n in names]


class FleetTests(Sandbox):
    def test_every_job_runs_in_its_own_folder_and_results_keep_order(self):
        events = []
        results = run_fleet(WriteOnce(), self.jobs("a", "b", "c"),
                            mode="yolo", max_workers=3,
                            on_event=lambda k, p: events.append(p["job"]))
        self.assertEqual([r.name for r in results], ["a", "b", "c"])
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(results[1].answer, "done: b")
        for name in "abc":
            with open(os.path.join(self.base, name, "NAME.txt")) as f:
                self.assertEqual(f.read(), name)
        self.assertEqual(set(events), {"a", "b", "c"})

    def test_jobs_really_run_in_parallel(self):
        # Each job's first request waits until both are waiting: this
        # only completes when the two jobs run at the same time.
        provider = WriteOnce(barrier=threading.Barrier(2))
        results = run_fleet(provider, self.jobs("a", "b"), mode="yolo",
                            max_workers=2)
        self.assertTrue(all(r.ok for r in results), results)

    def test_duplicate_names_and_shared_folders_are_refused(self):
        with self.assertRaisesRegex(ValueError, "two jobs are named"):
            run_fleet(WriteOnce(), [Job("a", self.base, "x"),
                                    Job("a", self.base + "/b", "y")])
        with self.assertRaisesRegex(ValueError, "share the working"):
            run_fleet(WriteOnce(), [Job("a", self.base, "x"),
                                    Job("b", self.base + "/.", "y")])

    def test_a_failing_job_does_not_stop_the_others(self):
        made = []

        def factory():
            made.append(1)
            if len(made) == 2:
                raise ConfigError("no key for this one")
            return WriteOnce()

        results = run_fleet(factory, self.jobs("a", "b", "c"), mode="yolo",
                            max_workers=1)
        self.assertEqual([r.ok for r in results], [True, False, True])
        self.assertEqual(results[1].answer, "ConfigError: no key for this one")

    def test_the_approver_is_asked_one_call_at_a_time(self):
        asking, most = [0], [0]

        def slow_approver(call, reason):
            asking[0] += 1
            most[0] = max(most[0], asking[0])
            time.sleep(0.05)
            asking[0] -= 1
            return True

        results = run_fleet(WriteOnce(), self.jobs("a", "b", "c"),
                            mode="safe", approver=slow_approver,
                            max_workers=3)
        self.assertTrue(all(r.ok for r in results))
        self.assertEqual(most[0], 1)

    def test_sessions_give_each_job_its_own_log(self):
        results = run_fleet(WriteOnce(), self.jobs("a", "b"), mode="yolo",
                            sessions=True)
        for result in results:
            self.assertTrue(result.session.startswith(
                os.path.join(self.base, result.name)))
            self.assertEqual(len(load_session(result.session)), 4)


class CliTests(Sandbox):
    def run_cli(self, provider, *argv):
        out = io.StringIO()
        with mock.patch.object(cli, "create_provider",
                               return_value=provider), \
                contextlib.redirect_stdout(out):
            code = cli.main(["-d", self.base, *argv])
        return code, out.getvalue()

    def test_headless_defaults_to_refusing_writes(self):
        code, out = self.run_cli(WriteOnce(), "-p", "task")
        self.assertEqual(code, 0)
        self.assertIn("writes and commands are refused", out)
        self.assertFalse(os.path.exists(os.path.join(self.base, "NAME.txt")))
        log = load_session(latest_session(self.base))
        self.assertTrue(log[2]["text"].startswith("BLOCKED"))

    def test_headless_yolo_writes(self):
        code, _ = self.run_cli(WriteOnce(), "-p", "task", "--mode", "yolo")
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(os.path.join(self.base, "NAME.txt")))

    def test_resume_continues_the_last_session(self):
        self.run_cli(WriteOnce(), "-p", "first", "--mode", "yolo")
        provider = WriteOnce()
        _, out = self.run_cli(provider, "-p", "second", "--mode", "yolo",
                              "--resume")
        self.assertIn("resumed 4 messages", out)
        first_request = provider.seen[0]
        self.assertEqual(first_request[0]["text"], "first")
        self.assertEqual(first_request[-1]["text"], "second")

    def test_ctrl_c_returns_130_and_leaves_a_valid_log(self):
        class Interrupted(WriteOnce):
            def complete(self, *args, **kwargs):
                raise KeyboardInterrupt

        code, out = self.run_cli(Interrupted(), "-p", "task")
        self.assertEqual(code, 130)
        self.assertIn("the session log is safe", out)
        self.assertEqual(load_session(latest_session(self.base)),
                         [{"role": "user", "text": "task"}])

    def test_a_missing_provider_is_an_error_not_a_traceback(self):
        err = io.StringIO()
        with mock.patch.object(cli, "create_provider",
                               side_effect=ConfigError("set a key")), \
                contextlib.redirect_stderr(err):
            code = cli.main(["-d", self.base, "-p", "task"])
        self.assertEqual(code, 2)
        self.assertIn("set a key", err.getvalue())


if __name__ == "__main__":
    unittest.main()
