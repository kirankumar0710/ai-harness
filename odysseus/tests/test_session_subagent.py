"""Offline tests: the session log, resume after a crash, and sub-agents."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from odysseus_v2 import (SessionCorrupt, core_tools, latest_session,
                         load_session, new_session, open_session, run_loop,
                         run_task, tool, tool_map)
from odysseus_v2.providers import Completion, Provider
from odysseus_v2.session import (INTERRUPTED, append_message, repair,
                                 session_recorder)
from odysseus_v2.subagent import MAX_REPORT_CHARS, SUBAGENT_PROMPT


class ScriptedProvider(Provider):
    """Replays canned replies and records every request."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True):
        self.calls.append({"system": system,
                           "messages": [dict(m) for m in messages],
                           "tools": [t["name"] for t in tools]})
        return self.replies.pop(0)

    def _build_request(self, *args):   # pragma: no cover - unused
        raise NotImplementedError

    def _parse_response(self, raw):    # pragma: no cover - unused
        raise NotImplementedError


def reply(text="", calls=()):
    return Completion(text=text, tool_calls=list(calls),
                      usage={"input": 0, "output": 0})


def call(tool_name, call_id="c1", **args):
    return {"id": call_id, "name": tool_name, "args": args}


def read_log(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()


class SessionFileTests(Sandbox):
    def test_sessions_in_the_same_second_get_separate_files(self):
        with mock.patch("odysseus_v2.session.time.strftime",
                        return_value="20261116-090000"):
            first = new_session(self.workdir, "fix tests!")
            second = new_session(self.workdir, "fix tests!")
        self.assertNotEqual(first, second)
        self.assertEqual(os.path.basename(first),
                         "20261116-090000-fix-tests-.jsonl")
        self.assertTrue(second.endswith("-fix-tests--2.jsonl"))

    def test_round_trip(self):
        path = new_session(self.workdir)
        messages = [{"role": "user", "text": "héllo"},
                    {"role": "assistant", "text": "hi", "tool_calls": []}]
        for message in messages:
            append_message(path, message)
        self.assertEqual(load_session(path), messages)

    def test_torn_last_line_is_dropped(self):
        path = new_session(self.workdir)
        append_message(path, {"role": "user", "text": "task"})
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"role": "assistant", "te')
        self.assertEqual(load_session(path),
                         [{"role": "user", "text": "task"}])

    def test_resume_cuts_a_torn_tail_so_appending_continues(self):
        path = new_session(self.workdir)
        append_message(path, {"role": "user", "text": "task"})
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"role": "assistant", "te')
        _, messages = open_session(self.workdir, resume=True)
        append_message(path, {"role": "user", "text": "next"})
        self.assertEqual(load_session(path),
                         [{"role": "user", "text": "task"},
                          {"role": "user", "text": "next"}])

    def test_resume_adds_a_missing_final_newline(self):
        path = new_session(self.workdir)
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"role": "user", "text": "task"}')   # no newline
        open_session(self.workdir, resume=True)
        append_message(path, {"role": "user", "text": "next"})
        self.assertEqual(len(load_session(path)), 2)

    def test_load_session_does_not_change_the_file(self):
        path = new_session(self.workdir)
        append_message(path, {"role": "assistant", "text": "",
                              "tool_calls": [call("bash", "a")]})
        with open(path, "rb") as f:
            before = f.read()
        self.assertEqual(load_session(path)[-1]["text"], INTERRUPTED)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_bad_middle_line_is_corruption(self):
        path = new_session(self.workdir)
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"role": "user", "text": "a"}\nnot json\n'
                    '{"role": "user", "text": "b"}\n')
        with self.assertRaises(SessionCorrupt):
            load_session(path)

    def test_latest_and_open_session(self):
        self.assertIsNone(latest_session(self.workdir))
        path, messages = open_session(self.workdir, "t", resume=True)
        self.assertEqual(messages, [])               # nothing to resume
        append_message(path, {"role": "user", "text": "task"})
        resumed, messages = open_session(self.workdir, "t", resume=True)
        self.assertEqual(resumed, path)
        self.assertEqual(messages, [{"role": "user", "text": "task"}])
        fresh, messages = open_session(self.workdir, "t")
        self.assertNotEqual(fresh, path)
        self.assertEqual(messages, [])


class RepairTests(unittest.TestCase):
    def test_unanswered_calls_are_answered_by_id(self):
        messages = [
            {"role": "user", "text": "task"},
            {"role": "assistant", "text": "",
             "tool_calls": [call("bash", "a"), call("edit_file", "b")]},
            # only the second call's result reached the log
            {"role": "tool", "tool_call_id": "b", "name": "edit_file",
             "text": "Edited", "is_error": False},
        ]
        repaired = repair(messages)
        self.assertEqual(len(repaired), 4)
        self.assertEqual(repaired[3]["tool_call_id"], "a")
        self.assertEqual(repaired[3]["text"], INTERRUPTED)
        self.assertTrue(repaired[3]["is_error"])
        self.assertIn("may or may not have run", INTERRUPTED)

    def test_complete_history_is_unchanged(self):
        messages = [{"role": "user", "text": "task"},
                    {"role": "assistant", "text": "done", "tool_calls": []}]
        self.assertEqual(repair(list(messages)), messages)


class Crash(BaseException):
    """Stands in for the process dying (not caught by the loop)."""


class RunTaskSessionTests(Sandbox):
    def test_every_message_is_logged(self):
        path = new_session(self.workdir)
        provider = ScriptedProvider([
            reply(calls=[call("list_files", "c1")]),
            reply("done"),
        ])
        messages = []
        run_task(provider, self.workdir, "list", mode="yolo",
                 messages=messages, session=path)
        self.assertEqual(read_log(path), messages)
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "tool", "assistant"])

    def test_crash_then_resume(self):
        @tool("Deploy the app.")
        def deploy() -> str:
            raise Crash()

        path = new_session(self.workdir)
        provider = ScriptedProvider([reply(calls=[call("deploy", "d1")])])
        with self.assertRaises(Crash):
            run_task(provider, self.workdir, "deploy it", mode="yolo",
                     tools=[deploy], session=path)

        # A new process: replay the log and continue.
        resumed_path, messages = open_session(self.workdir, resume=True)
        self.assertEqual(resumed_path, path)
        self.assertEqual(messages[-1]["text"], INTERRUPTED)
        provider = ScriptedProvider([reply("checked; it did not deploy")])
        answer = run_task(provider, self.workdir, "Continue.", mode="yolo",
                          tools=[deploy], messages=messages, session=path)
        self.assertEqual(answer, "checked; it did not deploy")
        sent = provider.calls[0]["messages"]
        self.assertEqual([m["role"] for m in sent],
                         ["user", "assistant", "tool", "user"])
        # The repaired result is in the log too, so a second resume
        # sees the same history.
        self.assertEqual(load_session(path), messages)

    def test_messages_after_a_compaction_are_still_logged(self):
        # The original kept an index into the list; after a compaction
        # shrank the list, the next reply landed below it and was lost.
        path = new_session(self.workdir)
        provider = ScriptedProvider([
            reply(calls=[call("list_files", "c1")]),
            reply("done"),
        ])
        messages = [{"role": "user", "text": f"old {i}"} for i in range(6)]
        run_loop(provider, "system", messages,
                 tool_map(core_tools(self.workdir)),
                 on_event=session_recorder(path),
                 before_turn=lambda history: history[-1:])
        logged = read_log(path)
        self.assertEqual([m["role"] for m in logged],
                         ["assistant", "tool", "assistant"])
        self.assertEqual(logged[-1]["text"], "done")


class HarnessDirTests(Sandbox):
    def test_file_tools_refuse_the_session_folder(self):
        path = new_session(self.workdir)
        append_message(path, {"role": "user", "text": "task"})
        tools = tool_map(core_tools(self.workdir))
        relative = os.path.relpath(path, self.workdir)
        for name, args in [("read_file", {"path": relative}),
                           ("write_file", {"path": relative,
                                           "content": "{}"}),
                           ("read_file", {"path": ".odysseus/../.odysseus/"
                                          "sessions/x.jsonl"})]:
            with self.assertRaises(PermissionError):
                tools[name].run(**args)
        self.assertNotIn(".odysseus", tools["list_files"].run())


class SubagentTests(Sandbox):
    def test_not_offered_by_default(self):
        provider = ScriptedProvider([reply("ok")])
        run_task(provider, self.workdir, "task")
        self.assertNotIn("spawn_agent", provider.calls[0]["tools"])

    def test_child_gets_a_fresh_conversation_and_returns_its_report(self):
        events = []
        provider = ScriptedProvider([
            reply(calls=[call("spawn_agent", "s1",
                              task="Count the files.")]),   # parent
            reply("2 files"),                                 # child
            reply("The child found 2 files."),                # parent
        ])
        answer = run_task(provider, self.workdir, "How many files?",
                          mode="yolo", subagent_depth=1,
                          on_event=lambda k, p: events.append((k, dict(p))))
        self.assertEqual(answer, "The child found 2 files.")

        parent_first, child, parent_last = provider.calls
        self.assertIn("spawn_agent", parent_first["tools"])
        # the child sees only its brief, and cannot spawn again
        self.assertEqual(child["messages"],
                         [{"role": "user", "text": "Count the files."}])
        self.assertIn(SUBAGENT_PROMPT, child["system"])
        self.assertNotIn("spawn_agent", child["tools"])
        # the parent sees one call and the report as its result
        self.assertEqual(parent_last["messages"][-1]["text"], "2 files")
        child_events = [p for k, p in events if p.get("subagent") == 1]
        self.assertTrue(child_events)
        self.assertTrue(all(p.get("subagent") is None
                            for k, p in events if k == "tool_start"
                            and p["name"] == "spawn_agent"))

    def test_depth_two_offers_the_tool_one_level_down(self):
        provider = ScriptedProvider([
            reply(calls=[call("spawn_agent", "s1", task="level 1")]),
            reply(calls=[call("spawn_agent", "s2", task="level 2")]),
            reply("level 2 done"),
            reply("level 1 done"),
            reply("all done"),
        ])
        events = []
        run_task(provider, self.workdir, "go", mode="yolo",
                 subagent_depth=2,
                 on_event=lambda k, p: events.append((k, dict(p))))
        tools_per_call = [c["tools"] for c in provider.calls]
        self.assertIn("spawn_agent", tools_per_call[1])       # level 1
        self.assertNotIn("spawn_agent", tools_per_call[2])    # level 2
        levels = {p.get("subagent") for k, p in events
                  if k == "assistant"}
        self.assertEqual(levels, {None, 1, 2})

    def test_report_is_capped(self):
        provider = ScriptedProvider([
            reply(calls=[call("spawn_agent", "s1", task="write a lot")]),
            reply("x" * (MAX_REPORT_CHARS + 500)),
            reply("ok"),
        ])
        run_task(provider, self.workdir, "go", mode="yolo",
                 subagent_depth=1)
        result = provider.calls[2]["messages"][-1]["text"]
        self.assertTrue(result.endswith(
            f"report cut at {MAX_REPORT_CHARS} of "
            f"{MAX_REPORT_CHARS + 500} chars"))

    def test_child_runs_under_the_parents_policy(self):
        # read-only: spawning is allowed, the child's write is blocked
        provider = ScriptedProvider([
            reply(calls=[call("spawn_agent", "s1", task="write a file")]),
            reply(calls=[call("write_file", "w1", path="a.txt",
                              content="x")]),
            reply("I could not write: read-only."),
            reply("The child could not write."),
        ])
        run_task(provider, self.workdir, "go", mode="read-only",
                 subagent_depth=1)
        child_result = provider.calls[2]["messages"][-1]
        self.assertTrue(child_result["text"].startswith("BLOCKED"))
        self.assertFalse(os.path.exists(os.path.join(self.workdir, "a.txt")))
        spawn_result = provider.calls[3]["messages"][-1]
        self.assertEqual(spawn_result["text"], "I could not write: read-only.")

    def test_child_messages_stay_out_of_the_parents_log(self):
        path = new_session(self.workdir)
        provider = ScriptedProvider([
            reply(calls=[call("spawn_agent", "s1", task="brief")]),
            reply("child report"),
            reply("done"),
        ])
        messages = []
        run_task(provider, self.workdir, "go", mode="yolo",
                 subagent_depth=1, messages=messages, session=path)
        self.assertEqual(read_log(path), messages)
        self.assertNotIn("brief", [m.get("text") for m in messages])


if __name__ == "__main__":
    unittest.main()
