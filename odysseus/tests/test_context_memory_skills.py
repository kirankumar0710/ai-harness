"""Offline tests: compaction, memory, skills and the connected harness.

    python -m unittest discover -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

from odysseus_v2 import Compactor, build_system_prompt, run_loop, run_task
from odysseus_v2.context import (MAX_CHARS_PER_MESSAGE, SUMMARY_HEADER,
                                 estimate_tokens, transcript)
from odysseus_v2.memory import MAX_MEMORY_CHARS, MEMORY_FILE, memory_tools
from odysseus_v2.providers import Completion, Provider
from odysseus_v2.security import Policy
from odysseus_v2.skills import catalog, catalog_prompt, skill_tools


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


def user(text):
    return {"role": "user", "text": text}


def assistant(text="", calls=()):
    return {"role": "assistant", "text": text, "tool_calls": list(calls)}


def result(call_id, text="ok"):
    return {"role": "tool", "tool_call_id": call_id, "name": "bash",
            "text": text, "is_error": False}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------
# Context and compaction
# --------------------------------------------------------------------------


class CompactorTests(unittest.TestCase):

    def long_history(self):
        """user, then 5 x (assistant call, tool result), then a user turn."""
        history = [user("Build the thing. " + "x" * 400)]
        for i in range(5):
            history += [assistant("step", [call("bash", f"c{i}")]),
                        result(f"c{i}", "y" * 400)]
        return history

    def test_under_budget_returns_the_same_list(self):
        provider = ScriptedProvider([])
        history = [user("hi")]
        self.assertIs(Compactor(provider, 1000).compact(history), history)
        self.assertEqual(provider.calls, [])

    def test_over_budget_folds_old_turns_into_a_summary(self):
        provider = ScriptedProvider([reply("Task: build. Done: steps 0-1.")])
        events = []
        compactor = Compactor(provider, budget_tokens=100, keep_recent=6,
                              on_event=lambda k, p: events.append((k, p)))
        history = self.long_history()
        compacted = compactor.compact(history)

        self.assertEqual(compacted[0]["role"], "user")
        self.assertTrue(compacted[0]["text"].startswith(SUMMARY_HEADER))
        self.assertIn("Done: steps 0-1.", compacted[0]["text"])
        self.assertEqual(compacted[1:], history[-6:])
        self.assertLess(estimate_tokens(compacted), estimate_tokens(history))
        self.assertEqual(compactor.compactions, 1)
        self.assertEqual(events[0][0], "compact")
        self.assertEqual(events[0][1]["summarised"], len(history) - 6)
        sent = provider.calls[0]["messages"][0]["text"]
        self.assertIn("called bash(", sent)

    def test_cut_never_separates_a_call_from_its_result(self):
        provider = ScriptedProvider([reply("summary")])
        history = self.long_history()
        # keep_recent=5 would start the recent part on a tool result.
        compacted = Compactor(provider, 100, keep_recent=5).compact(history)
        self.assertEqual(compacted[1]["role"], "assistant")
        self.assertEqual(compacted[2]["role"], "tool")
        self.assertEqual(compacted[1:], history[-6:])

    def test_summary_merges_into_a_leading_user_message(self):
        provider = ScriptedProvider([reply("summary")])
        history = self.long_history() + [user("Now add tests.")]
        compacted = Compactor(provider, 100, keep_recent=1).compact(history)
        self.assertEqual(len(compacted), 1)
        self.assertTrue(compacted[0]["text"].startswith(SUMMARY_HEADER))
        self.assertTrue(compacted[0]["text"].endswith("Now add tests."))

    def test_transcript_keeps_the_end_of_long_messages(self):
        long_result = "start\n" + "filler\n" * 1000 + "VALUE: 42\n"
        text = transcript([result("c1", long_result)])
        self.assertIn("start", text)
        self.assertIn("VALUE: 42", text)
        self.assertIn("chars cut", text)
        self.assertLess(len(text), MAX_CHARS_PER_MESSAGE + 100)

    def test_too_little_history_is_left_alone(self):
        provider = ScriptedProvider([])
        history = [user("x" * 4000), assistant("y" * 4000)]
        self.assertIs(Compactor(provider, 10, keep_recent=1).compact(history),
                      history)

    def test_loop_sends_the_compacted_history(self):
        provider = ScriptedProvider([reply("summary"), reply("done")])
        compactor = Compactor(provider, budget_tokens=100, keep_recent=6)
        messages = self.long_history()
        run_loop(provider, "", messages, {}, before_turn=compactor.compact)
        sent = provider.calls[1]["messages"]
        self.assertTrue(sent[0]["text"].startswith(SUMMARY_HEADER))
        self.assertEqual(len(messages), 8)    # summary + 6 kept + answer

    def test_bad_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            Compactor(ScriptedProvider([]), budget_tokens=0)


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


class MemoryTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = self.tmp.name
        self.remember = memory_tools(self.workdir)[0]

    def test_prompt_has_rules_environment_memory_and_extras(self):
        write(os.path.join(self.workdir, MEMORY_FILE), "- use python3\n")
        prompt = build_system_prompt(self.workdir, "", "EXTRA SECTION")
        self.assertIn("You are Odysseus", prompt)
        self.assertIn(os.path.realpath(self.workdir), prompt)
        self.assertIn("Project memory (AGENT_MEMORY.md):\n- use python3",
                      prompt)
        self.assertTrue(prompt.endswith("EXTRA SECTION"))

    def test_no_memory_file_means_no_memory_section(self):
        self.assertNotIn("Project memory", build_system_prompt(self.workdir))

    def test_remember_appends_one_tidy_line(self):
        answer = self.remember.run(note="  run tests\n with  unittest ")
        self.assertEqual(answer, "Remembered in AGENT_MEMORY.md")
        self.remember.run(note="second fact")
        with open(os.path.join(self.workdir, MEMORY_FILE)) as f:
            self.assertEqual(f.read(),
                             "- run tests with unittest\n- second fact\n")

    def test_remember_rejects_empty_and_long_notes(self):
        self.assertIn("empty", self.remember.run(note="   "))
        self.assertIn("characters", self.remember.run(note="x" * 600))

    def test_large_memory_keeps_the_newest_notes(self):
        notes = "".join(f"- note {i}\n" for i in range(5000))
        write(os.path.join(self.workdir, MEMORY_FILE), notes)
        prompt = build_system_prompt(self.workdir)
        self.assertIn("older notes truncated", prompt)
        self.assertIn("- note 4999", prompt)
        self.assertNotIn("- note 0\n", prompt)
        self.assertLess(len(prompt), MAX_MEMORY_CHARS + 2000)


# --------------------------------------------------------------------------
# Skills
# --------------------------------------------------------------------------


class SkillTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = self.tmp.name
        skills = os.path.join(self.workdir, "skills")
        write(os.path.join(skills, "testing", "SKILL.md"),
              "---\ndescription: How to run the tests\n---\n"
              "Use unittest.\ndescription: not this one\n")
        write(os.path.join(skills, "plain", "SKILL.md"),
              "description: No front matter\n\nBody text.\n")
        write(os.path.join(skills, "nodesc", "SKILL.md"),
              "Just instructions.\n\ndescription: buried, ignored\n")
        os.makedirs(os.path.join(skills, "empty-folder"))
        self.use_skill = skill_tools(self.workdir)[0]

    def test_catalog_reads_descriptions_from_the_header_only(self):
        found = catalog(self.workdir)
        self.assertEqual(sorted(found), ["nodesc", "plain", "testing"])
        self.assertEqual(found["testing"]["description"],
                         "How to run the tests")
        self.assertEqual(found["plain"]["description"], "No front matter")
        self.assertEqual(found["nodesc"]["description"], "nodesc")

    def test_catalog_prompt_lists_every_skill(self):
        prompt = catalog_prompt(self.workdir)
        self.assertIn("use_skill", prompt)
        self.assertIn("- testing: How to run the tests", prompt)

    def test_no_skills_folder_means_no_section(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(catalog_prompt(empty), "")

    def test_use_skill_returns_the_full_text(self):
        self.assertIn("Use unittest.", self.use_skill.run(name="testing"))

    def test_use_skill_accepts_names_not_paths(self):
        for name in ("missing", "../testing", "testing/SKILL.md"):
            with self.subTest(name=name):
                self.assertTrue(self.use_skill.run(name=name)
                                .startswith("ERROR: no skill named"))

    def test_use_skill_is_a_read_tool(self):
        self.assertIsNone(Policy("read-only").check(call("use_skill")))


# --------------------------------------------------------------------------
# The connected harness
# --------------------------------------------------------------------------


class HarnessTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = self.tmp.name
        write(os.path.join(self.workdir, MEMORY_FILE), "- use python3\n")
        write(os.path.join(self.workdir, "skills", "testing", "SKILL.md"),
              "---\ndescription: How to run the tests\n---\nUse unittest.\n")

    def test_run_task_wires_tools_prompt_and_gate(self):
        provider = ScriptedProvider([
            reply(calls=[call("use_skill", "c1", name="testing"),
                         call("remember", "c2", note="tests use unittest")]),
            reply("done"),
        ])
        messages = []
        answer = run_task(provider, self.workdir, "Fix it", mode="read-only",
                          messages=messages)

        self.assertEqual(answer, "done")
        first = provider.calls[0]
        self.assertIn("- use python3", first["system"])
        self.assertIn("- testing: How to run the tests", first["system"])
        self.assertEqual(sorted(first["tools"]),
                         ["bash", "edit_file", "grep", "list_files",
                          "read_file", "remember", "use_skill",
                          "write_file"])
        results = [m for m in messages if m["role"] == "tool"]
        self.assertIn("Use unittest.", results[0]["text"])
        self.assertTrue(results[1]["text"].startswith("BLOCKED"))

    def test_messages_carry_over_to_the_next_task(self):
        provider = ScriptedProvider([reply("one"), reply("two")])
        messages = []
        run_task(provider, self.workdir, "first", messages=messages)
        run_task(provider, self.workdir, "second", messages=messages)
        self.assertEqual([m["text"] for m in messages],
                         ["first", "one", "second", "two"])

    def test_missing_workdir_is_a_clear_error(self):
        with self.assertRaises(NotADirectoryError):
            run_task(ScriptedProvider([]), "/no/such/dir", "task")


if __name__ == "__main__":
    unittest.main()
