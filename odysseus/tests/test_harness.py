"""Offline tests: the loop and every provider's wire format.

No network and no API key: providers get a fake transport that records
the request and returns a canned vendor reply.

    python -m unittest discover -v
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
import urllib.error
from typing import Any
from unittest import mock

from odysseus_v2 import run_loop
from odysseus_v2.providers import (
    AnthropicConfig,
    AnthropicProvider,
    Completion,
    ConfigError,
    GeminiConfig,
    GeminiProvider,
    OpenAIConfig,
    OpenRouterConfig,
    OpenRouterProvider,
    Provider,
    ProviderError,
    create_provider,
)
from odysseus_v2.providers import base

DICE_SCHEMA = {"name": "roll_dice", "description": "Roll dice",
               "parameters": {"type": "object",
                              "properties": {"count": {"type": "integer"}},
                              "required": ["count"]}}


class Dice:
    name = "roll_dice"
    spec = {"schema": DICE_SCHEMA}

    def run(self, count: int) -> str:
        return "[4, 5, 6]"


class Broken:
    name = "broken"
    spec = {"schema": {"name": "broken", "description": "Always fails",
                       "parameters": {"type": "object", "properties": {}}}}

    def run(self) -> str:
        raise OSError("disk on fire")


class ScriptedProvider(Provider):
    """Replays canned completions; records every call's arguments."""

    def __init__(self, replies: list[Completion]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def complete(self, system, messages, tools=(), *,
                 allow_tool_calls=True):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "allow_tool_calls": allow_tool_calls})
        return self.replies.pop(0)

    def _build_request(self, *args):   # pragma: no cover - unused
        raise NotImplementedError

    def _parse_response(self, raw):    # pragma: no cover - unused
        raise NotImplementedError


def reply(text="", calls=()):
    return Completion(text=text, tool_calls=list(calls),
                      usage={"input": 10, "output": 2})


def call(call_id, name="roll_dice", **args):
    return {"id": call_id, "name": name, "args": args}


class RecordingTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.request: dict[str, Any] = {}

    def __call__(self, url, body, headers, timeout, max_attempts):
        self.request = {"url": url, "body": body, "headers": headers}
        return self.response


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


class LoopTests(unittest.TestCase):
    def test_dice_transcript_has_four_messages_in_order(self):
        provider = ScriptedProvider([
            reply("Rolling.", [call("c1", count=3)]),
            reply("Total 15, which beats 10."),
        ])
        messages = [{"role": "user", "text": "Roll 3 dice"}]
        answer = run_loop(provider, "sys", messages, {"roll_dice": Dice()})

        self.assertEqual(answer, "Total 15, which beats 10.")
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[2]["tool_call_id"], "c1")
        self.assertFalse(messages[2]["is_error"])

    def test_every_parallel_call_is_answered_with_its_own_id(self):
        provider = ScriptedProvider([
            reply(calls=[call("a", count=1), call("b", count=2)]),
            reply("done"),
        ])
        messages = [{"role": "user", "text": "two rolls"}]
        run_loop(provider, "", messages, {"roll_dice": Dice()})
        self.assertEqual([m.get("tool_call_id") for m in messages[2:4]],
                         ["a", "b"])

    def test_tool_failures_become_results_not_crashes(self):
        provider = ScriptedProvider([
            reply(calls=[call("x", name="broken"),
                         call("y", name="missing"),
                         {**call("z"), "error": "invalid tool arguments"}]),
            reply("recovered"),
        ])
        messages = [{"role": "user", "text": "go"}]
        tools = {"broken": Broken(), "roll_dice": Dice()}
        self.assertEqual(run_loop(provider, "", messages, tools),
                         "recovered")
        results = [m for m in messages if m["role"] == "tool"]
        self.assertTrue(all(m["is_error"] for m in results))
        self.assertIn("OSError: disk on fire", results[0]["text"])
        self.assertIn("unknown tool 'missing'", results[1]["text"])
        self.assertIn("invalid tool arguments", results[2]["text"])

    def test_blocked_call_is_still_answered(self):
        provider = ScriptedProvider([reply(calls=[call("c1", count=1)]),
                                     reply("ok")])
        messages = [{"role": "user", "text": "roll"}]
        run_loop(provider, "", messages, {"roll_dice": Dice()},
                 before_tool=lambda c: "not approved")
        self.assertEqual(messages[2]["text"], "BLOCKED: not approved")

    def test_turn_limit_ends_with_a_tool_free_report(self):
        provider = ScriptedProvider([
            reply(calls=[call("c1", count=1)]),
            reply("Status: rolled once.", [call("late", count=1)]),
        ])
        messages = [{"role": "user", "text": "roll forever"}]
        answer = run_loop(provider, "", messages, {"roll_dice": Dice()},
                          max_turns=1)
        self.assertEqual(answer, "Status: rolled once.")
        self.assertFalse(provider.calls[-1]["allow_tool_calls"])
        self.assertEqual(messages[-1]["tool_calls"], [])  # no dangling call

    def test_before_turn_can_rewrite_history(self):
        provider = ScriptedProvider([reply("hi")])
        messages = [{"role": "user", "text": "old"},
                    {"role": "user", "text": "new"}]
        run_loop(provider, "", messages, {},
                 before_turn=lambda msgs: msgs[-1:])
        self.assertEqual(provider.calls[0]["messages"],
                         [{"role": "user", "text": "new"}])


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class TruncationTests(unittest.TestCase):
    """A reply cut off at the output limit is reported, never silent."""

    def test_each_vendor_signal_sets_truncated(self):
        cases = [
            (AnthropicProvider, AnthropicConfig,
             {"content": [{"type": "text", "text": "half"}],
              "stop_reason": "max_tokens"}),
            (GeminiProvider, GeminiConfig,
             {"candidates": [{"content": {"parts": [{"text": "half"}]},
                              "finishReason": "MAX_TOKENS"}]}),
            (OpenRouterProvider, OpenRouterConfig,
             {"choices": [{"message": {"content": "half"},
                           "finish_reason": "length"}]}),
        ]
        for provider_class, config_class, raw in cases:
            with self.subTest(provider=provider_class.__name__):
                provider = provider_class(
                    config_class(api_key="k", model="m", base_url="u"),
                    transport=RecordingTransport(raw))
                self.assertTrue(provider.complete("", [])["truncated"])
                raw_done = dict(raw, stop_reason="end_turn")
                if "candidates" in raw:
                    raw_done = {"candidates": [dict(raw["candidates"][0],
                                                    finishReason="STOP")]}
                if "choices" in raw:
                    raw_done = {"choices": [dict(raw["choices"][0],
                                                 finish_reason="stop")]}
                provider = provider_class(
                    config_class(api_key="k", model="m", base_url="u"),
                    transport=RecordingTransport(raw_done))
                self.assertFalse(provider.complete("", [])["truncated"])

    def test_loop_warns_and_marks_the_answer(self):
        cut = reply("The page ends here <di")
        cut["truncated"] = True
        provider = ScriptedProvider([cut])
        events = []
        messages = [{"role": "user", "text": "build"}]
        answer = run_loop(provider, "", messages, {},
                          on_event=lambda k, p: events.append(k))
        self.assertIn("truncated", events)
        self.assertTrue(answer.endswith(
            "cut off at the model's output-token limit; it may be "
            "incomplete.]"))
        self.assertEqual(messages[-1]["text"], "The page ends here <di")


class ConfigTests(unittest.TestCase):
    def test_vendor_key_wins_over_generic_key(self):
        env = {"GEMINI_API_KEY": "vendor", "ODYSSEUS_API_KEY": "generic"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(GeminiConfig.from_env().api_key, "vendor")

    def test_override_wins_and_base_url_is_normalised(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"},
                             clear=True):
            config = GeminiConfig.from_env(base_url="http://local/v1/")
        self.assertEqual(config.base_url, "http://local/v1")

    def test_missing_key_or_model_is_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ConfigError):
                GeminiConfig.from_env()
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"},
                             clear=True):
            with self.assertRaises(ConfigError):   # OpenAI has no default
                OpenAIConfig.from_env()

    def test_factory_picks_vendor_from_environment(self):
        env = {"ODYSSEUS_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(create_provider(), AnthropicProvider)
        with self.assertRaises(ConfigError):
            create_provider("no-such-vendor")

    def test_no_provider_selected_is_an_error_not_a_default(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"},
                             clear=True):
            with self.assertRaises(ConfigError):
                create_provider()

    def test_defaults_come_from_the_packaged_toml(self):
        defaults = base.load_vendor_defaults("gemini")
        self.assertEqual(defaults.api_key_env,
                         ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"},
                             clear=True):
            config = GeminiConfig.from_env()
        self.assertEqual(config.model, defaults.model)
        self.assertEqual(config.base_url, defaults.base_url)

    def test_user_toml_overrides_packaged_defaults_key_by_key(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml",
                                         delete=False) as handle:
            handle.write('[gemini]\nmodel = "my-model"\n'
                         'api_key_env = ["MY_KEY"]\n')
        self.addCleanup(os.unlink, handle.name)
        env = {"ODYSSEUS_CONFIG": handle.name, "MY_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            config = GeminiConfig.from_env()
        self.assertEqual(config.model, "my-model")
        self.assertEqual(config.api_key, "k")
        self.assertEqual(config.base_url,   # untouched key: packaged value
                         base.load_vendor_defaults("gemini").base_url)

    def test_unreadable_user_toml_is_a_clear_error(self):
        env = {"ODYSSEUS_CONFIG": "/no/such/file.toml",
               "GEMINI_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ConfigError):
                GeminiConfig.from_env()

    def _user_toml(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".toml",
                                         delete=False) as handle:
            handle.write(text)
        self.addCleanup(os.unlink, handle.name)
        return {"ODYSSEUS_CONFIG": handle.name, "GEMINI_API_KEY": "k"}

    def test_invalid_user_toml_is_a_clear_error(self):
        env = self._user_toml("[gemini]\nmodel = my-model\n")  # no quotes
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ConfigError, "not valid TOML"):
                GeminiConfig.from_env()

    def test_wrongly_typed_setting_is_a_clear_error(self):
        env = self._user_toml("[gemini]\nmodel = 5\n")
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ConfigError, "must be a string"):
                GeminiConfig.from_env()

    def test_user_toml_may_add_a_vendor_table(self):
        env = self._user_toml('[mine]\nbase_url = "u"\nmodel = "m"\n'
                              'api_key_env = "MINE_KEY"\n')
        with mock.patch.dict(os.environ, env, clear=True):
            defaults = base.load_vendor_defaults("mine")
        self.assertEqual(defaults.api_key_env, ("MINE_KEY",))
        with self.assertRaises(ConfigError):
            base.load_vendor_defaults("nobody")

    def test_provider_rejects_the_wrong_config_class(self):
        with self.assertRaises(ConfigError):
            GeminiProvider(AnthropicConfig(api_key="k", model="m",
                                           base_url="u"))


# --------------------------------------------------------------------------
# Wire formats: one conversation, three vendors
# --------------------------------------------------------------------------

CONVERSATION = [
    {"role": "user", "text": "Roll twice"},
    {"role": "assistant", "text": "",
     "tool_calls": [{"id": "t1", "name": "roll_dice", "args": {"count": 1},
                     "signature": "sig-1"},
                    {"id": "t2", "name": "roll_dice", "args": {"count": 2}}]},
    {"role": "tool", "tool_call_id": "t1", "name": "roll_dice",
     "text": "[3]", "is_error": False},
    {"role": "tool", "tool_call_id": "t2", "name": "roll_dice",
     "text": "[1, 6]", "is_error": False},
    {"role": "user", "text": "Wrap up"},
]


class GeminiWireTests(unittest.TestCase):
    def setUp(self):
        self.transport = RecordingTransport({
            "candidates": [{"content": {"parts": [
                {"text": "thinking...", "thought": True},
                {"text": "Rolling."},
                {"functionCall": {"name": "roll_dice", "args": {"count": 3}},
                 "thoughtSignature": "sig-9"}]}}],
            "usageMetadata": {"promptTokenCount": 50,
                              "candidatesTokenCount": 7}})
        config = GeminiConfig(api_key="secret", model="gemini-test",
                              base_url="https://g.example/v1beta")
        self.provider = GeminiProvider(config, transport=self.transport)

    def test_request_shape(self):
        self.provider.complete("be brief", CONVERSATION, [DICE_SCHEMA],
                               allow_tool_calls=False)
        request = self.transport.request
        body = request["body"]
        self.assertNotIn("secret", request["url"])
        self.assertEqual(request["headers"]["x-goog-api-key"], "secret")
        self.assertNotIn("generationConfig", body)      # vendor defaults
        self.assertEqual(body["toolConfig"]["functionCallingConfig"]["mode"],
                         "NONE")
        contents = body["contents"]
        self.assertEqual([c["role"] for c in contents],
                         ["user", "model", "user"])
        model_parts = contents[1]["parts"]
        self.assertEqual(model_parts[0]["thoughtSignature"], "sig-1")
        # Both function responses and the follow-up text share one turn.
        self.assertEqual(len(contents[2]["parts"]), 3)

    def test_response_parsing(self):
        result = self.provider.complete("", CONVERSATION[:1])
        self.assertEqual(result["text"], "Rolling.")
        tool_call = result["tool_calls"][0]
        self.assertEqual(tool_call["signature"], "sig-9")
        self.assertTrue(tool_call["id"].startswith(base.GENERATED_ID_PREFIX))
        self.assertEqual(result["usage"], {"input": 50, "output": 7})

    def test_generated_ids_are_not_sent_back(self):
        result = self.provider.complete("", CONVERSATION[:1])
        history = [CONVERSATION[0],
                   {"role": "assistant", "text": "",
                    "tool_calls": result["tool_calls"]},
                   {"role": "tool", "name": "roll_dice", "text": "[2]",
                    "tool_call_id": result["tool_calls"][0]["id"]}]
        self.provider.complete("", history)
        contents = self.transport.request["body"]["contents"]
        self.assertNotIn("id", contents[1]["parts"][0]["functionCall"])
        self.assertNotIn("id", contents[2]["parts"][0]["functionResponse"])


class OpenAIWireTests(unittest.TestCase):
    def setUp(self):
        self.transport = RecordingTransport({
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "roll_dice",
                              "arguments": '{"count": 2}'}},
                {"id": "call_2", "type": "function",
                 "function": {"name": "roll_dice", "arguments": "{bad"}}]}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 9}})
        config = OpenRouterConfig(api_key="secret", model="vendor/model",
                                  base_url="https://or.example/api/v1")
        self.provider = OpenRouterProvider(config, transport=self.transport)

    def test_request_shape(self):
        self.provider.complete("be brief", CONVERSATION, [DICE_SCHEMA],
                               allow_tool_calls=False)
        request = self.transport.request
        body = request["body"]
        self.assertEqual(request["url"],
                         "https://or.example/api/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"],
                         "Bearer secret")
        self.assertEqual(request["headers"]["X-Title"], "Odysseus")
        self.assertEqual(body["tool_choice"], "none")
        wire = body["messages"]
        self.assertEqual([m["role"] for m in wire],
                         ["system", "user", "assistant", "tool", "tool",
                          "user"])
        self.assertIsNone(wire[2]["content"])
        self.assertEqual([m["tool_call_id"] for m in wire[3:5]],
                         ["t1", "t2"])

    def test_response_parsing_with_bad_arguments(self):
        result = self.provider.complete("", CONVERSATION[:1])
        first, second = result["tool_calls"]
        self.assertEqual(first["args"], {"count": 2})
        self.assertEqual(second["args"], {})
        self.assertIn("invalid tool arguments", second["error"])
        self.assertEqual(result["text"], "")

    def test_max_tokens_field_differs_by_vendor(self):
        openai = OpenAIConfig(api_key="k", model="m", base_url="u",
                              max_output_tokens=100)
        self.assertEqual(openai.max_tokens_field, "max_completion_tokens")
        self.assertEqual(OpenRouterConfig.max_tokens_field, "max_tokens")


class AnthropicWireTests(unittest.TestCase):
    def setUp(self):
        self.transport = RecordingTransport({
            "content": [{"type": "text", "text": "Rolling."},
                        {"type": "tool_use", "id": "toolu_1",
                         "name": "roll_dice", "input": {"count": 4}}],
            "usage": {"input_tokens": 30, "output_tokens": 5}})
        config = AnthropicConfig(api_key="secret", model="claude-test",
                                 base_url="https://a.example/v1")
        self.provider = AnthropicProvider(config, transport=self.transport)

    def test_request_shape(self):
        self.provider.complete("be brief", CONVERSATION, [DICE_SCHEMA],
                               allow_tool_calls=False)
        request = self.transport.request
        body = request["body"]
        self.assertEqual(request["headers"]["x-api-key"], "secret")
        self.assertEqual(body["system"], "be brief")
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(body["tool_choice"], {"type": "none"})
        self.assertIn("input_schema", body["tools"][0])
        wire = body["messages"]
        self.assertEqual([m["role"] for m in wire],
                         ["user", "assistant", "user"])
        self.assertEqual([b["type"] for b in wire[1]["content"]],
                         ["tool_use", "tool_use"])
        # Both results, then the text, in one user message.
        self.assertEqual([b["type"] for b in wire[2]["content"]],
                         ["tool_result", "tool_result", "text"])
        self.assertEqual(wire[2]["content"][1]["tool_use_id"], "t2")

    def test_response_parsing(self):
        result = self.provider.complete("", CONVERSATION[:1])
        self.assertEqual(result["text"], "Rolling.")
        self.assertEqual(result["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(result["usage"], {"input": 30, "output": 5})


# --------------------------------------------------------------------------
# HTTP transport
# --------------------------------------------------------------------------


class TransportTests(unittest.TestCase):
    @staticmethod
    def _http_error(code, retry_after=None):
        headers = {"Retry-After": retry_after} if retry_after else {}
        return urllib.error.HTTPError("u", code, "err", headers,
                                      io.BytesIO(b"details"))

    def test_rate_limit_is_retried_honouring_retry_after(self):
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b'{"ok": true}'
        responses = [self._http_error(429, "3"), ok]

        def fake_urlopen(request, timeout):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(base.urllib.request, "urlopen",
                               fake_urlopen), \
                mock.patch.object(base.time, "sleep") as sleep:
            result = base.post_json("http://x.test", {}, {}, 1, 3)
        self.assertEqual(result, {"ok": True})
        sleep.assert_called_once_with(3.0)

    def test_client_errors_fail_fast(self):
        def fake_urlopen(request, timeout):
            raise self._http_error(400)

        with mock.patch.object(base.urllib.request, "urlopen",
                               fake_urlopen), \
                mock.patch.object(base.time, "sleep") as sleep:
            with self.assertRaises(ProviderError) as caught:
                base.post_json("http://x.test", {}, {}, 1, 5)
        self.assertEqual(caught.exception.status, 400)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
