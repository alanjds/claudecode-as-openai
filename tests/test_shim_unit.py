"""Offline unit tests: mock the `claude` CLI subprocess so parsing/retry
logic is verifiable in CI without a live `claude` binary or subscription.

Why mock the subprocess and not use VCR-style HTTP recording: this shim
never makes the Anthropic HTTP call itself -- it spawns `claude` (a
separate compiled binary) as a subprocess, and THAT process makes the real
HTTP call, invisibly to us. VCR/responses-style libraries only intercept
HTTP made by the process under test; they have no hook into a child
process's network traffic. The actual, mockable seam this shim crosses is
"a subprocess that writes NDJSON lines to stdout", so that's what's
mocked here (subprocess.Popen), matching real output shapes captured
during live manual testing (see README.md).

Run: python3 -m unittest tests.test_shim_unit -v
     (or: python3 tests/test_shim_unit.py)
"""
import json
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claudecode_as_openai import shim  # noqa: E402


def _ndjson_lines(*dicts):
    """Build the list of raw NDJSON lines a real `claude --output-format
    stream-json` process would write to stdout for the given chunks."""
    return [json.dumps(d) + "\n" for d in dicts]


class FakeProcess:
    """Stand-in for subprocess.Popen(...) that replays canned NDJSON lines
    on .stdout and records what was written to .stdin, matching the real
    claude CLI's behaviour closely enough for call_claude_streaming()."""

    def __init__(self, lines, stderr_text=""):
        self.stdin = MagicMock()
        self.stdout = iter(lines)
        self.stderr = MagicMock()
        self.stderr.read.return_value = stderr_text
        self._terminated = False

    def terminate(self):
        self._terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self._terminated = True


class TestMessageBuilding(unittest.TestCase):
    """Pure functions, no subprocess involved."""

    def test_system_message_becomes_system_prompt(self):
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
        claude_messages, system_prompt = shim.build_claude_messages(messages)
        self.assertEqual(system_prompt, "be terse")
        self.assertEqual(claude_messages, [{"role": "user", "content": "hi"}])

    def test_tool_result_becomes_tool_result_block(self):
        messages = [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "72F"},
        ]
        claude_messages, _ = shim.build_claude_messages(messages)
        self.assertEqual(
            claude_messages,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}
                    ],
                }
            ],
        )

    def test_assistant_tool_calls_become_tool_use_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_2",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            }
        ]
        claude_messages, _ = shim.build_claude_messages(messages)
        self.assertEqual(len(claude_messages), 1)
        blocks = claude_messages[0]["content"]
        self.assertEqual(blocks[0]["type"], "tool_use")
        self.assertEqual(blocks[0]["name"], "get_weather")
        self.assertEqual(blocks[0]["input"], {"city": "Paris"})

    def test_tools_rendered_into_system_prompt_with_strong_framing(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        prompt = shim.render_tools_into_system_prompt(tools, "base prompt")
        self.assertIn("base prompt", prompt)
        self.assertIn("get_weather", prompt)
        # Regression guard: weak/hedgy framing ("custom tools available")
        # measured 0/8 real tool_use dispatch in manual testing; the
        # strong framing below measured 6/6 -- don't regress the wording.
        self.assertIn("ARE implemented", prompt)
        self.assertNotIn("custom tools available", prompt)


class TestCallClaudeStreaming(unittest.TestCase):
    """Exercises call_claude_streaming() against canned NDJSON matching
    real observed shapes from live manual testing."""

    def _run(self, fake_proc):
        with patch.object(shim.subprocess, "Popen", return_value=fake_proc):
            return shim.call_claude_streaming([{"role": "user", "content": "hi"}], "", "sonnet")

    def test_immediate_text_message(self):
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "hello there"}],
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                },
            },
            {"type": "result", "subtype": "success"},
        )
        text, tool_calls, usage = self._run(FakeProcess(lines))
        self.assertEqual(text, "hello there")
        self.assertEqual(tool_calls, [])
        self.assertEqual(usage["input_tokens"], 3)

    def test_thinking_block_then_tool_use_is_not_missed(self):
        """Regression test for the bug where an assistant message
        consisting of ONLY a `thinking` block made the reader stop early
        and miss the real tool_use on the next NDJSON line."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "thinking", "thinking": ""}]},
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 22},
                },
            },
            {"type": "result", "subtype": "error_max_turns", "is_error": True},
        )
        text, tool_calls, usage = self._run(FakeProcess(lines))
        self.assertIsNone(text)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(tool_calls[0]["input"], {"city": "Paris"})

    def test_narration_instead_of_dispatch_returns_text_not_tool_call(self):
        """The documented flakiness mode: Claude describes the action in
        text instead of emitting a tool_use block."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Sure! Let me check the weather in Paris for you right now."}],
                    "usage": {"input_tokens": 3, "output_tokens": 15},
                },
            },
            {"type": "result", "subtype": "success"},
        )
        text, tool_calls, usage = self._run(FakeProcess(lines))
        self.assertIn("Let me check", text)
        self.assertEqual(tool_calls, [])

    def test_error_message_raises(self):
        lines = _ndjson_lines(
            {"type": "assistant", "message": {"error": "overloaded_error"}},
        )
        with self.assertRaises(RuntimeError):
            self._run(FakeProcess(lines))


class TestToolRetryWrapper(unittest.TestCase):
    """call_claude_with_tool_retry: verifies retry count, backoff timing,
    and that it stops as soon as a real tool_use shows up."""

    def test_no_retry_needed_when_tool_call_on_first_try(self):
        with patch.object(shim, "call_claude_streaming", return_value=(None, [{"name": "x"}], {})) as mock_call, \
             patch.object(shim.time, "sleep") as mock_sleep:
            text, tool_calls, usage = shim.call_claude_with_tool_retry([], "", "sonnet", tools_requested=True)
        self.assertEqual(mock_call.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(tool_calls, [{"name": "x"}])

    def test_no_retry_when_no_tools_requested(self):
        with patch.object(shim, "call_claude_streaming", return_value=("just text", [], {})) as mock_call, \
             patch.object(shim.time, "sleep") as mock_sleep:
            text, tool_calls, usage = shim.call_claude_with_tool_retry([], "", "sonnet", tools_requested=False)
        self.assertEqual(mock_call.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(text, "just text")

    def test_retries_with_backoff_until_tool_call_appears(self):
        # Fails twice (narration), succeeds on the 3rd attempt.
        side_effects = [
            ("narrating...", [], {}),
            ("narrating again...", [], {}),
            (None, [{"name": "get_weather"}], {}),
        ]
        with patch.object(shim, "call_claude_streaming", side_effect=side_effects) as mock_call, \
             patch.object(shim.time, "sleep") as mock_sleep:
            text, tool_calls, usage = shim.call_claude_with_tool_retry([], "", "sonnet", tools_requested=True)
        self.assertEqual(mock_call.call_count, 3)
        self.assertEqual(tool_calls, [{"name": "get_weather"}])
        # Backoff schedule: 2s, then 4s (exponential, base 2.0, doubling).
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)

    def test_gives_up_after_total_attempts_exhausted(self):
        total_attempts = 1 + shim.TOOL_CALL_MAX_RETRIES
        side_effects = [("still narrating", [], {})] * total_attempts
        with patch.object(shim, "call_claude_streaming", side_effect=side_effects) as mock_call, \
             patch.object(shim.time, "sleep"):
            text, tool_calls, usage = shim.call_claude_with_tool_retry([], "", "sonnet", tools_requested=True)
        self.assertEqual(mock_call.call_count, total_attempts)
        self.assertEqual(tool_calls, [])
        self.assertEqual(text, "still narrating")

    def test_backoff_schedule_matches_cline_derived_formula(self):
        self.assertEqual(shim._backoff_delay_s(0), 2.0)
        self.assertEqual(shim._backoff_delay_s(1), 4.0)
        self.assertEqual(shim._backoff_delay_s(2), 8.0)
        self.assertEqual(shim._backoff_delay_s(3), 15.0)  # capped
        self.assertEqual(shim._backoff_delay_s(10), 15.0)  # stays capped


if __name__ == "__main__":
    unittest.main()
