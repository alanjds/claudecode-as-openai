"""Offline unit tests for claudecode_as_openai/tracking.py: the pluggable
usage-tracker registry, the redaction helper, and the OTEL/Logfire span
no-op contract. litellm/logfire are NOT installed in this test
environment by design -- that exercises the real no-op import-failure
path, not a mocked stand-in for it; the mocked-success cases inject a
fake module via sys.modules to cover the "dependency present" path
without actually requiring it installed.

Run: python3 -m unittest tests.test_tracking -v
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claudecode_as_openai import tracking  # noqa: E402


class TestEmitUsage(unittest.TestCase):
    """register_tracker/emit_usage: the plugin registry itself."""

    def setUp(self):
        self._saved_trackers = list(tracking._TRACKERS)
        tracking._TRACKERS.clear()

    def tearDown(self):
        tracking._TRACKERS[:] = self._saved_trackers

    def test_calls_every_registered_tracker(self):
        calls = []

        class RecordingTracker(tracking.UsageTracker):
            def record(self, event):
                calls.append(event)

        tracking.register_tracker(RecordingTracker())
        tracking.register_tracker(RecordingTracker())
        event = {"model": "sonnet", "usage": {}}
        tracking.emit_usage(event)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0], event)

    def test_swallows_a_raising_tracker_and_still_calls_the_rest(self):
        calls = []

        class BrokenTracker(tracking.UsageTracker):
            def record(self, event):
                raise RuntimeError("boom")

        class RecordingTracker(tracking.UsageTracker):
            def record(self, event):
                calls.append(event)

        tracking.register_tracker(BrokenTracker())
        tracking.register_tracker(RecordingTracker())
        tracking.emit_usage({"model": "sonnet", "usage": {}})
        self.assertEqual(len(calls), 1)

    def test_trackers_run_in_registration_order_and_share_mutations(self):
        seen = []

        class AttachesCost(tracking.UsageTracker):
            def record(self, event):
                event["cost_usd"] = 0.01

        class ReadsCost(tracking.UsageTracker):
            def record(self, event):
                seen.append(event.get("cost_usd"))

        tracking.register_tracker(AttachesCost())
        tracking.register_tracker(ReadsCost())
        tracking.emit_usage({"model": "sonnet", "usage": {}})
        self.assertEqual(seen, [0.01])


class TestJsonlFileTracker(unittest.TestCase):
    def test_appends_one_valid_json_line_per_event(self):
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            t = tracking.JsonlFileTracker(path)
            t.record({"model": "sonnet", "usage": {"prompt_tokens": 1}})
            t.record({"model": "opus", "usage": {"prompt_tokens": 2}})
            with open(path) as f:
                lines = [line for line in f.read().splitlines() if line]
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            second = json.loads(lines[1])
            self.assertEqual(first["model"], "sonnet")
            self.assertEqual(second["model"], "opus")
        finally:
            Path(path).unlink(missing_ok=True)


class TestLiteLLMCostTracker(unittest.TestCase):
    """litellm is genuinely not installed in this venv -- the first test
    exercises that real absence. The second injects a fake module to
    cover the case where it IS installed, without requiring it here."""

    def test_noops_cleanly_when_litellm_not_installed(self):
        event = {"model": "sonnet", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        tracking.LiteLLMCostTracker().record(event)
        self.assertNotIn("cost_usd", event)

    def test_attaches_cost_when_litellm_available(self):
        fake_litellm = types.ModuleType("litellm")
        fake_litellm.completion_cost = lambda model, completion_response: 0.0042
        with patch.dict(sys.modules, {"litellm": fake_litellm}):
            event = {"model": "sonnet", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
            tracking.LiteLLMCostTracker().record(event)
        self.assertEqual(event["cost_usd"], 0.0042)


class TestRedactCmdForLog(unittest.TestCase):
    def test_redacts_system_prompt_value(self):
        cmd = ["claude", "--system-prompt", "sekrit instructions", "-p"]
        out = tracking.redact_cmd_for_log(cmd)
        self.assertNotIn("sekrit instructions", out)
        self.assertIn("<len=19>", out)

    def test_redacts_json_schema_value(self):
        cmd = ["claude", "--json-schema", '{"type": "object"}', "-p"]
        out = tracking.redact_cmd_for_log(cmd)
        self.assertNotIn('{"type": "object"}', out)
        self.assertIn("<len=", out)

    def test_redacts_mcp_config_value(self):
        cmd = ["claude", "--mcp-config", '{"mcpServers": {"shim_tools": {}}}', "-p"]
        out = tracking.redact_cmd_for_log(cmd)
        self.assertNotIn("mcpServers", out)
        self.assertIn("<len=", out)

    def test_redacts_disallowed_tools_value(self):
        cmd = ["claude", "--disallowedTools", "Task,Bash,Read,Write", "-p"]
        out = tracking.redact_cmd_for_log(cmd)
        self.assertNotIn("Task,Bash,Read,Write", out)
        self.assertIn("<len=20>", out)

    def test_redacts_allowed_tools_value(self):
        cmd = ["claude", "--allowedTools", "mcp__shim_tools__get_weather", "-p"]
        out = tracking.redact_cmd_for_log(cmd)
        self.assertNotIn("mcp__shim_tools__get_weather", out)
        self.assertIn("<len=", out)

    def test_leaves_system_prompt_file_path_untouched(self):
        cmd = ["claude", "--system-prompt-file", "/tmp/sysprompt.txt", "-p"]
        out = tracking.redact_cmd_for_log(cmd)
        self.assertIn("/tmp/sysprompt.txt", out)

    def test_leaves_unrelated_args_untouched(self):
        cmd = ["claude", "--model", "sonnet", "--resume", "abc-123"]
        out = tracking.redact_cmd_for_log(cmd)
        self.assertIn("sonnet", out)
        self.assertIn("abc-123", out)

    def test_does_not_mutate_input_list(self):
        cmd = ["claude", "--system-prompt", "secret", "-p"]
        original = list(cmd)
        tracking.redact_cmd_for_log(cmd)
        self.assertEqual(cmd, original)


class TestSpan(unittest.TestCase):
    """logfire is genuinely not installed in this test environment --
    tracing stays disabled regardless, so span() must always no-op here.
    Separately verifies the no-op path directly by forcing the disabled
    state, and the mocked-enabled path via a fake logfire module."""

    def setUp(self):
        self._saved_enabled = tracking._tracing_enabled
        tracking._tracing_enabled = False

    def tearDown(self):
        tracking._tracing_enabled = self._saved_enabled

    def test_noops_when_tracing_disabled(self):
        with tracking.span("test_span", foo="bar") as sp:
            sp.set_attribute("anything", 1)  # must not raise

    def test_configure_tracing_from_env_noop_without_flag(self):
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("CLAUDE_OPENAI_TRACING", None)
            tracking.configure_tracing_from_env()
        self.assertFalse(tracking._tracing_enabled)

    def test_wrapped_exception_propagates_through_noop_span(self):
        with self.assertRaises(ValueError):
            with tracking.span("test_span"):
                raise ValueError("real error")

    def test_span_enabled_with_fake_logfire_sets_attributes(self):
        recorded = {}

        class FakeSpanCM:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def set_attribute(self, key, value):
                recorded[key] = value

        fake_logfire = types.ModuleType("logfire")
        fake_logfire.span = lambda name, **attrs: FakeSpanCM()
        tracking._tracing_enabled = True
        with patch.dict(sys.modules, {"logfire": fake_logfire}):
            with tracking.span("test_span") as sp:
                sp.set_attribute("gen_ai.usage.input_tokens", 42)
        self.assertEqual(recorded["gen_ai.usage.input_tokens"], 42)

    def test_span_swallows_logfire_failure_and_still_runs_body(self):
        fake_logfire = types.ModuleType("logfire")

        def broken_span(name, **attrs):
            raise RuntimeError("logfire misconfigured")

        fake_logfire.span = broken_span
        tracking._tracing_enabled = True
        ran = []
        with patch.dict(sys.modules, {"logfire": fake_logfire}):
            with tracking.span("test_span") as sp:
                ran.append(True)
                sp.set_attribute("x", 1)  # must not raise even on the noop fallback
        self.assertEqual(ran, [True])


if __name__ == "__main__":
    unittest.main()
