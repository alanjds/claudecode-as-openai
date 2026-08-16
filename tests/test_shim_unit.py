"""Offline unit tests: mock the `claude` CLI subprocess so parsing/retry/
session-caching/error-translation logic is verifiable in CI without a live
`claude` binary or subscription.

Why mock the subprocess and not use VCR-style HTTP recording: this shim
never makes the Anthropic HTTP call itself -- it spawns `claude` (a
separate compiled binary) as a subprocess, and THAT process makes the real
HTTP call, invisibly to us. VCR/responses-style libraries only intercept
HTTP made by the process under test; they have no hook into a child
process's network traffic. The actual, mockable seam this shim crosses is
"a subprocess that writes NDJSON lines to stdout", so that's what's
mocked here (subprocess.Popen), matching real output shapes captured
during live manual testing (see README.md "Capability audit").

Run: python3 -m unittest tests.test_shim_unit -v
     (or: python3 tests/test_shim_unit.py)
"""
import json
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claudecode_as_openai import shim  # noqa: E402


def _ndjson_lines(*dicts):
    return [json.dumps(d) + "\n" for d in dicts]


def _parse_ndjson_lines(raw_lines):
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


class FakeProcess:
    """Stand-in for subprocess.Popen(...) that replays canned NDJSON lines
    on .stdout, matching the real claude CLI's behaviour closely enough
    for call_claude_streaming()."""

    def __init__(self, lines, stderr_text=""):
        self.stdin = MagicMock()
        self.stdout = iter(lines)
        self.stderr = MagicMock()
        self.stderr.read.return_value = stderr_text

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class TestMessageBuilding(unittest.TestCase):
    def test_system_messages_extracted_separately(self):
        messages = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
        self.assertEqual(shim.system_prompt_from_messages(messages), "be terse")
        self.assertEqual(shim.build_claude_messages(messages), [{"role": "user", "content": "hi"}])

    def test_tool_result_becomes_tool_result_block(self):
        messages = [{"role": "tool", "tool_call_id": "toolu_1", "content": "72F"}]
        claude_messages = shim.build_claude_messages(messages)
        self.assertEqual(
            claude_messages,
            [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}]}],
        )

    def test_assistant_tool_calls_become_tool_use_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "toolu_2", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
            }
        ]
        claude_messages = shim.build_claude_messages(messages)
        blocks = claude_messages[0]["content"]
        self.assertEqual(blocks[0]["type"], "tool_use")
        self.assertEqual(blocks[0]["input"], {"city": "Paris"})

    def test_tools_rendered_with_strong_framing_not_weak(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "get weather", "parameters": {}}}]
        prompt = shim.render_tools_into_system_prompt(tools, "base prompt")
        self.assertIn("get_weather", prompt)
        # Regression guard: weak/hedgy framing measured 0/8 real tool_use
        # dispatch in manual testing; strong framing measured 6/6.
        self.assertIn("ARE implemented", prompt)
        self.assertNotIn("custom tools available", prompt)


class TestSessionCaching(unittest.TestCase):
    """resolve_session/record_session: fingerprint-based conversation
    tracking so a continuing conversation resumes with only the delta
    instead of resending full history."""

    def setUp(self):
        shim._SESSION_STORE.clear()

    def test_new_conversation_is_fresh(self):
        messages = [{"role": "user", "content": "hello"}]
        mode, session_id, delta, key = shim.resolve_session(messages)
        self.assertEqual(mode, "fresh")
        self.assertEqual(delta, messages)

    def test_continuing_conversation_resumes_with_delta_only(self):
        turn1 = [{"role": "user", "content": "remember X"}]
        mode, session_id, delta, key = shim.resolve_session(turn1)
        assistant_reply = {"role": "assistant", "content": "ok, remembered X"}
        shim.record_session(key, session_id, turn1, assistant_reply)

        turn2 = turn1 + [assistant_reply, {"role": "user", "content": "what was X?"}]
        mode2, session_id2, delta2, key2 = shim.resolve_session(turn2)

        self.assertEqual(mode2, "resume")
        self.assertEqual(session_id2, session_id)
        self.assertEqual(key2, key)
        # Only the NEW trailing message should be in the delta, not the
        # full history -- this is the entire point of session caching.
        self.assertEqual(delta2, [{"role": "user", "content": "what was X?"}])

    def test_diverged_history_falls_back_to_fresh(self):
        turn1 = [{"role": "user", "content": "remember X"}]
        mode, session_id, delta, key = shim.resolve_session(turn1)
        assistant_reply = {"role": "assistant", "content": "ok, remembered X"}
        shim.record_session(key, session_id, turn1, assistant_reply)

        # Client edited/regenerated turn1's reply -- history diverged from
        # what was synced. Must NOT try to resume (would desync the
        # Claude-side transcript); must fall back to fresh with full history.
        diverged_reply = {"role": "assistant", "content": "a completely different reply"}
        turn2 = turn1 + [diverged_reply, {"role": "user", "content": "what was X?"}]
        mode2, session_id2, delta2, key2 = shim.resolve_session(turn2)

        self.assertEqual(mode2, "fresh")
        self.assertEqual(delta2, turn2)

    def test_different_conversations_get_different_keys(self):
        conv_a = [{"role": "user", "content": "conversation A"}]
        conv_b = [{"role": "user", "content": "conversation B"}]
        _, _, _, key_a = shim.resolve_session(conv_a)
        _, _, _, key_b = shim.resolve_session(conv_b)
        self.assertNotEqual(key_a, key_b)

    def test_system_prompt_participates_in_conversation_key(self):
        base = [{"role": "user", "content": "hi"}]
        with_sys = [{"role": "system", "content": "be terse"}] + base
        _, _, _, key_plain = shim.resolve_session(base)
        _, _, _, key_sys = shim.resolve_session(with_sys)
        self.assertNotEqual(key_plain, key_sys)


class TestBuildClaudeCmd(unittest.TestCase):
    def test_tools_requested_uses_disallowed_tools_list(self):
        cmd = shim._build_claude_cmd("sonnet", "fresh", "sid", tools_requested=True, max_turns=1, json_schema=None)
        self.assertIn("--disallowedTools", cmd)
        self.assertNotIn("--strict-mcp-config", cmd)

    def test_no_tools_requested_locks_down_mcp_too(self):
        """Regression test for the MCP tool leakage finding: when no
        tools are requested, both built-ins AND locally-configured MCP
        servers must be blocked (--tools "" alone leaves MCP tools live)."""
        cmd = shim._build_claude_cmd("sonnet", "fresh", "sid", tools_requested=False, max_turns=1, json_schema=None)
        self.assertIn("--strict-mcp-config", cmd)
        self.assertIn('{"mcpServers":{}}', cmd)
        tools_idx = cmd.index("--tools")
        self.assertEqual(cmd[tools_idx + 1], "")

    def test_resume_mode_uses_resume_flag(self):
        cmd = shim._build_claude_cmd("sonnet", "resume", "abc-123", tools_requested=False, max_turns=1, json_schema=None)
        self.assertIn("--resume", cmd)
        self.assertEqual(cmd[cmd.index("--resume") + 1], "abc-123")
        self.assertNotIn("--session-id", cmd)

    def test_fresh_mode_uses_session_id_flag(self):
        cmd = shim._build_claude_cmd("sonnet", "fresh", "abc-123", tools_requested=False, max_turns=1, json_schema=None)
        self.assertIn("--session-id", cmd)
        self.assertNotIn("--resume", cmd)

    def test_json_schema_adds_flag(self):
        cmd = shim._build_claude_cmd("sonnet", "fresh", "sid", tools_requested=False, max_turns=3, json_schema={"type": "object"})
        self.assertIn("--json-schema", cmd)

    def test_setting_sources_always_excluded(self):
        """--setting-sources "" is always passed, regardless of tools/mode
        -- this is what actually prevents locally-configured hooks
        (SessionStart etc) from firing and injecting arbitrary extra
        context into every completion. Verified live (2026-08-14): a real
        SessionStart hook fired and injected a marker into model context
        without this flag; with it, the model confirmed no hook message
        was present. Does not affect OAuth (reads from a separate
        credentials file, not settings.json)."""
        for tools_requested in (True, False):
            cmd = shim._build_claude_cmd(
                "sonnet", "fresh", "sid", tools_requested=tools_requested, max_turns=1, json_schema=None
            )
            self.assertIn("--setting-sources", cmd)
            idx = cmd.index("--setting-sources")
            self.assertEqual(cmd[idx + 1], "")

    def test_mcp_tool_config_overrides_tools_requested_path(self):
        """When mcp_tool_config is given, the command uses --strict-mcp-config
        + --allowedTools scoped to the manifest server (not the bare
        --disallowedTools-only path used for the prose-fallback case), and
        --disallowedTools is still present to keep built-ins blocked."""
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        mcp_tool_config = shim.build_mcp_tool_config(tools)
        self.addCleanup(os.unlink, mcp_tool_config["manifest_path"])
        cmd = shim._build_claude_cmd(
            "sonnet", "fresh", "sid", tools_requested=True, max_turns=2, json_schema=None,
            mcp_tool_config=mcp_tool_config,
        )
        self.assertIn("--disallowedTools", cmd)
        self.assertIn("--strict-mcp-config", cmd)
        self.assertIn("--allowedTools", cmd)
        allowed_idx = cmd.index("--allowedTools")
        self.assertEqual(cmd[allowed_idx + 1], "mcp__shim_tools__get_weather")
        mcp_config_idx = cmd.index("--mcp-config")
        mcp_config = json.loads(cmd[mcp_config_idx + 1])
        self.assertIn("shim_tools", mcp_config["mcpServers"])


class TestMcpToolConfig(unittest.TestCase):
    """The native MCP tool registration path -- the real fix for the ~40%
    single-shot tool-call reliability problem (verified live 2026-08-14:
    8/8, then 5/5, then 5/5-through-the-real-shim turn-1 dispatch, vs
    ~40% for the prose-description fallback)."""

    def test_manifest_translates_openai_tools_shape(self):
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }}]
        manifest = shim.build_mcp_tool_manifest(tools)
        self.assertEqual(manifest, [{
            "name": "get_weather",
            "description": "get weather",
            "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }])

    def test_manifest_defaults_missing_description_and_parameters(self):
        tools = [{"type": "function", "function": {"name": "ping"}}]
        manifest = shim.build_mcp_tool_manifest(tools)
        self.assertEqual(manifest[0]["description"], "")
        self.assertEqual(manifest[0]["inputSchema"], {"type": "object", "properties": {}})

    def test_manifest_returns_none_for_empty_tools(self):
        self.assertIsNone(shim.build_mcp_tool_manifest([]))
        self.assertIsNone(shim.build_mcp_tool_manifest(None))

    def test_manifest_returns_none_for_invalid_tool_name(self):
        """A tool name that can't satisfy MCP's ^[a-zA-Z0-9_-]{1,64}$
        constraint (e.g. containing a space) must fall back to the prose
        path -- returning None here is what signals the caller to use
        render_tools_into_system_prompt() instead."""
        tools = [{"type": "function", "function": {"name": "get weather", "parameters": {}}}]
        self.assertIsNone(shim.build_mcp_tool_manifest(tools))

    def test_config_writes_manifest_file_and_builds_mcp_config(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        cfg = shim.build_mcp_tool_config(tools)
        self.addCleanup(os.unlink, cfg["manifest_path"])
        self.assertTrue(os.path.exists(cfg["manifest_path"]))
        with open(cfg["manifest_path"]) as f:
            written = json.load(f)
        self.assertEqual(written, [{"name": "get_weather", "description": "", "inputSchema": {"type": "object", "properties": {}}}])
        self.assertEqual(cfg["allowed_tools"], ["mcp__shim_tools__get_weather"])
        server = cfg["mcp_config"]["mcpServers"]["shim_tools"]
        self.assertEqual(server["args"][-1], cfg["manifest_path"])

    def test_config_returns_none_when_manifest_build_fails(self):
        tools = [{"type": "function", "function": {"name": "bad name", "parameters": {}}}]
        self.assertIsNone(shim.build_mcp_tool_config(tools))

    def test_mcp_server_command_is_stdio_not_tcp(self):
        """Explicit requirement: this must never bind a TCP port. The
        mcpServers entry only ever specifies a "command"/"args" pair (a
        child process wired via stdio), never a "url"/"port"/transport
        field of any kind."""
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        cfg = shim.build_mcp_tool_config(tools)
        self.addCleanup(os.unlink, cfg["manifest_path"])
        server = cfg["mcp_config"]["mcpServers"]["shim_tools"]
        self.assertIn("command", server)
        self.assertIn("args", server)
        self.assertNotIn("url", server)
        self.assertNotIn("port", server)

    def test_manifest_path_is_only_thing_that_varies_across_calls(self):
        """Same tool set built twice produces the identical mcp_config
        SERVER COMMAND (just a different manifest file path) -- this is
        what keeps Anthropic's prompt cache viable across a resumed
        session for a stable tool set (verified live: cache_creation
        stays flat across --resume'd turns, only busting when the actual
        tool set changes -- see README "MCP native tool registration")."""
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        cfg1 = shim.build_mcp_tool_config(tools)
        cfg2 = shim.build_mcp_tool_config(tools)
        self.addCleanup(os.unlink, cfg1["manifest_path"])
        self.addCleanup(os.unlink, cfg2["manifest_path"])
        server1 = cfg1["mcp_config"]["mcpServers"]["shim_tools"]
        server2 = cfg2["mcp_config"]["mcpServers"]["shim_tools"]
        self.assertEqual(server1["command"], server2["command"])
        self.assertEqual(server1["args"][0], server2["args"][0])  # server script path
        self.assertNotEqual(server1["args"][1], server2["args"][1])  # manifest path differs

    def test_strip_mcp_tool_prefix(self):
        self.assertEqual(shim.strip_mcp_tool_prefix("mcp__shim_tools__get_weather"), "get_weather")
        self.assertEqual(shim.strip_mcp_tool_prefix("get_weather"), "get_weather")
        self.assertEqual(shim.strip_mcp_tool_prefix(""), "")
        self.assertIsNone(shim.strip_mcp_tool_prefix(None))


class TestCallClaudeStreaming(unittest.TestCase):
    def _run(self, fake_proc, **kwargs):
        # call_claude_streaming opens a REAL PTY (via pty.openpty()) and
        # reads from its real fd whenever stream_callback is set (see
        # _iter_ndjson_lines_pty's docstring for why: claude's own stdout
        # is fully buffered on a plain pipe, so real client streaming
        # needs a real terminal attached). FakeProcess never actually
        # writes anything to that fd, so exercising that path here would
        # hang the test waiting on a fd nothing feeds. Redirect
        # _iter_ndjson_lines_pty to replay the same canned `lines` the
        # mocked Popen would have produced via a plain pipe, so PTY-path
        # tests get identical NDJSON content without touching a real fd.
        with patch.object(shim.subprocess, "Popen", return_value=fake_proc), \
             patch.object(shim.pty, "openpty", return_value=(-1, -1)), \
             patch.object(shim.os, "close", lambda fd: None), \
             patch.object(shim, "_iter_ndjson_lines_pty", lambda master_fd, proc, timeout_s: iter(_parse_ndjson_lines(fake_proc.stdout))):
            return shim.call_claude_streaming([{"role": "user", "content": "hi"}], "", "sonnet", **kwargs)

    def test_immediate_text_message(self):
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello there"}], "usage": {"input_tokens": 3, "output_tokens": 4}}},
            {"type": "result", "subtype": "success"},
        )
        result = self._run(FakeProcess(lines))
        self.assertEqual(result["text"], "hello there")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["usage"]["input_tokens"], 3)

    def test_thinking_block_then_tool_use_is_not_missed(self):
        """Regression test: an assistant message can consist of ONLY a
        thinking block before the real tool_use arrives on a later line."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": ""}]}},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}],
                    "usage": {"input_tokens": 1, "output_tokens": 22},
                },
            },
            {"type": "result", "subtype": "error_max_turns", "is_error": True},
        )
        result = self._run(FakeProcess(lines))
        self.assertIsNone(result["text"])
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "get_weather")

    def test_narration_instead_of_dispatch_returns_text_not_tool_call(self):
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Let me check the weather in Paris."}], "usage": {"input_tokens": 3, "output_tokens": 15}}},
            {"type": "result", "subtype": "success"},
        )
        result = self._run(FakeProcess(lines))
        self.assertIn("Let me check", result["text"])
        self.assertEqual(result["tool_calls"], [])

    def test_text_narration_THEN_tool_use_in_separate_messages_is_not_missed(self):
        """Regression test for a real bug found live (2026-08-13): Claude
        frequently narrates ("Sure! Let me check...") in one assistant
        NDJSON message, then dispatches the actual tool_use in a SEPARATE,
        LATER assistant message within the same turn. An earlier version
        of this function stopped reading on the first text-only message,
        silently swallowing the tool_use that followed -- confirmed 0/10
        in a live regression run before this fix, 10/10 after."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "..."}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Sure! Let me check the weather in Oslo for you right now!"}]}},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Oslo"}}],
                    "usage": {"input_tokens": 3, "output_tokens": 8},
                },
            },
            {"type": "result", "subtype": "error_max_turns", "is_error": True},
        )
        result = self._run(FakeProcess(lines))
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["name"], "get_weather")
        self.assertEqual(result["tool_calls"][0]["input"], {"city": "Oslo"})
        # The narration text is retained too (harmless, and useful for
        # debugging), but tool_calls being present is what matters for
        # finish_reason="tool_calls" downstream.
        self.assertIn("Let me check", result["text"])

    def test_stop_sequence_truncates_and_terminates_early(self):
        """A stop sequence appearing mid-response should truncate the text
        AND stop reading further NDJSON lines (i.e. not consume/depend on
        a later text block that would arrive after the real subprocess is
        killed)."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "The answer is 42###STOP###ignore this part"}],
                    "usage": {"input_tokens": 3, "output_tokens": 10},
                },
            },
            # This later chunk should never be consumed -- the generator
            # is expected to stop pulling lines once the stop sequence is
            # found in the message above.
            {"type": "assistant", "message": {"content": [{"type": "text", "text": " should be unreachable"}]}},
            {"type": "result", "subtype": "success"},
        )
        fake_proc = FakeProcess(lines)
        result = self._run(fake_proc, stop=["###STOP###"])
        self.assertEqual(result["text"], "The answer is 42")
        self.assertTrue(result["stop_matched"])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["tool_calls"], [])

    def test_stop_sequence_kills_subprocess(self):
        """The actual latency/cost win only materializes if the real
        subprocess is torn down on an early stop match, not just if our
        NDJSON-reading loop stops. Verify terminate()/wait() are invoked
        via the finally block regardless of an early stop-match break."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "abc STOPHERE def"}], "usage": {}}},
            {"type": "result", "subtype": "success"},
        )
        fake_proc = FakeProcess(lines)
        fake_proc.terminate = MagicMock()
        fake_proc.wait = MagicMock(return_value=0)
        self._run(fake_proc, stop=["STOPHERE"])
        fake_proc.terminate.assert_called_once()
        fake_proc.wait.assert_called_once()

    def test_no_stop_sequences_leaves_stop_matched_false(self):
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "plain reply"}], "usage": {}}},
            {"type": "result", "subtype": "success"},
        )
        result = self._run(FakeProcess(lines))
        self.assertFalse(result.get("stop_matched"))
        self.assertEqual(result["text"], "plain reply")

    def test_stop_sequence_matched_via_token_level_stream_event(self):
        """When --include-partial-messages is active (only happens when a
        stop sequence is requested), a match should be caught at the
        content_block_delta level -- BEFORE the full "assistant" chunk for
        that block ever arrives. This is what actually delivers an early
        cutoff mid-generation, not just at a message boundary."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "stream_event", "event": {"type": "content_block_start", "index": 0}},
            {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": "The answer "}}},
            {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": "is 42###STOP"}}},
            # This full "assistant" chunk and everything after it should
            # never be reached -- the token-level delta above already
            # matched and broke out.
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "The answer is 42###STOP###extra unreachable text"}], "usage": {"input_tokens": 1, "output_tokens": 99}}},
            {"type": "result", "subtype": "success"},
        )
        fake_proc = FakeProcess(lines)
        result = self._run(fake_proc, stop=["###STOP"])
        self.assertEqual(result["text"], "The answer is 42")
        self.assertTrue(result["stop_matched"])
        self.assertEqual(result["finish_reason"], "stop")
        # The usage from the never-reached "assistant"/"result" chunks
        # must NOT show up -- confirms we genuinely broke out early.
        self.assertNotIn("output_tokens", result["usage"])

    def test_stream_callback_receives_real_time_text_deltas(self):
        """stream_callback should be invoked with each real token-level
        text_delta AS IT ARRIVES (from --include-partial-messages
        content_block_delta events), not just once at the end with the
        full accumulated text. This is what genuine client-facing
        streaming depends on."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "stream_event", "event": {"type": "content_block_start", "index": 0}},
            {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": "Hello"}}},
            {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": ", world"}}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello, world"}],
                                               "usage": {"input_tokens": 1, "output_tokens": 2}}},
            {"type": "result", "subtype": "success"},
        )
        received = []
        result = self._run(FakeProcess(lines), stream_callback=received.append)
        self.assertEqual(received, ["Hello", ", world"])
        self.assertEqual(result["text"], "Hello, world")

    def test_stream_callback_truncated_at_stop_sequence(self):
        """When both stream_callback and stop are set, the callback only
        gets a shorter-than-already-sent view for the delta where the
        match actually completes -- but text already streamed in an
        EARLIER delta cannot be retroactively un-sent (an inherent
        limitation of real-time streaming: you can't take back tokens
        already delivered to the client). Here the stop sequence
        "STOPHERE" spans two deltas ("abc STOP" + "HERE def"), so the
        first delta streams in full before the match is even detectable;
        only the second delta's contribution is suppressed. The FINAL
        buffered `result["text"]` is still correctly truncated to "abc "
        -- only the live SSE stream itself briefly showed a few extra
        characters before the stop was recognized, which is the accepted
        real-time-streaming/stop-sequence tradeoff (documented in
        README)."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {"type": "stream_event", "event": {"type": "content_block_start", "index": 0}},
            {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": "abc STOP"}}},
            {"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
                                                 "delta": {"type": "text_delta", "text": "HERE def"}}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "abc STOPHERE def"}], "usage": {}}},
            {"type": "result", "subtype": "success"},
        )
        received = []
        result = self._run(FakeProcess(lines), stop=["STOPHERE"], stream_callback=received.append)
        # The first delta ("abc STOP") streamed before the match was
        # detectable; the second delta ("HERE def") was fully suppressed
        # since the match already completed by then.
        self.assertEqual("".join(received), "abc STOP")
        # But the final returned/recorded text IS correctly truncated.
        self.assertEqual(result["text"], "abc ")
        self.assertTrue(result["stop_matched"])

    def test_structured_output_tool_call_extracted_as_json(self):
        """--json-schema enforcement mechanism: StructuredOutput tool_use
        is the actual answer, not a real tool call to expose."""
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "StructuredOutput", "input": {"greeting": "hi"}}],
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                },
            },
            {"type": "result", "subtype": "success"},
        )
        result = self._run(FakeProcess(lines), json_schema={"type": "object"})
        self.assertEqual(result["structured_json"], {"greeting": "hi"})
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(json.loads(result["text"]), {"greeting": "hi"})

    def test_invalid_model_raises_classified_error(self):
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "There's an issue with the selected model (bogus). It may not exist."}],
                    "usage": {},
                },
                "error": "invalid_request",
            },
        )
        with self.assertRaises(shim.ClaudeCliError) as ctx:
            self._run(FakeProcess(lines))
        self.assertEqual(ctx.exception.http_status, 404)
        self.assertEqual(ctx.exception.error_type, "invalid_request_error")
        self.assertEqual(ctx.exception.code, "model_not_found")

    def test_max_output_tokens_error_maps_to_length_finish_reason(self):
        lines = _ndjson_lines(
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "API Error: exceeded the 20 output token maximum."}], "usage": {}},
                "error": "max_output_tokens",
            },
        )
        result = self._run(FakeProcess(lines))
        self.assertEqual(result["finish_reason"], "length")

    def test_cli_not_found_raises_503(self):
        with patch.object(shim.subprocess, "Popen", side_effect=FileNotFoundError()):
            with self.assertRaises(shim.ClaudeCliError) as ctx:
                shim.call_claude_streaming([{"role": "user", "content": "hi"}], "", "sonnet")
        self.assertEqual(ctx.exception.http_status, 503)
        self.assertEqual(ctx.exception.code, "claude_cli_not_found")


class TestErrorClassification(unittest.TestCase):
    def test_auth_errors(self):
        status, etype, code = shim._classify_error_text("Not logged in · Please run /login")
        self.assertEqual(status, 401)
        self.assertEqual(etype, "authentication_error")

    def test_rate_limit_errors(self):
        status, etype, code = shim._classify_error_text("You've hit your session limit")
        self.assertEqual(status, 429)
        self.assertEqual(etype, "rate_limit_error")

    def test_invalid_model_errors(self):
        status, etype, code = shim._classify_error_text("There's an issue with the selected model (xyz)")
        self.assertEqual(status, 404)
        self.assertEqual(code, "model_not_found")

    def test_context_length_errors(self):
        status, etype, code = shim._classify_error_text("Prompt is too long")
        self.assertEqual(status, 400)
        self.assertEqual(code, "context_length_exceeded")

    def test_unknown_error_defaults_to_500_api_error(self):
        status, etype, code = shim._classify_error_text("something totally unrecognized happened")
        self.assertEqual(status, 500)
        self.assertEqual(etype, "api_error")


class TestClaudeCliErrorShape(unittest.TestCase):
    def test_to_openai_body_shape(self):
        err = shim.ClaudeCliError(404, "invalid_request_error", "model not found", code="model_not_found", param="model")
        body = err.to_openai_body()
        self.assertEqual(body["error"]["message"], "model not found")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["code"], "model_not_found")
        self.assertEqual(body["error"]["param"], "model")


class TestStopSequences(unittest.TestCase):
    def test_truncates_at_earliest_match(self):
        text, matched = shim._apply_stop_sequences("hello world STOP more text", ["STOP", "world"])
        self.assertTrue(matched)
        self.assertEqual(text, "hello ")

    def test_no_match_returns_unchanged(self):
        text, matched = shim._apply_stop_sequences("hello world", ["XYZ"])
        self.assertFalse(matched)
        self.assertEqual(text, "hello world")

    def test_string_stop_treated_as_single_sequence(self):
        text, matched = shim._apply_stop_sequences("abcSTOPdef", "STOP")
        self.assertTrue(matched)
        self.assertEqual(text, "abc")

    def test_empty_stop_is_noop(self):
        text, matched = shim._apply_stop_sequences("hello", None)
        self.assertFalse(matched)
        self.assertEqual(text, "hello")


class TestToolRetryWrapper(unittest.TestCase):
    """call_claude_with_tool_retry: verifies retry count, backoff timing,
    session-mode switching on retry, and that it stops as soon as a real
    tool_use shows up."""

    def _stub_result(self, tool_calls=None, text=None):
        return {"text": text, "tool_calls": tool_calls or [], "usage": {}, "finish_reason": "stop", "structured_json": None}

    def test_no_retry_needed_when_tool_call_on_first_try(self):
        with patch.object(shim, "call_claude_streaming", return_value=self._stub_result(tool_calls=[{"name": "x"}])) as mock_call, \
             patch.object(shim.time, "sleep") as mock_sleep:
            result, mode, sid = shim.call_claude_with_tool_retry(
                [], [], "", "sonnet", tools_requested=True, session_mode="resume", session_id="abc"
            )
        self.assertEqual(mock_call.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(result["tool_calls"], [{"name": "x"}])
        self.assertEqual(mode, "resume")
        self.assertEqual(sid, "abc")

    def test_no_retry_when_no_tools_requested(self):
        with patch.object(shim, "call_claude_streaming", return_value=self._stub_result(text="just text")) as mock_call, \
             patch.object(shim.time, "sleep") as mock_sleep:
            result, mode, sid = shim.call_claude_with_tool_retry(
                [], [], "", "sonnet", tools_requested=False, session_mode="fresh", session_id="abc"
            )
        self.assertEqual(mock_call.call_count, 1)
        mock_sleep.assert_not_called()
        self.assertEqual(result["text"], "just text")

    def test_retries_with_backoff_until_tool_call_appears(self):
        side_effects = [
            self._stub_result(text="narrating..."),
            self._stub_result(text="narrating again..."),
            self._stub_result(tool_calls=[{"name": "get_weather"}]),
        ]
        with patch.object(shim, "call_claude_streaming", side_effect=side_effects) as mock_call, \
             patch.object(shim.time, "sleep") as mock_sleep:
            result, mode, sid = shim.call_claude_with_tool_retry(
                [], [], "", "sonnet", tools_requested=True, session_mode="resume", session_id="abc"
            )
        self.assertEqual(mock_call.call_count, 3)
        self.assertEqual(result["tool_calls"], [{"name": "get_weather"}])
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)

    def test_retry_switches_to_fresh_session_with_full_history(self):
        """A retry must NOT keep resuming the same session (causes an
        'already in use' CLI error) or resume with more delta messages
        (would duplicate transcript entries) -- it must switch to a
        brand-new fresh session sending the FULL history."""
        side_effects = [self._stub_result(text="narrating"), self._stub_result(tool_calls=[{"name": "x"}])]
        full_history = [{"role": "user", "content": "full history msg"}]
        with patch.object(shim, "call_claude_streaming", side_effect=side_effects) as mock_call, \
             patch.object(shim.time, "sleep"):
            shim.call_claude_with_tool_retry(
                [{"role": "user", "content": "delta only"}], full_history, "", "sonnet",
                tools_requested=True, session_mode="resume", session_id="original-sid",
            )
        first_call_kwargs = mock_call.call_args_list[0].kwargs
        second_call_kwargs = mock_call.call_args_list[1].kwargs
        self.assertEqual(first_call_kwargs["session_mode"], "resume")
        self.assertEqual(first_call_kwargs["session_id"], "original-sid")
        self.assertEqual(second_call_kwargs["session_mode"], "fresh")
        self.assertNotEqual(second_call_kwargs["session_id"], "original-sid")
        self.assertEqual(mock_call.call_args_list[1].args[0], full_history)

    def test_gives_up_after_total_attempts_exhausted(self):
        total_attempts = 1 + shim.TOOL_CALL_MAX_RETRIES
        side_effects = [self._stub_result(text="still narrating")] * total_attempts
        with patch.object(shim, "call_claude_streaming", side_effect=side_effects) as mock_call, \
             patch.object(shim.time, "sleep"):
            result, mode, sid = shim.call_claude_with_tool_retry(
                [], [], "", "sonnet", tools_requested=True, session_mode="fresh", session_id="abc"
            )
        self.assertEqual(mock_call.call_count, total_attempts)
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["text"], "still narrating")

    def test_backoff_schedule_matches_cline_derived_formula(self):
        self.assertEqual(shim._backoff_delay_s(0), 2.0)
        self.assertEqual(shim._backoff_delay_s(1), 4.0)
        self.assertEqual(shim._backoff_delay_s(2), 8.0)
        self.assertEqual(shim._backoff_delay_s(3), 15.0)  # capped
        self.assertEqual(shim._backoff_delay_s(10), 15.0)  # stays capped


class TestUnsupportedSamplingParamsWarning(unittest.TestCase):
    """temperature/top_p/etc have no equivalent anywhere in the Claude
    Code CLI (verified via `claude --help` and the official env-vars
    docs, 2026-08-13). These must never hard-error (real clients routinely
    send explicit defaults like temperature: 1.0 on every request, which
    carries no signal the caller wants non-default behavior) -- only a
    one-time-per-name stderr warning, request otherwise unaffected."""

    def setUp(self):
        shim._warned_sampling_params.clear()

    def test_warns_once_per_parameter_name(self):
        with patch.object(shim.sys, "stderr") as mock_stderr:
            shim._warn_unsupported_sampling_params({"temperature": 0.7})
            shim._warn_unsupported_sampling_params({"temperature": 0.9})  # same name again
        self.assertEqual(mock_stderr.write.call_count, 1)
        self.assertIn("temperature", mock_stderr.write.call_args[0][0])

    def test_does_not_warn_for_absent_or_null_params(self):
        with patch.object(shim.sys, "stderr") as mock_stderr:
            shim._warn_unsupported_sampling_params({"temperature": None, "model": "sonnet"})
        mock_stderr.write.assert_not_called()

    def test_warns_for_each_distinct_unsupported_param(self):
        with patch.object(shim.sys, "stderr") as mock_stderr:
            shim._warn_unsupported_sampling_params({"temperature": 0.5, "top_p": 0.9, "seed": 42})
        self.assertEqual(mock_stderr.write.call_count, 3)

    def test_supported_params_never_warn(self):
        with patch.object(shim.sys, "stderr") as mock_stderr:
            shim._warn_unsupported_sampling_params({"max_tokens": 100, "stop": ["x"], "stream": True})
        mock_stderr.write.assert_not_called()


class TestFetchModelList(unittest.TestCase):
    """/v1/models should prefer Claude Code's own OAuth credentials over
    an API key over a hardcoded fallback (user direction, 2026-08-14):
    OAuth first because the whole point of this shim is avoiding metered
    API billing, so this one free metadata call should work without
    requiring the caller to separately configure ANTHROPIC_API_KEY."""

    def setUp(self):
        shim._model_list_cache["data"] = None
        shim._model_list_cache["fetched_at"] = 0.0

    def _fake_urlopen_response(self, data):
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = json.dumps({"data": data}).encode()
        resp.__enter__.return_value = resp
        return resp

    def test_prefers_oauth_over_api_key_when_both_available(self):
        oauth_models = [{"id": "claude-oauth-model"}]
        api_key_models = [{"id": "claude-apikey-model"}]

        def fake_urlopen(req, timeout=None):
            auth = req.headers.get("Authorization")
            if auth and auth.startswith("Bearer"):
                return self._fake_urlopen_response(oauth_models)
            return self._fake_urlopen_response(api_key_models)

        with patch.object(shim, "_read_claude_oauth_token", return_value="tok123"), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake"}), \
             patch.object(shim.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = shim.fetch_model_list()
        self.assertEqual(result, [{"id": "claude-oauth-model", "object": "model"}])

    def test_falls_back_to_api_key_when_oauth_absent(self):
        with patch.object(shim, "_read_claude_oauth_token", return_value=None), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake"}), \
             patch.object(shim.urllib.request, "urlopen",
                           return_value=self._fake_urlopen_response([{"id": "claude-apikey-model"}])):
            result = shim.fetch_model_list()
        self.assertEqual(result, [{"id": "claude-apikey-model", "object": "model"}])

    def test_falls_back_to_hardcoded_list_when_no_auth_available(self):
        with patch.object(shim, "_read_claude_oauth_token", return_value=None), \
             patch.dict(os.environ, {}, clear=True):
            result = shim.fetch_model_list()
        self.assertEqual(result, [{"id": m, "object": "model"} for m in shim.KNOWN_MODEL_ALIASES])

    def test_falls_back_to_hardcoded_list_when_api_call_fails(self):
        """Even with a token present, a network/auth failure must fall
        through to the hardcoded list, never raise up into a /v1/models
        request."""
        with patch.object(shim, "_read_claude_oauth_token", return_value="tok123"), \
             patch.dict(os.environ, {}, clear=True), \
             patch.object(shim.urllib.request, "urlopen", side_effect=shim.urllib.error.URLError("boom")):
            result = shim.fetch_model_list()
        self.assertEqual(result, [{"id": m, "object": "model"} for m in shim.KNOWN_MODEL_ALIASES])

    def test_result_is_cached_within_ttl(self):
        with patch.object(shim, "_read_claude_oauth_token", return_value="tok123"), \
             patch.object(shim.urllib.request, "urlopen",
                           return_value=self._fake_urlopen_response([{"id": "claude-cached-model"}])) as mock_urlopen:
            first = shim.fetch_model_list()
            second = shim.fetch_model_list()
        self.assertEqual(first, second)
        mock_urlopen.assert_called_once()

    def test_oauth_token_read_returns_none_on_missing_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            self.assertIsNone(shim._read_claude_oauth_token())

    def test_oauth_token_read_returns_none_when_expired(self):
        expired_creds = json.dumps({
            "claudeAiOauth": {"accessToken": "tok123", "expiresAt": 1}  # far in the past
        })
        with patch("builtins.open", MagicMock(return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=lambda: expired_creds)),
                __exit__=MagicMock(return_value=False)))), \
             patch.object(shim.json, "load", return_value=json.loads(expired_creds)):
            self.assertIsNone(shim._read_claude_oauth_token())

    def test_oauth_token_read_returns_token_when_valid(self):
        valid_creds = {"claudeAiOauth": {"accessToken": "tok123", "expiresAt": (time.time() + 3600) * 1000}}
        with patch("builtins.open", MagicMock()), \
             patch.object(shim.json, "load", return_value=valid_creds):
            self.assertEqual(shim._read_claude_oauth_token(), "tok123")


class TestNormalizeModelName(unittest.TestCase):
    """OpenRouter (https://openrouter.ai) model-slug compatibility --
    translates its Anthropic naming convention into Claude Code's own
    `--model` convention. Expected values below were confirmed live
    against the real `claude` CLI (2026-08-14): each transformed name
    was passed to `claude --model <x> -p` with a "state your exact
    model version" probe and the CLI accepted it / echoed back the
    matching real model id."""

    def test_tilde_and_anthropic_prefix_latest_alias(self):
        for prefix_form in (
            "~anthropic/claude-sonnet-latest",
            "anthropic/claude-sonnet-latest",
            "claude-sonnet-latest",
        ):
            self.assertEqual(shim.normalize_model_name(prefix_form), "sonnet")
        self.assertEqual(shim.normalize_model_name("anthropic/claude-opus-latest"), "opus")
        self.assertEqual(shim.normalize_model_name("anthropic/claude-haiku-latest"), "haiku")

    def test_dotted_version_becomes_dashed(self):
        self.assertEqual(shim.normalize_model_name("anthropic/claude-sonnet-4.5"), "claude-sonnet-4-5")
        self.assertEqual(shim.normalize_model_name("claude-sonnet-4.5"), "claude-sonnet-4-5")
        self.assertEqual(shim.normalize_model_name("anthropic/claude-opus-4.8"), "claude-opus-4-8")

    def test_fast_suffix_stripped_before_dot_conversion(self):
        shim._warned_fast_models.clear()
        self.assertEqual(shim.normalize_model_name("anthropic/claude-opus-4.8-fast"), "claude-opus-4-8")

    def test_fast_suffix_warns_once_per_model_string(self):
        shim._warned_fast_models.clear()
        with patch.object(shim.sys.stderr, "write") as mock_write:
            shim.normalize_model_name("anthropic/claude-opus-4.8-fast")
            shim.normalize_model_name("anthropic/claude-opus-4.8-fast")
        self.assertEqual(mock_write.call_count, 1)

    def test_already_native_names_pass_through_unchanged(self):
        self.assertEqual(shim.normalize_model_name("claude-sonnet-4-5-20250929"), "claude-sonnet-4-5-20250929")
        self.assertEqual(shim.normalize_model_name("sonnet"), "sonnet")
        self.assertEqual(shim.normalize_model_name("opus"), "opus")
        self.assertEqual(shim.normalize_model_name("gpt-4"), "gpt-4")

    def test_falsy_input_passes_through(self):
        self.assertIsNone(shim.normalize_model_name(None))
        self.assertEqual(shim.normalize_model_name(""), "")


if __name__ == "__main__":
    unittest.main()
