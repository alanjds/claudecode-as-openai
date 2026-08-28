#!/usr/bin/env python3
"""OpenAI-chat-completions-compatible shim over the local Claude Code CLI.

Translates OpenAI's `/v1/chat/completions` API onto `claude -p` (Claude
Code's non-interactive mode): native tool-calling via MCP tool
registration, session caching via --session-id/--resume, OpenAI-shaped
error translation, response_format via --json-schema, OpenRouter model-
name and reasoning-effort compatibility, and stop-sequence emulation.

See README.md "Capability audit" for the full, empirically-verified list
of what works, what's approximated, and what's a genuine CLI limitation,
and CHANGELOG.md for the investigation trail behind each design choice
referenced in comments below.

Run:
    python3 -m claudecode_as_openai.shim [port]   # default port 8977

Point Hermes at it:
    hermes config set model.provider custom
    hermes config set model.base_url http://127.0.0.1:8977/v1
    hermes config set model.api_key not-needed
    hermes config set model.default sonnet
"""
import hashlib
import json
import os
import pty
import queue
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLAUDE_BIN = "claude"
DEFAULT_MODEL = "sonnet"
# Default port, overridable via CLAUDE_OPENAI_PORT env var. The running
# Hermes Agent itself may already be consuming port 8977 (the shim it runs on),
# so tests or secondary instances should use a different port via the env var.
DEFAULT_PORT = int(os.environ.get("CLAUDE_OPENAI_PORT", "8977"))
# When a caller doesn't pass `reasoning` at all (e.g. Hermes on a custom
# provider URL, which skips extra_body.reasoning to avoid 400s on unknown
# backends), fall back to this effort level rather than silently disabling
# thinking. Set to the effort level configured in Hermes's reasoning_effort
# config or any other value from _REASONING_EFFORT_MAP. Empty string or
# absent = no default effort (thinking disabled unless explicitly requested).
_DEFAULT_EFFORT_ENV = os.environ.get("CLAUDE_OPENAI_DEFAULT_EFFORT", "").strip().lower() or None
KNOWN_MODEL_ALIASES = [
    # Hardcoded last-resort fallback for /v1/models when neither OAuth nor
    # an API key is available to query the real Anthropic /v1/models
    # endpoint (see fetch_model_list()). Extracted from strings embedded
    # in the compiled `claude` binary (2026-08-14) -- will go stale as new
    # models ship, which is exactly why the live query is preferred.
    "sonnet", "opus", "haiku",
]
CLAUDE_TIMEOUT_S = 300

# OpenRouter model-slug compatibility (see CHANGELOG). Translates
# OpenRouter's Anthropic model naming convention
# (`anthropic/claude-sonnet-4.5`, `~anthropic/claude-sonnet-latest`) into
# Claude Code's own `--model` convention (`claude-sonnet-4-5`, bare
# `sonnet`/`opus`/`haiku` aliases) so a client pointed at this shim with
# an OpenRouter-style model string just works.
_OPENROUTER_LATEST_ALIAS_RE = re.compile(r"^claude-(sonnet|opus|haiku)-latest$")
_OPENROUTER_VERSION_RE = re.compile(r"^claude-(sonnet|opus|haiku)-(\d+)\.(\d+)$")
_OPENROUTER_FAST_SUFFIX_RE = re.compile(r"^(claude-(?:sonnet|opus|haiku)-[\d.]+)-fast$")
_warned_fast_models = set()




# Local imports
from claudecode_as_openai.constants import CLAUDE_TIMEOUT_S, TOOL_CALL_MAX_RETRIES
from claudecode_as_openai.state import _CLAUDE_CWD, _rate_limit_cache, _rate_limit_cache_lock
from claudecode_as_openai.errors import ClaudeCliError, _classify_error_text
from claudecode_as_openai.tools import build_mcp_tool_manifest, strip_mcp_tool_prefix
from claudecode_as_openai.messages import build_claude_messages
from claudecode_as_openai.parsing import _iter_ndjson_lines, _iter_ndjson_lines_pty, _parse_ndjson_line
from claudecode_as_openai.warm_pool import WarmPool
from claudecode_as_openai.quota import _log_quota_snapshot

def call_claude_streaming(
    claude_messages,
    system_prompt,
    model,
    session_mode="fresh",
    session_id=None,
    tools_requested=False,
    tools=None,
    json_schema=None,
    stop=None,
    stream_callback=None,
    effort=None,
):
    """Spawn `claude -p` in native stream-json mode, feed the message array
    on stdin, and return as soon as a usable assistant message (text and/or
    tool_use blocks) is seen. Terminates the subprocess immediately after
    extracting what's needed so a downstream max-turns/self-resolution
    failure never surfaces as a shim-level crash.

    `stop`, if given, is checked against the accumulated text after every
    text content block arrives (Claude Code has no native stop-sequence
    flag) -- killing the subprocess the instant a match appears avoids
    paying for and waiting on the rest of an unwanted response.

    `stream_callback`, if given, is called with each real token-level text
    chunk as it arrives via --include-partial-messages content_block_delta
    events -- this delivers genuine real-time streaming to an OpenAI
    client. Only wired up for the single-choice, no-tools, no-json_schema
    path (see _handle_chat_completion): with tool retry in play, a failed
    attempt's narration text must not reach the client before the shim
    knows to discard it and retry.

    `tools`, if given, is registered as a real MCP tool server (see
    build_mcp_tool_config) instead of being described in prose -- the fix
    for the ~40% single-shot tool-call reliability problem, reaching 100%
    turn-1 dispatch. Falls back to tools_requested's prose-description
    path (system_prompt already carries it -- see _handle_chat_completion)
    when the tool set can't be represented as a valid MCP manifest.

    Returns a dict: {"text", "tool_calls", "usage", "finish_reason",
    "structured_json"}. Raises ClaudeCliError for conditions that should
    become an OpenAI-shaped error response -- see _classify_error_text.

    See EXTENDED_DISALLOWED_TOOLS and _build_claude_cmd for why
    --disallowedTools (not --tools "") is used when tools are requested,
    and why the no-tools path uses --tools "" + --strict-mcp-config
    instead."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    stop_sequences = []
    if stop:
        stop_sequences = [stop] if isinstance(stop, str) else [s for s in stop if s]

    system_prompt_file = None
    mcp_tool_config = build_mcp_tool_config(tools) if tools else None
    # --json-schema needs a couple of internal turns (an internal
    # "StructuredOutput" tool call + a corrective retry if the model
    # forgets it) to actually enforce the schema -- max_turns=1 leaves it
    # hanging at error_max_turns. MCP tool dispatch (and prose fallback, now
    # removed) both break out on tool_use before a second turn matters, so
    # max_turns stays >= 2 only when json_schema is set.
    max_turns = 3 if json_schema is not None else 1
    cmd = _build_claude_cmd(
        model, session_mode, session_id, tools_requested, max_turns, json_schema,
        want_partial_messages=bool(stop_sequences) or stream_callback is not None,
        mcp_tool_config=mcp_tool_config, effort=effort,
    )
    if system_prompt:
        if len(system_prompt) > 4000:
            fd, system_prompt_file = tempfile.mkstemp(prefix="claudecode-as-openai-sysprompt-", suffix=".txt")
            with os.fdopen(fd, "w") as f:
                f.write(system_prompt)
            cmd += ["--system-prompt-file", system_prompt_file]
        else:
            cmd += ["--system-prompt", system_prompt]

    try:
        use_pty = stream_callback is not None
        pty_master_fd = None
        if use_pty:
            # See _iter_ndjson_lines_pty docstring: a plain pipe leaves
            # `claude`'s stdout fully buffered (bursty, not real-time), so
            # real client-facing streaming needs a PTY attached as stdout
            # to force line buffering.
            pty_master_fd, pty_slave_fd = pty.openpty()
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=pty_slave_fd,
                stderr=subprocess.PIPE,
                cwd=_CLAUDE_CWD,
            )
            os.close(pty_slave_fd)
        else:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=_CLAUDE_CWD,
            )
    except FileNotFoundError:
        if system_prompt_file:
            try:
                os.unlink(system_prompt_file)
            except OSError:
                pass
        raise ClaudeCliError(
            503, "api_error",
            f"'{CLAUDE_BIN}' CLI not found on PATH. Install Claude Code "
            "(https://code.claude.com) and ensure `claude` is runnable.",
            code="claude_cli_not_found",
        )

    try:
        if use_pty:
            proc.stdin.write(json.dumps(claude_messages).encode())
        else:
            proc.stdin.write(json.dumps(claude_messages))
        proc.stdin.close()

        deadline = time.time() + CLAUDE_TIMEOUT_S
        chunk_source = (
            _iter_ndjson_lines_pty(pty_master_fd, proc, CLAUDE_TIMEOUT_S)
            if use_pty else _iter_ndjson_lines(proc)
        )
        return _consume_claude_response(chunk_source, deadline, stop_sequences, stream_callback)
    finally:
        # Terminating here (rather than letting the subprocess run to its
        # own natural end) is what actually delivers the latency/cost win
        # for an early stop-sequence match -- the `break` above only stops
        # US reading further NDJSON lines; the underlying `claude` process
        # would otherwise keep generating (and being billed for) tokens
        # nobody will see. This path already ran for every other early-exit
        # case (tool_use, error, max-turns); stop-sequence matches now share
        # the same real subprocess teardown instead of a fake early return.
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if use_pty and pty_master_fd is not None:
            try:
                os.close(pty_master_fd)
            except OSError:
                pass
        if system_prompt_file:
            try:
                os.unlink(system_prompt_file)
            except OSError:
                pass
        if mcp_tool_config is not None:
            try:
                os.unlink(mcp_tool_config["manifest_path"])
            except OSError:
                pass


