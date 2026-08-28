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
from claudecode_as_openai.constants import CLAUDE_TIMEOUT_S
from claudecode_as_openai.state import _CLAUDE_CWD, _WARM_POOL, _WARM_POOL_LOCK
from claudecode_as_openai.sessions import resolve_session, record_session
from claudecode_as_openai.parsing import _iter_ndjson_lines, _iter_ndjson_lines_pty
from claudecode_as_openai.quota import _log_quota_snapshot

class WarmProcess:
    """One already-spawned `claude -p --input-format stream-json
    --output-format stream-json` subprocess, resumed onto a specific
    Claude session_id, parked waiting to serve exactly one more turn of
    that same conversation. Not reused after that turn is served --
    callers spawn a fresh WarmProcess for the turn after this one (see
    WarmPool)."""

    def __init__(self, fingerprint, session_id, model, system_prompt, tools_requested, tools,
                 mcp_tool_config, effort, use_pty, env_overrides=None):
        self.fingerprint = fingerprint
        self.session_id = session_id
        self.model = model
        self.system_prompt = system_prompt
        self.tools_requested = tools_requested
        self.tools = tools
        self.mcp_tool_config = mcp_tool_config
        self.effort = effort
        self.use_pty = use_pty
        self.env_overrides = env_overrides
        self.proc = None
        self.pty_master_fd = None
        self.system_prompt_file = None
        self._reader_thread = None
        self._line_queue = queue.Queue()
        self._lock = threading.Lock()
        self._claimed = False
        self.spawned_at = time.time()  # when this process was spawned
        self.parked_at = None          # set by WarmPool.park(); when it was parked idle
        self._start()

    def _start(self):
        # Warm pool always resumes a known session; json_schema=None here
        # (json_schema requires fresh spawns with full orchestration). With
        # prose fallback removed, max_turns=1 is always safe.
        max_turns = 1
        cmd = _build_claude_cmd(
            self.model, "resume", self.session_id, self.tools_requested, max_turns,
            json_schema=None, want_partial_messages=self.use_pty,
            mcp_tool_config=self.mcp_tool_config, effort=self.effort,
            input_format="stream-json",
        )
        if self.system_prompt:
            # Matches call_claude_streaming's own >4000-char threshold
            # for --system-prompt-file vs --system-prompt (see there for
            # why: argv length limits). A resumed session already has
            # its system prompt recorded server-side, so this is
            # belt-and-suspenders consistency with the cold path rather
            # than something a resume strictly needs.
            if len(self.system_prompt) > 4000:
                fd, self.system_prompt_file = tempfile.mkstemp(
                    prefix="claudecode-as-openai-sysprompt-", suffix=".txt",
                )
                with os.fdopen(fd, "w") as f:
                    f.write(self.system_prompt)
                cmd += ["--system-prompt-file", self.system_prompt_file]
            else:
                cmd += ["--system-prompt", self.system_prompt]
        # A warm process's env is fixed for its whole parked lifetime
        # (unlike the cold path's _scoped_env_overrides, which only
        # needs to patch subprocess.Popen for the duration of one
        # request's call stack) -- passed directly at spawn time here
        # instead. Without this, CLAUDE_CODE_MAX_OUTPUT_TOKENS and
        # CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC (see _BASE_ENV_OVERRIDES)
        # would silently NOT apply to any turn served by a warm process.
        spawn_env = dict(os.environ)
        if self.env_overrides:
            spawn_env.update(self.env_overrides)
        if self.use_pty:
            self.pty_master_fd, pty_slave_fd = pty.openpty()
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=pty_slave_fd,
                stderr=subprocess.PIPE, cwd=_CLAUDE_CWD, env=spawn_env,
            )
            os.close(pty_slave_fd)
        else:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, cwd=_CLAUDE_CWD, env=spawn_env,
            )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        # Runs for the whole process lifetime, pushing every parsed
        # NDJSON line onto a queue that send_turn drains per-turn. A
        # background reader (rather than reading synchronously inside
        # send_turn) is what lets the process sit parked indefinitely
        # between turns without a blocking read call pinning a thread
        # doing nothing useful -- this thread blocks on I/O, which is
        # cheap, not on CPU. timeout_s=None on the PTY path is
        # deliberate: _iter_ndjson_lines_pty's deadline is per-
        # generator-lifetime, not per-line, and this generator IS the
        # process's whole lifetime (spawn through however many turns get
        # served) -- a finite deadline here would silently kill an
        # idle-but-still-alive parked process out from under send_turn.
        # Per-turn bounding happens instead in send_turn's own
        # queue.get(timeout=...).
        try:
            if self.use_pty:
                for parsed in _iter_ndjson_lines_pty(self.pty_master_fd, self.proc, timeout_s=None):
                    self._line_queue.put(parsed)
            else:
                for parsed in _iter_ndjson_lines(self.proc):
                    self._line_queue.put(parsed)
        except Exception:
            pass
        finally:
            self._line_queue.put(None)  # sentinel: stdout closed / process exited

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def claim(self):
        """Returns True exactly once -- the first caller to claim this
        process for a turn. Guards against a race between the pool
        handing out this same parked process to two concurrent
        requests."""
        with self._lock:
            if self._claimed or not self.is_alive():
                return False
            self._claimed = True
            return True

    def send_turn(self, claude_messages, stop=None, stream_callback=None):
        """Feed ONE turn (a list of new Claude-shaped messages -- the
        delta beyond what this session already has) to the already-
        running process and block for its result. Must only be called
        after claim() returns True. Uses the exact same response-parsing
        logic as the cold path (_consume_claude_response) so behavior is
        identical either way."""
        stop_sequences = _normalize_stop_sequences(stop)
        # claude_messages is the delta list this shim would otherwise
        # have written whole to a fresh process's stdin; stream-json
        # input takes one JSON object per line instead, so each element
        # is sent as its own {"type": "user", "message": ...} frame.
        for msg in claude_messages:
            line = json.dumps({"type": msg.get("role", "user"), "message": msg})
            if self.use_pty:
                os.write(self.proc.stdin.fileno(), (line + "\n").encode())
            else:
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()

        def chunk_source():
            while True:
                try:
                    item = self._line_queue.get(timeout=CLAUDE_TIMEOUT_S)
                except queue.Empty:
                    # No output from the warm process for CLAUDE_TIMEOUT_S
                    # seconds -- it is stuck. Kill it so quota stops burning
                    # and surface a proper error to the caller.
                    elapsed = time.time() - self.spawned_at
                    sys.stderr.write(
                        f"claudecode-as-openai: warm-pool: subprocess"
                        f" {self.session_id} produced no output for"
                        f" {CLAUDE_TIMEOUT_S}s (total age {elapsed:.0f}s);"
                        f" killing\n"
                    )
                    self.kill()
                    raise ClaudeCliError(
                        500, "api_error",
                        f"Warm-pool subprocess timed out after"
                        f" {CLAUDE_TIMEOUT_S}s with no output.",
                        code="warm_process_timeout",
                    )
                if item is None:
                    return
                yield item

        deadline = time.time() + CLAUDE_TIMEOUT_S
        return _consume_claude_response(chunk_source(), deadline, stop_sequences, stream_callback)

    def kill(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        if self.use_pty and self.pty_master_fd is not None:
            try:
                os.close(self.pty_master_fd)
            except OSError:
                pass
        if self.system_prompt_file:
            try:
                os.unlink(self.system_prompt_file)
            except OSError:
                pass
        if self.mcp_tool_config is not None:
            try:
                os.unlink(self.mcp_tool_config["manifest_path"])
            except OSError:
                pass


