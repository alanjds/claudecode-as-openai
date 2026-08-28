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



# parsing module is mostly standalone
def _parse_ndjson_line(line):
    """Shared tail logic for both NDJSON readers below: strip, skip blank,
    parse JSON, skip silently on a decode error (a malformed/partial line
    is not fatal -- just not a usable chunk). Returns the parsed dict, or
    None if the line should be skipped."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _iter_ndjson_lines(proc):
    for raw_line in proc.stdout:
        parsed = _parse_ndjson_line(raw_line)
        if parsed is not None:
            yield parsed


def _iter_ndjson_lines_pty(master_fd, proc, timeout_s):
    """Same contract as _iter_ndjson_lines, but reads from a PTY master fd
    instead of a plain subprocess.PIPE.

    Why this exists: `claude`'s own stdout is FULLY BUFFERED (not just
    line-buffered) when its stdout is a plain pipe with no TTY attached --
    verified live: with --include-partial-messages, real per-token
    content_block_delta events all arrive in 2-3 giant bursts within the
    same ~20ms window regardless of the actual generation taking many
    seconds, because glibc's stdio only flushes on a full buffer or
    process exit when stdout isn't a terminal. Attaching a real PTY as
    `claude`'s stdout makes it use LINE buffering like an interactive
    terminal session -- verified live: the same request then delivers
    dozens of individual deltas spread realistically across the full
    generation time (e.g. 21 deltas over ~22s of a 300-word story,
    roughly one every 0.5-0.7s, matching genuine token-generation pacing).
    This is the only way to get real client-facing streaming out of this
    CLI; there is no flag to force unbuffered/line-buffered stdout.

    `timeout_s`, if given, is a single deadline for the WHOLE generator
    lifetime, not per-line -- correct for call_claude_streaming's
    one-shot use (one call, one bounded wait), but WRONG for a
    WarmProcess's background reader, which must survive indefinitely
    across an unbounded number of turns with idle gaps between them.
    Pass timeout_s=None for that case: the generator then only returns
    on EOF/process-exit, with no wall-clock cutoff at all (see
    WarmProcess._read_loop)."""
    deadline = None if timeout_s is None else time.time() + timeout_s
    buf = b""
    while True:
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            select_timeout = min(remaining, 1.0)
        else:
            select_timeout = 1.0
        try:
            ready, _, _ = select.select([master_fd], [], [], select_timeout)
        except (OSError, ValueError):
            return
        if not ready:
            if proc.poll() is not None:
                # Process exited and nothing left to read.
                try:
                    ready2, _, _ = select.select([master_fd], [], [], 0)
                except (OSError, ValueError):
                    ready2 = []
                if not ready2:
                    return
            continue
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            # EIO is the normal "slave side closed" signal on Linux PTYs.
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            raw_line, buf = buf.split(b"\n", 1)
            parsed = _parse_ndjson_line(raw_line.decode("utf-8", errors="replace"))
            if parsed is not None:
                yield parsed


