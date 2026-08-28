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
from claudecode_as_openai.state import _SESSIONS_DB, _SESSIONS_DB_LOCK
from claudecode_as_openai.messages import system_prompt_from_messages

def _conversation_key(openai_messages):
    """Stable key for the conversation this message list belongs to: hash
    of the system messages + the first non-system message. Stays stable
    across the whole conversation even as more turns are appended."""
    sys_msgs = [m for m in openai_messages if m.get("role") == "system"]
    first_non_system = next((m for m in openai_messages if m.get("role") != "system"), None)
    basis = json.dumps([sys_msgs, first_non_system], sort_keys=True, default=str)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _messages_equal(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def resolve_session(openai_messages):
    """Returns (mode, claude_session_id, delta_openai_messages, conv_key).
    mode is "resume" (continuing a known conversation -- only send the new
    tail messages) or "fresh" (new conversation, or the history diverged
    from what we last synced -- send everything)."""
    key = _conversation_key(openai_messages)
    with _SESSION_LOCK:
        entry = _SESSION_STORE.get(key)
        if entry:
            synced = entry["synced_messages"]
            match = len(openai_messages) > len(synced) and _messages_equal(openai_messages[: len(synced)], synced)
            if match:
                delta = openai_messages[len(synced) :]
                return "resume", entry["claude_session_id"], delta, key
    return "fresh", str(uuid.uuid4()), openai_messages, key


def record_session(conv_key, claude_session_id, full_openai_messages, assistant_reply_message):
    with _SESSION_LOCK:
        synced = list(full_openai_messages) + [assistant_reply_message]
        _SESSION_STORE[conv_key] = {"claude_session_id": claude_session_id, "synced_messages": synced}
        while len(_SESSION_STORE) > _SESSION_STORE_MAX:
            oldest = next(iter(_SESSION_STORE))
            if oldest == conv_key:
                break
            _SESSION_STORE.pop(oldest, None)

