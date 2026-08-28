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



class ClaudeCliError(Exception):
    """Carries enough info to build an OpenAI-shaped error response.
    http_status: int: e.g. 400, 401, 404, 429, 500, 503.
    error_type: OpenAI error taxonomy string, e.g. "invalid_request_error",
        "authentication_error", "rate_limit_error", "api_error".
    code: short machine-readable code, e.g. "model_not_found", or None.
    """

    def __init__(self, http_status, error_type, message, code=None, param=None):
        super().__init__(message)
        self.http_status = http_status
        self.error_type = error_type
        self.message = message
        self.code = code
        self.param = param

    def to_openai_body(self):
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


# Heuristic classification of Claude Code's own plain-text error messages
# into OpenAI's error taxonomy. This is necessarily fragile -- it string-
# matches phrases from https://code.claude.com/docs/en/errors (checked
# 2026-08-13) -- and will need updating if Anthropic changes their error
# wording. Order matters: more specific patterns should come first.
_ERROR_PATTERNS = [
    (
        (
            "not logged in", "please run /login", "login expired",
            "oauth token", "invalid api key", "could not resolve authentication",
            "invalid auth token", "authentication credentials",
            "organization has disabled api key authentication",
            "organization has disabled claude subscription access",
        ),
        401, "authentication_error", "authentication_failed",
    ),
    (
        (
            "session limit", "weekly limit", "credit balance is too low",
            "spend limit", "request rejected (429)",
            "server is temporarily limiting requests", "rate limit",
        ),
        429, "rate_limit_error", "rate_limit_exceeded",
    ),
    (
        (
            "issue with the selected model", "not a recognized model id",
            "restricted by your organization", "not available with the claude",
        ),
        404, "invalid_request_error", "model_not_found",
    ),
    (
        (
            "prompt is too long", "context exceeds", "request too large",
            "conversation too long", "extra inputs are not permitted",
        ),
        400, "invalid_request_error", "context_length_exceeded",
    ),
    (
        ("overloaded", "internal server error", "500 internal"),
        503, "api_error", "overloaded",
    ),
]


def _classify_error_text(text):
    lower = (text or "").lower()
    for phrases, status, etype, code in _ERROR_PATTERNS:
        if any(p in lower for p in phrases):
            return status, etype, code
    return 500, "api_error", None


