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
from claudecode_as_openai.constants import (
    _MCP_TOOL_NAME_RE,
    _MCP_SERVER_NAME,
    _MCP_TOOL_SERVER_PATH,
)

def build_mcp_tool_manifest(tools):
    """Translate an OpenAI `tools` array into an MCP tool manifest (list of
    {"name", "description", "inputSchema"}). Returns None if any tool name
    fails MCP's naming constraint -- callers should fall back to
    render_tools_into_system_prompt() in that case rather than silently
    dropping a tool the caller asked for."""
    if not tools:
        return None
    manifest = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name")
        if not name or not _MCP_TOOL_NAME_RE.match(name):
            return None
        manifest.append({
            "name": name,
            "description": fn.get("description", "") or "",
            "inputSchema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return manifest


def build_mcp_tool_config(tools):
    """Returns a dict {"mcp_config", "manifest_path", "allowed_tools"} for
    the given OpenAI tools array, or None if tools couldn't be represented
    as an MCP manifest (see build_mcp_tool_manifest). allowed_tools holds
    the "mcp__<server>__<name>" forms Claude Code reports tool_use blocks
    under -- callers must strip this prefix again before handing the name
    back to an OpenAI client (see strip_mcp_tool_prefix). The manifest is
    written to a per-call temp file (not inlined into the mcp_config JSON)
    so the *server command* stays byte-identical across calls -- only the
    file path changes if the tool set itself changes, which keeps prompt
    caching intact whenever the actual tool set is stable across a
    resumed session (see README \"MCP native tool registration\" for the
    caching mechanics verified live)."""
    manifest = build_mcp_tool_manifest(tools)
    if manifest is None:
        return None
    fd, manifest_path = tempfile.mkstemp(prefix="claudecode-as-openai-mcp-manifest-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f, sort_keys=True)
    mcp_config = {
        "mcpServers": {
            _MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": [_MCP_TOOL_SERVER_PATH, manifest_path],
            }
        }
    }
    allowed = [f"mcp__{_MCP_SERVER_NAME}__{t['name']}" for t in manifest]
    return {"mcp_config": mcp_config, "manifest_path": manifest_path, "allowed_tools": allowed}


_MCP_TOOL_PREFIX = f"mcp__{_MCP_SERVER_NAME}__"


def strip_mcp_tool_prefix(name):
    """Reverses the "mcp__<server>__" mangling Claude Code applies to MCP
    tool names in tool_use blocks, so the name reported back to an OpenAI
    client matches exactly what the caller originally declared."""
    if name and name.startswith(_MCP_TOOL_PREFIX):
        return name[len(_MCP_TOOL_PREFIX):]
    return name


# ---------------------------------------------------------------------------
# Session caching: fingerprint a conversation and track how much of it has
# already been synced to a Claude Code session, so a continuing conversation
# can send only the new trailing messages via --resume instead of the full
# history every time. Verified live: a resumed turn re-processes only the
# delta (a few hundred tokens) instead of the whole conversation.
# ---------------------------------------------------------------------------
_SESSION_LOCK = threading.Lock()
_SESSION_STORE = {}  # conv_key -> {"claude_session_id": str, "synced_messages": list}
_SESSION_STORE_MAX = 200


