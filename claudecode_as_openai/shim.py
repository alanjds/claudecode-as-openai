#!/usr/bin/env python3
"""OpenAI-chat-completions-compatible shim over the local Claude Code CLI,
built the way Cline's "Claude Code" provider actually does it (verified
against cline/cline's open-source ClaudeCodeHandler + runClaudeCode,
2026-08-12) rather than the ad-hoc JSON-envelope-in-prose approach this
file used previously.

See README.md "Capability audit" for the full, empirically-verified list
of what works, what's approximated, and what's a genuine CLI limitation.
This file implements everything found to be implementable as of
2026-08-13: session caching (--session-id/--resume), OpenAI-shaped error
translation, max_tokens (via CLAUDE_CODE_MAX_OUTPUT_TOKENS), response_format
(via --json-schema), a tighter MCP-tool lockdown for the no-tools-requested
case, tool_choice: "none", n (bounded fan-out), and stop sequences
(client-side truncation).

Key differences from the original ad-hoc approach, and why they fix the
flakiness:

1. Uses Claude Code's REAL native tool-calling protocol
   (`--output-format stream-json`, parsing genuine `tool_use` content
   blocks from the assistant message) instead of asking Claude to emit a
   custom JSON envelope in plain text.

2. Uses `--disallowedTools <extended built-in list>` (not `--tools ""`)
   when tools were requested, so Claude Code's own built-ins are blocked
   by name but ANY custom tool names the caller passes remain callable.
   When NO tools were requested, uses `--tools "" --strict-mcp-config
   --mcp-config '{"mcpServers":{}}'` for a fully locked-down zero-tool
   session -- see "MCP tool leakage" in README for why both flags are
   needed (--disallowedTools alone does not block a machine's locally
   configured MCP servers).

3. Parses the NDJSON stream incrementally, skipping `thinking` blocks
   until a real `text`/`tool_use` block appears (an assistant turn can
   consist of ONLY a thinking block before the real content arrives on a
   later NDJSON line).

4. Session caching: fingerprints each conversation (system prompt + first
   message) and tracks how much of it has already been synced to a given
   Claude Code session. A continuing conversation sends only the new
   trailing messages via `--resume <id>`; a new or diverged conversation
   starts fresh via `--session-id <id>` with the full history. Verified
   directly: a resumed turn re-processes only the delta (a few hundred
   tokens) instead of the whole conversation (tens of thousands).

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLAUDE_BIN = "claude"
DEFAULT_MODEL = "sonnet"
KNOWN_MODEL_ALIASES = [
    # Hardcoded last-resort fallback for /v1/models when neither OAuth nor
    # an API key is available to query the real Anthropic /v1/models
    # endpoint (see fetch_model_list()). Extracted from strings embedded
    # in the compiled `claude` binary (2026-08-14) -- will go stale as new
    # models ship, which is exactly why the live query is preferred.
    "sonnet", "opus", "haiku",
]
CLAUDE_TIMEOUT_S = 300

# OpenRouter model-slug compatibility (2026-08-14). OpenRouter
# (https://openrouter.ai) is a widely-used OpenAI-compatible proxy in
# front of many providers, and its Anthropic model naming convention
# (`anthropic/claude-sonnet-4.5`, `~anthropic/claude-sonnet-latest`,
# etc -- see https://openrouter.ai/~anthropic/claude-sonnet-latest) is
# what a lot of existing OpenAI-shaped tooling already expects to send
# as the `model` field. Claude Code's own `--model` flag uses a
# different convention (`claude-sonnet-4-5`, bare `sonnet`/`opus`/
# `haiku` aliases that resolve to "latest") -- verified live via `claude
# --model <x> -p` with a "state your exact model version" probe prompt
# for each transform below. normalize_model_name() translates the
# former into the latter so a client pointed at this shim with an
# OpenRouter-style model string just works, without the caller needing
# to know Claude Code's own naming scheme.
_OPENROUTER_LATEST_ALIAS_RE = re.compile(r"^claude-(sonnet|opus|haiku)-latest$")
_OPENROUTER_VERSION_RE = re.compile(r"^claude-(sonnet|opus|haiku)-(\d+)\.(\d+)$")
_OPENROUTER_FAST_SUFFIX_RE = re.compile(r"^(claude-(?:sonnet|opus|haiku)-[\d.]+)-fast$")
_warned_fast_models = set()


def normalize_model_name(model):
    """Translate an OpenRouter-style Anthropic model slug into the
    equivalent Claude Code `--model` value. Verified live against the
    real `claude` CLI for each transform:

    - `~anthropic/claude-sonnet-latest`, `anthropic/claude-sonnet-latest`,
      `claude-sonnet-latest` -> `sonnet` (bare alias; Claude Code itself
      resolves this to whatever is currently "latest", verified live to
      return `claude-sonnet-4-6` at time of testing). Same for
      opus/haiku.
    - `anthropic/claude-sonnet-4.5`, `claude-sonnet-4.5` ->
      `claude-sonnet-4-5` (dot-to-dash in the version number; verified
      live -- Claude Code's own `--model` flag rejects the dotted form
      outright with "may not exist or you may not have access to it",
      but accepts the dashed form and echoes back the exact model it
      picked).
    - A `-fast` suffix (OpenRouter's Fast-mode variant naming, e.g.
      `anthropic/claude-opus-4.8-fast`) is stripped with a one-time
      stderr warning -- Claude Code's `-p` mode has no reachable
      fast-mode equivalent to route to, so this degrades to the normal
      (non-fast) model rather than erroring the whole request.
    - Anything else (already-native Claude Code model strings, bare
      `sonnet`/`opus`/`haiku`, full dated IDs like
      `claude-sonnet-4-5-20250929`) passes through unchanged.
    """
    if not model:
        return model
    normalized = model.strip()
    if normalized.startswith("~"):
        normalized = normalized[1:]
    if normalized.startswith("anthropic/"):
        normalized = normalized[len("anthropic/"):]

    latest_match = _OPENROUTER_LATEST_ALIAS_RE.match(normalized)
    if latest_match:
        return latest_match.group(1)

    fast_match = _OPENROUTER_FAST_SUFFIX_RE.match(normalized)
    if fast_match:
        if model not in _warned_fast_models:
            _warned_fast_models.add(model)
            sys.stderr.write(
                f"claudecode-as-openai: warning: model '{model}' requests OpenRouter's "
                f"Fast-mode variant, which has no -p-mode equivalent in Claude Code -- "
                f"falling back to the normal-speed model.\n"
            )
        normalized = fast_match.group(1)

    version_match = _OPENROUTER_VERSION_RE.match(normalized)
    if version_match:
        family, major, minor = version_match.groups()
        return f"claude-{family}-{major}-{minor}"

    return normalized


# OpenRouter reasoning-tokens compatibility (2026-08-14). OpenRouter's
# `reasoning` request parameter (https://openrouter.ai/docs/guides/best-
# practices/reasoning-tokens) is how OpenAI-shaped clients ask a model to
# think step-by-step and get that thinking back. Claude Code's own
# equivalent is `--effort <low|medium|high|max>` -- verified live (via a
# stream-json capture with a math prompt) that setting `--effort` genuinely
# produces real `thinking` content blocks (with a `signature` field) ahead
# of the final `text` block, not just a cosmetic flag.
#
# `--effort` only accepts exactly low/medium/high/max (verified live:
# "none", "minimal", and "xhigh" are all rejected outright with "argument
# ... is invalid. It must be one of: low, medium, high, max") -- so
# OpenRouter's wider effort vocabulary is clamped down to the nearest
# accepted value rather than passed through and erroring the whole
# request.
_REASONING_EFFORT_MAP = {
    "none": None,        # reasoning explicitly disabled -- no --effort flag at all
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def resolve_reasoning_effort(payload):
    """Extract Claude Code's `--effort` value (or None for "no reasoning
    requested") from an OpenAI/OpenRouter-shaped request payload.

    Supports both of OpenRouter's real request shapes for this
    (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens):
    - `{"reasoning": {"effort": "high"}}` -- effort string, clamped via
      _REASONING_EFFORT_MAP.
    - `{"reasoning": {"max_tokens": N}}` -- Anthropic-style token budget;
      approximated to an effort level using OpenRouter's own documented
      percentage bands (each effort level's stated % of some overall
      token budget), since Claude Code's -p mode has no raw token-budget
      equivalent to pass through directly.
    - `{"reasoning": {"enabled": true}}` alone (no effort/max_tokens) ->
      "medium", matching OpenRouter's own documented default.
    - `{"reasoning": {"exclude": true}}` -- accepted but NOT enforced:
      Claude Code has no mechanism to reason internally while witholding
      the thinking blocks from the transcript, so this shim's reasoning
      is always returned when requested (see build_reasoning_details()).

    Returns None when no `reasoning` key is present, or when
    `reasoning.effort` is explicitly "none".
    """
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, dict):
        return None
    if reasoning.get("enabled") is False:
        return None
    effort = reasoning.get("effort")
    if effort is not None:
        return _REASONING_EFFORT_MAP.get(str(effort).lower(), "medium")
    max_tokens = reasoning.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and max_tokens > 0:
        # OpenRouter's own documented effort/token-budget percentage
        # bands (see reasoning-tokens docs): low ~20%, medium ~50%,
        # high ~80%, max ~95% of an overall budget. Without a concrete
        # overall budget to divide by, treat the raw token count itself
        # against the same band thresholds as a reasonable proxy --
        # this is a best-effort approximation, not an exact mapping,
        # since Claude Code has no raw reasoning-token-budget flag.
        if max_tokens >= 8000:
            return "max"
        if max_tokens >= 4000:
            return "high"
        if max_tokens >= 1000:
            return "medium"
        return "low"
    if reasoning.get("enabled"):
        return "medium"
    return None


# Claude Code's OAuth credentials (subscription auth, NOT an API key) --
# used to authenticate the real Anthropic /v1/models call in
# fetch_model_list() so /v1/models can return the actual current model
# list instead of a hardcoded guess, without requiring the caller to have
# a separate ANTHROPIC_API_KEY. Verified live: a Bearer token read from
# this file successfully authenticated a GET to
# https://api.anthropic.com/v1/models and returned the real, current
# model list (10 entries, live 2026-08-14) -- this is a free metadata
# call, not a billed completion, so it doesn't touch subscription usage.
_CLAUDE_CREDENTIALS_PATH = os.path.expanduser("~/.claude/.credentials.json")
_MODEL_LIST_CACHE_TTL_S = 300
_model_list_cache = {"data": None, "fetched_at": 0.0}
_model_list_cache_lock = threading.Lock()

# Every spawned `claude` subprocess inherits whatever directory the shim
# process happens to be running from -- verified live: Claude Code reports
# that directory as its `cwd` and can reference/read real files there
# (confirmed by asking a tool_choice:"none" request "any tool you have" and
# getting back fabricated-looking but suspiciously specific Read/Glob
# references to this very shim's own source files). Spawning every `claude`
# call from a dedicated, empty, per-run temp directory instead closes this:
# verified `claude`'s own reported `cwd` matches the sandbox dir, not the
# shim's real working directory, once this is passed as subprocess cwd=.
_CLAUDE_CWD = tempfile.mkdtemp(prefix="claudecode-as-openai-sandbox-")
MAX_N_CHOICES = 5

# Claude Code's built-in tool names (as of CLI 2.1.94, checked 2026-08-13),
# PLUS the MCP-adjacent helper tools that also leak custom-tool dispatch
# surface: RemoteTrigger (a generic dispatcher that can invoke ANY declared
# custom tool by name even when not in the caller's tools array -- found
# live, see README "MCP tool leakage"), and ListMcpResourcesTool/
# ReadMcpResourceTool (MCP resource browsers, not server-specific so
# --strict-mcp-config doesn't touch them). This does NOT fully close the
# leak -- a machine's actual configured MCP tools (mcp__servername__tool)
# can still rarely fire when tools ARE requested (--strict-mcp-config
# can't be combined with real tool dispatch without reintroducing the
# --tools "" text-fallback problem). When NO tools are requested, the
# no-tools path below (--tools "" --strict-mcp-config) closes this
# completely instead.
EXTENDED_DISALLOWED_TOOLS = ",".join(
    [
        "Task", "TaskOutput", "Bash", "Glob", "Grep", "Read", "Edit", "Write",
        "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch", "TaskStop",
        "AskUserQuestion", "Skill", "EnterPlanMode", "ExitPlanMode",
        "EnterWorktree", "ExitWorktree", "CronCreate", "CronDelete",
        "CronList", "ToolSearch", "RemoteTrigger", "ListMcpResourcesTool",
        "ReadMcpResourceTool",
    ]
)


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


def _flatten_content(content):
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return content or ""


def build_claude_messages(openai_messages):
    """Translate a list of OpenAI chat messages into Claude Code's native
    message array shape: [{"role": "user"|"assistant", "content": <str or
    blocks>}]. System messages are skipped here (handled separately by the
    caller via system_prompt_from_messages) since Claude Code takes the
    system prompt as a separate CLI flag, not as an array element."""
    claude_messages = []

    for m in openai_messages:
        role = m.get("role", "user")
        if role == "system":
            continue

        if role == "tool":
            claude_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id", ""),
                            "content": _flatten_content(m.get("content")),
                        }
                    ],
                }
            )
            continue

        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            content_blocks = []
            text = _flatten_content(m.get("content"))
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:20]}"),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            claude_messages.append(
                {"role": "assistant", "content": content_blocks or text}
            )
            continue

        # user (or anything else) -> plain user message
        claude_messages.append({"role": "user", "content": _flatten_content(m.get("content"))})

    return claude_messages


def system_prompt_from_messages(openai_messages):
    parts = [_flatten_content(m.get("content")) for m in openai_messages if m.get("role") == "system"]
    return "\n\n".join(p for p in parts if p)


def _read_claude_oauth_token():
    """Read Claude Code's own OAuth access token from its local
    credentials file (subscription auth, NOT an API key -- this is the
    same token `claude` itself uses day-to-day). Returns the token string,
    or None if the file is missing, unreadable, malformed, or the token
    has already expired (checked against the same `expiresAt` field
    Claude Code itself uses, so an expired-but-present token doesn't get
    used to make a doomed request). Never raises -- every failure mode
    here should fall through to the next auth method in
    fetch_model_list(), not blow up /v1/models."""
    try:
        with open(_CLAUDE_CREDENTIALS_PATH, "r") as f:
            creds = json.load(f)
        oauth = creds.get("claudeAiOauth") or {}
        token = oauth.get("accessToken")
        expires_at = oauth.get("expiresAt")
        if not token:
            return None
        if expires_at and time.time() * 1000 > expires_at:
            return None
        return token
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _fetch_models_from_anthropic_api(auth_header):
    """GET https://api.anthropic.com/v1/models with the given auth header
    dict merged into the request. This is a free metadata call (not a
    billed completion) -- verified live with both an OAuth Bearer token
    and a real ANTHROPIC_API_KEY, each returning the current real model
    list (10 entries, live 2026-08-14). Returns the parsed `data` list, or
    None on any failure (network, auth, non-200, malformed JSON) so the
    caller can fall through to the next auth method or the hardcoded
    fallback -- this must never raise up into a /v1/models request."""
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/models",
        headers={"anthropic-version": "2023-06-01", **auth_header},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return None
            body = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
        return None
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return None
    return data


def fetch_model_list():
    """Returns the real, current Anthropic model list for /v1/models,
    preferring OAuth (Claude Code's own subscription credentials) over an
    API key over a hardcoded static fallback, per user direction
    (2026-08-14): OAuth first because it works without requiring the
    caller to separately configure ANTHROPIC_API_KEY -- the whole point of
    this shim is to avoid needing metered API billing, so authenticating
    this one free metadata call with the OAuth token Claude Code already
    has (rather than a paid-API-key-only path) keeps that promise intact.

    Order tried:
    1. Claude Code's own OAuth access token (~/.claude/.credentials.json)
       -- verified live: works identically to an API key against the real
       /v1/models endpoint.
    2. ANTHROPIC_API_KEY from the environment, if OAuth is absent/expired
       /unreadable.
    3. KNOWN_MODEL_ALIASES (hardcoded, extracted from the claude binary)
       if neither auth method is available or the request fails for any
       reason (network down, revoked token, etc).

    Results are cached in-process for _MODEL_LIST_CACHE_TTL_S to avoid a
    network round-trip on every single /v1/models poll (some clients poll
    this endpoint frequently on startup).

    Returns a list of dicts shaped like OpenAI's model list entries:
    [{"id": ..., "object": "model"}, ...]. Never raises."""
    with _model_list_cache_lock:
        cached = _model_list_cache["data"]
        if cached is not None and (time.time() - _model_list_cache["fetched_at"]) < _MODEL_LIST_CACHE_TTL_S:
            return cached

    data = None
    oauth_token = _read_claude_oauth_token()
    if oauth_token:
        data = _fetch_models_from_anthropic_api({"Authorization": f"Bearer {oauth_token}"})

    if data is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            data = _fetch_models_from_anthropic_api({"x-api-key": api_key})

    if data is not None:
        result = [{"id": m["id"], "object": "model"} for m in data if m.get("id")]
    else:
        result = [{"id": m, "object": "model"} for m in KNOWN_MODEL_ALIASES]

    with _model_list_cache_lock:
        _model_list_cache["data"] = result
        _model_list_cache["fetched_at"] = time.time()
    return result


def render_tools_into_system_prompt(tools, base_system_prompt):
    """LEGACY fallback path. Superseded by build_mcp_tool_config() (see
    below) as of 2026-08-14: custom tools are now passed to Claude Code as
    real MCP tool schemas, not prose. Kept and still wired up as an
    automatic fallback for edge cases where MCP registration itself can't
    be used (e.g. a tool name that can't be made into a valid MCP tool
    name at all) -- see call_claude_streaming's use of this function only
    when build_mcp_tool_config() returns None.

    Framing matters a lot here: describing tools as "custom"/hypothetical
    made Claude hedge and refuse to call them (0/8 in testing) even with
    native tool_use active. Framing them as already-wired, real,
    implemented-by-the-harness tools fixed this completely (6/6) --
    combined with the extended --disallowedTools list removing Claude's
    own competing built-in dispatch options. This path still only reaches
    ~40% single-shot reliability (see TOOL_CALL_MAX_RETRIES); the MCP path
    is the real fix."""
    if not tools:
        return base_system_prompt
    lines = [
        "<tools>",
        "You have access to the following tools. They ARE implemented and "
        "wired up by the calling harness -- call them directly via the "
        "standard tool-use mechanism whenever relevant. Do not describe or "
        "narrate a call, and do not claim a tool isn't available or isn't "
        "backed by a real implementation; just call it.",
        "",
    ]
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        lines.append(f"Tool: {name}")
        lines.append(f"Description: {desc}")
        lines.append(f"Input schema: {json.dumps(params)}")
        lines.append("")
    lines.append("</tools>")
    tool_desc = "\n".join(lines)
    if base_system_prompt:
        return base_system_prompt + "\n\n" + tool_desc
    return tool_desc


# ---------------------------------------------------------------------------
# Native MCP tool registration: the real fix for the ~40% single-shot
# tool-call reliability problem. Verified live (2026-08-14): Claude Code's
# `-p` mode has no flag to accept arbitrary JSON-schema tool definitions
# directly, but MCP tool schemas registered via `--mcp-config` ARE passed
# to the underlying Anthropic API as genuine, ajv-validated `tools`
# entries -- with `_meta["anthropic/alwaysLoad"] = True` skipping Claude
# Code's internal "ToolSearch" deferred-loading indirection, this reaches
# 100% turn-1 dispatch reliability across repeated trials (8/8, then
# 5/5), vs ~40% for the prose-based fallback above.
#
# Transport is stdio, not HTTP/SSE -- deliberately, per explicit user
# request: this must never bind a TCP port (no port-clash risk, nothing
# exposed on any network interface). The MCP server is a plain child
# process of `claude`, wired via stdio pipes only.
#
# MCP tool names must match ^[a-zA-Z0-9_-]{1,64}$ (same constraint as
# OpenAI function names) -- checked defensively since a caller could still
# send something invalid; falls back to the prose path if so.
_MCP_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MCP_SERVER_NAME = "shim_tools"
# Absolute path, not "-m claudecode_as_openai.mcp_tool_server" -- the
# `claude` subprocess runs with cwd=_CLAUDE_CWD (an empty sandbox temp
# dir, see _CLAUDE_CWD docstring), which is NOT on sys.path, so `-m`
# resolution fails there with "No module named 'claudecode_as_openai'"
# (verified live: reproduced the exact ModuleNotFoundError this way).
# The absolute file path works from any cwd.
_MCP_TOOL_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_tool_server.py")


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
            if len(openai_messages) > len(synced) and _messages_equal(openai_messages[: len(synced)], synced):
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


def _iter_ndjson_lines(proc):
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


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
    CLI; there is no flag to force unbuffered/line-buffered stdout."""
    deadline = time.time() + timeout_s
    buf = b""
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        try:
            ready, _, _ = select.select([master_fd], [], [], min(remaining, 1.0))
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
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _build_claude_cmd(model, session_mode, session_id, tools_requested, max_turns, json_schema, want_partial_messages=False, mcp_tool_config=None, effort=None):
    cmd = [CLAUDE_BIN]
    # Exclude user/project/local settings.json entirely -- this is what
    # actually prevents locally-configured SessionStart/other hooks from
    # firing and injecting arbitrary extra context into every completion,
    # which a stateless API shim should never be silently subject to.
    # Verified live (2026-08-14): a real SessionStart hook configured in
    # ~/.claude/settings.json fired and injected a marker string into
    # model context on a plain call; with --setting-sources "" set, the
    # model explicitly confirmed no hook message was present. Verified
    # this does NOT break anything else that matters to this shim: OAuth
    # auth is untouched (auth reads from a separate credentials file, not
    # settings.json), tool_use dispatch still fires correctly, and
    # --session-id/--resume continuity still works. This was considered
    # as an alternative to --bare mode, which looked like a stronger
    # lockdown (also skips CLAUDE.md, plugin sync, LSP, etc) but was
    # ruled out because --bare strictly requires ANTHROPIC_API_KEY and
    # never reads OAuth/keychain (confirmed live: --bare fails with "Not
    # logged in" under pure OAuth auth, works fine with a real API key)
    # -- unacceptable since this shim's whole premise is riding the
    # user's Claude subscription, not metered API billing.
    # policySettings/flagSettings (enterprise-managed, not user-facing)
    # remain unaffected either way -- getEnabledSettingSources() always
    # includes those regardless of --setting-sources.
    cmd += ["--setting-sources", ""]
    if mcp_tool_config is not None:
        # Native MCP tool registration path (see build_mcp_tool_config
        # docstring): --strict-mcp-config restricts MCP servers to ONLY
        # the one just-built manifest server (closing the same
        # locally-configured-MCP leak the no-tools path below closes),
        # --allowedTools further restricts to exactly this session's
        # declared tool names (nothing else callable even within our own
        # server), and --disallowedTools still blocks every Claude Code
        # built-in by name. Verified live (2026-08-14): this combination
        # exposes ONLY the intended mcp__shim_tools__<name> entries in
        # the session's tool list -- no built-ins, no other leakage.
        cmd += [
            "--disallowedTools", EXTENDED_DISALLOWED_TOOLS,
            "--strict-mcp-config", "--mcp-config", json.dumps(mcp_tool_config["mcp_config"]),
            "--allowedTools", ",".join(mcp_tool_config["allowed_tools"]),
        ]
    elif tools_requested:
        cmd += ["--disallowedTools", EXTENDED_DISALLOWED_TOOLS]
    else:
        # No tools requested at all: fully lock down BOTH built-ins and any
        # locally-configured MCP servers. See EXTENDED_DISALLOWED_TOOLS
        # docstring and README "MCP tool leakage" for why this is the only
        # combination that closes the leak completely (verified: tools
        # available == [] with this combo, vs. a real MCP tool call leaking
        # through unprompted without it).
        cmd += ["--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    cmd += ["--output-format", "stream-json", "--verbose", "--max-turns", str(max_turns)]
    if want_partial_messages:
        # Requested when a `stop` sequence is set (for early termination)
        # and/or when real token-level client streaming is active. This
        # adds token-level "stream_event"/"content_block_delta" lines
        # interleaved with the existing full "assistant" message chunks
        # (verified live: the "assistant" chunks stay complete/final,
        # identical to without this flag -- it only ADDS extra lines, it
        # doesn't change existing parsing). Needed because Claude Code's
        # default stream-json granularity is per-MESSAGE, not per-token:
        # without this flag, a stop sequence can only be detected after an
        # entire text block has already been fully generated and billed
        # (defeating the purpose of an early stop), and a client asking
        # for `stream: true` would get one giant content delta instead of
        # a real typing effect.
        cmd += ["--include-partial-messages"]
    if session_mode == "resume":
        cmd += ["--resume", session_id]
    else:
        cmd += ["--session-id", session_id]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    if json_schema is not None:
        cmd += ["--json-schema", json.dumps(json_schema)]
    cmd += ["-p"]
    return cmd


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

    `stop`, if given, is checked against the ACCUMULATED text after every
    text content block arrives, not just once at the end -- Claude Code has
    no native stop-sequence flag (verified: no such CLI option exists), but
    since stream-json delivers assistant messages incrementally, killing
    the subprocess the instant a stop sequence appears avoids paying for
    (and waiting on) the rest of a response nobody asked for. This is a
    real latency/cost win over truncating client-side only after the full
    response completes, which is what an earlier version of this shim did.

    `stream_callback`, if given, is called with each real token-level text
    chunk (str) as it arrives via --include-partial-messages
    content_block_delta events -- this is what delivers genuine real-time
    streaming to an OpenAI client instead of one giant chunk after full
    generation. Only wired up by the caller for the single-choice,
    no-tools-requested, no-json_schema path (see _handle_chat_completion):
    with tool retry in play, an earlier failed attempt's narration text
    would otherwise get streamed to the client before the shim knows that
    attempt needs to be discarded and retried.

    `tools`, if given (the raw OpenAI `tools` array), is registered as a
    real MCP tool server (see build_mcp_tool_config) instead of being
    described in prose -- this is the fix for the ~40% single-shot
    tool-call reliability problem, verified live to reach 100% turn-1
    dispatch. Falls back to tools_requested's prose-description path
    automatically (system_prompt is expected to already carry the prose
    fallback in that case -- see _handle_chat_completion) when the tool
    set can't be represented as a valid MCP manifest (see
    build_mcp_tool_manifest).

    Returns a dict: {"text", "tool_calls", "usage", "finish_reason",
    "structured_json"}. Raises ClaudeCliError for conditions that should
    become an OpenAI-shaped error response (invalid model, auth failure,
    rate limiting, etc -- see _classify_error_text).

    Tool suppression: see EXTENDED_DISALLOWED_TOOLS and _build_claude_cmd
    docstrings for why --disallowedTools (not --tools "") is used when
    tools are requested, and why the no-tools path uses --tools "" +
    --strict-mcp-config instead."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    stop_sequences = []
    if stop:
        stop_sequences = [stop] if isinstance(stop, str) else [s for s in stop if s]

    system_prompt_file = None
    mcp_tool_config = build_mcp_tool_config(tools) if tools else None
    # --json-schema needs a couple of internal turns (an internal
    # "StructuredOutput" tool call + a corrective retry if the model
    # forgets it) to actually enforce the schema -- verified empirically:
    # max_turns=1 leaves it hanging at error_max_turns with the schema
    # never actually produced; max_turns=3 completes cleanly. Native MCP
    # tool dispatch (verified live) also needs 2 turns when a tool fires:
    # turn 1 emits the tool_use (ending that turn), turn 2 is needed for
    # the follow-up text after the (client-side) tool result -- though
    # this shim breaks out on tool_use before that matters (see below),
    # --max-turns 1 alone left error_max_turns on the wire in earlier
    # testing when a fallback path needed the second turn, so this stays
    # >= 2 whenever any real tool dispatch is possible.
    max_turns = 3 if json_schema is not None else (2 if mcp_tool_config else 1)
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

        text_parts = []
        reasoning_parts = []
        reasoning_signature = None
        tool_calls = []
        structured_json = None
        usage = {}
        finish_reason = "stop"
        stop_matched = False
        streaming_partial_text = ""
        deadline = time.time() + CLAUDE_TIMEOUT_S

        chunk_source = (
            _iter_ndjson_lines_pty(pty_master_fd, proc, CLAUDE_TIMEOUT_S)
            if use_pty else _iter_ndjson_lines(proc)
        )
        for chunk in chunk_source:
            if time.time() > deadline:
                break
            ctype = chunk.get("type")

            if ctype == "stream_event" and (stop_sequences or stream_callback):
                # Only present when --include-partial-messages was passed
                # (see _build_claude_cmd), which happens when a `stop`
                # sequence is set and/or a stream_callback was provided.
                # Real token-level deltas -- verified live these interleave
                # with, and arrive BEFORE, the full "assistant" chunk for
                # the same content block: content_block_start ->
                # content_block_delta(s) -> the full "assistant" chunk ->
                # content_block_stop -> next content_block_start. That
                # ordering is what makes resetting the per-block
                # accumulator on content_block_start safe: the prior
                # block's text has already been folded into text_parts by
                # its "assistant" chunk before the next block starts.
                #
                # This is the actual fix for a real stop sequence only
                # being detectable after an entire text block finished
                # generating (and got billed): now a match can be caught
                # mid-block, at real token granularity. It's also what
                # delivers real client-facing streaming when
                # stream_callback is set: each text_delta is forwarded to
                # the caller immediately, as it's generated, instead of
                # being held until the whole response completes.
                event = chunk.get("event", {})
                etype = event.get("type")
                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        new_text = delta["text"]
                        streaming_partial_text += new_text
                        if stop_sequences:
                            combined = "".join(text_parts) + streaming_partial_text
                            matched_text, matched = _apply_stop_sequences(combined, stop_sequences)
                            if matched:
                                # Truncate what we forward to the client too
                                # -- stream only the portion up to the stop
                                # match, never the text past it.
                                already_streamed_len = len(combined) - len(new_text)
                                visible_new_text = matched_text[already_streamed_len:]
                                if stream_callback and visible_new_text:
                                    stream_callback(visible_new_text)
                                text_parts = [matched_text]
                                stop_matched = True
                                finish_reason = "stop"
                                break
                        if stream_callback:
                            stream_callback(new_text)
                elif etype == "content_block_start":
                    streaming_partial_text = ""
                continue

            if ctype == "assistant" and "message" in chunk:
                message = chunk["message"]
                # NOTE: "error" is a sibling key of "message" at the chunk
                # level (chunk["error"]), NOT inside message itself --
                # verified against live captures for both an invalid model
                # ({"type":"assistant","message":{...},"error":"invalid_request"})
                # and an output-token-cap breach ({"error":"max_output_tokens"}).
                err = chunk.get("error")
                if err:
                    text = _flatten_content(message.get("content")) or ""
                    if err == "max_output_tokens":
                        # Claude Code hard-fails on exceeding the output
                        # cap rather than truncating like OpenAI's
                        # max_tokens does. Closest honest mapping: report
                        # finish_reason="length" with whatever text (if
                        # any) had already accumulated -- there usually
                        # isn't any, since the cap is enforced before the
                        # text block completes (verified empirically).
                        finish_reason = "length"
                        break
                    status, etype, code = _classify_error_text(text)
                    raise ClaudeCliError(status, etype, text or f"claude reported error: {err}", code=code)
                got_tool_use = False
                for content in message.get("content", []):
                    ctype2 = content.get("type")
                    if ctype2 == "thinking" and content.get("thinking"):
                        # Real extended-thinking output, only present when
                        # --effort was passed (see resolve_reasoning_effort
                        # in _handle_chat_completion) -- verified live via
                        # a stream-json capture with a math prompt: setting
                        # --effort genuinely produces this block (with a
                        # real `signature` field) ahead of the final `text`
                        # block, not a cosmetic no-op. Surfaced back to the
                        # caller as OpenRouter's `reasoning`/
                        # `reasoning_details` shape -- see
                        # build_reasoning_details().
                        reasoning_parts.append(content["thinking"])
                        if content.get("signature"):
                            reasoning_signature = content["signature"]
                        continue
                    if ctype2 == "text" and content.get("text"):
                        text_parts.append(content["text"])
                        # IMPORTANT: do NOT break here just because we saw
                        # text. Claude frequently narrates in one assistant
                        # NDJSON message ("Sure! Let me check...") and
                        # dispatches the actual tool_use in a SEPARATE,
                        # LATER assistant message within the same turn --
                        # verified live: a 3-message sequence of [thinking,
                        # text, tool_use] is common. Breaking on the
                        # text-only message (as an earlier version of this
                        # function did) silently swallowed the tool_use
                        # that followed 100% of the time in a 10/10
                        # regression run. Keep reading; only stop early on
                        # an actual tool_use/structured-output block below,
                        # a stop-sequence match (checked right here), an
                        # error, or the stream ending naturally (the
                        # "result" chunk case further down).
                        if stop_sequences:
                            accumulated = "".join(text_parts)
                            matched_text, stop_matched = _apply_stop_sequences(accumulated, stop_sequences)
                            if stop_matched:
                                text_parts = [matched_text]
                                finish_reason = "stop"
                                break
                    elif ctype2 == "tool_use":
                        name = content.get("name", "")
                        if name == "StructuredOutput":
                            # --json-schema enforcement mechanism: the
                            # actual answer is the tool's input, not a
                            # real tool call to expose to the caller.
                            structured_json = content.get("input", {})
                            got_tool_use = True
                        else:
                            tool_calls.append(
                                {
                                    "id": content.get("id", f"toolu_{uuid.uuid4().hex[:20]}"),
                                    "name": strip_mcp_tool_prefix(name),
                                    "input": content.get("input", {}),
                                }
                            )
                            got_tool_use = True
                    # "redacted_thinking" blocks (safety-redacted reasoning,
                    # content deliberately withheld by Anthropic) are still
                    # intentionally ignored -- there is no real content to
                    # surface. Real "thinking" blocks are captured above.
                if usage_data := message.get("usage"):
                    usage = usage_data
                if got_tool_use or stop_matched:
                    # A stop-sequence match ends the response right here --
                    # `finally` below tears down the subprocess immediately,
                    # which is the actual latency/cost win over letting
                    # `claude` keep generating a response nobody will see.
                    break
                continue

            if ctype == "result":
                if chunk.get("is_error") and chunk.get("subtype") not in ("error_max_turns",):
                    text = chunk.get("result") or f"claude error: {chunk.get('subtype')}"
                    status, etype, code = _classify_error_text(text)
                    raise ClaudeCliError(status, etype, text, code=code or chunk.get("subtype"))
                if not usage and chunk.get("usage"):
                    usage = chunk["usage"]
                break

        if structured_json is not None:
            return {
                "text": json.dumps(structured_json),
                "tool_calls": [],
                "usage": usage,
                "finish_reason": "stop",
                "structured_json": structured_json,
                "reasoning": None,
                "reasoning_signature": None,
            }

        return {
            "text": "".join(text_parts) or None,
            "tool_calls": tool_calls,
            "usage": usage,
            "finish_reason": finish_reason,
            "structured_json": None,
            "stop_matched": stop_matched,
            "reasoning": "".join(reasoning_parts) or None,
            "reasoning_signature": reasoning_signature,
        }
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


def build_reasoning_details(reasoning_text, signature):
    """Build OpenRouter's `reasoning_details` array shape
    (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
    from a captured Claude `thinking` block. Returns None when there's no
    reasoning text (either --effort wasn't set, or the model didn't
    produce any), so callers can omit the field entirely rather than
    emit an empty array.

    Uses the `reasoning.text` detail type (the raw-text variant, per
    OpenRouter's documented schema) with `format: "anthropic-claude-v1"`,
    which is the exact format OpenRouter itself uses to tag real
    Anthropic thinking blocks. The `signature` field is Claude's own
    cryptographic signature over the thinking block (verified present on
    every live `thinking` block captured with --effort set) -- passed
    through as-is since it's an opaque per-block token, not something
    this shim can or should regenerate."""
    if not reasoning_text:
        return None
    return [
        {
            "type": "reasoning.text",
            "text": reasoning_text,
            "signature": signature,
            "format": "anthropic-claude-v1",
        }
    ]


def _build_openai_usage(usage_totals):
    """Reshape this shim's internal flat usage dict (real numbers, custom
    field names -- cache_read_input_tokens/cache_creation_input_tokens)
    into a superset that ALSO carries OpenAI/OpenRouter's actual nested
    `usage.prompt_tokens_details.cached_tokens` shape
    (https://platform.openai.com/docs/api-reference/chat/object), so
    clients that specifically parse that nested structure (several cost-
    tracking dashboards and proxies do) get real numbers instead of
    silently ignoring a custom field they don't recognize. The flat
    custom keys are kept too, unchanged, for backward compatibility with
    anything already reading them directly.

    `cached_tokens` maps from cache_read_input_tokens -- an actual
    served-from-cache count, which is exactly what OpenAI's field means.
    `completion_tokens_details.reasoning_tokens` is intentionally omitted:
    Claude Code's usage payload has no separate reasoning-token count
    (verified live -- the `usage` block's `output_tokens` already
    includes any thinking-block tokens, undifferentiated), so fabricating
    a split would be a made-up number, not a real one."""
    usage = dict(usage_totals)
    usage["prompt_tokens_details"] = {
        "cached_tokens": usage_totals.get("cache_read_input_tokens", 0),
    }
    return usage


# Bounded retries when tools were requested but Claude answered with plain
# text instead of a real tool_use block (--disallowedTools + strong framing
# gets ~40% single-shot native tool_use reliability, verified 6/15 over a
# fair sample on 2026-08-12). Retrying turns that into a much higher
# practical success rate at the cost of extra latency/spend only on the
# failing path.
#
# Parameters were tuned to match the retry decorator found in an OLD,
# now-deleted Cline commit (`@withRetry({maxRetries: 4, baseDelay: 2000,
# maxDelay: 15000})`). Verified against Cline's CURRENT mainline
# (2026-08-12): that file no longer exists -- Cline now delegates entirely
# to a third-party npm package with no retry decorator of its own. Checked
# the official @anthropic-ai/claude-agent-sdk too: it retries transport/API
# errors only, not "model narrated instead of dispatching a tool" -- no
# tool_choice:required-equivalent is exposed through the harness. No
# deterministic fix exists anywhere in this ecosystem for this failure
# mode; every implementation that handles it does so with a retry loop.
TOOL_CALL_MAX_RETRIES = 4
TOOL_CALL_BASE_DELAY_S = 2.0
TOOL_CALL_MAX_DELAY_S = 15.0


def _backoff_delay_s(attempt_index):
    """Exponential backoff: baseDelay * 2^attempt, capped at maxDelay.
    attempt_index is 0-based (0 = first retry)."""
    return min(TOOL_CALL_BASE_DELAY_S * (2 ** attempt_index), TOOL_CALL_MAX_DELAY_S)


def call_claude_with_tool_retry(
    delta_claude_messages,
    full_claude_messages,
    system_prompt,
    model,
    tools_requested,
    session_mode,
    session_id,
    tools=None,
    stop=None,
    stream_callback=None,
    effort=None,
):
    """Wraps call_claude_streaming with bounded retries for the documented
    tool-dispatch flakiness. The FIRST attempt uses whatever session mode
    was resolved by resolve_session() (a resumed session sending only the
    new delta messages, or a fresh session sending everything) so the
    common, successful case gets the full caching benefit. If a retry is
    needed (tools were requested but no tool_use came back), retries use a
    brand-new fresh session with the FULL conversation history instead of
    resuming -- reusing the same --session-id across attempts causes an
    "already in use" CLI error, and resuming with more delta messages would
    duplicate entries in that session's persisted transcript. Returns
    (result_dict, final_session_mode, final_session_id) so the caller knows
    which session to record for the next external turn.

    `tools`, if given, is forwarded to call_claude_streaming for native MCP
    tool registration (see build_mcp_tool_config) -- this is what actually
    fixes the ~40% single-shot reliability this retry loop exists to paper
    over; verified live to reach 100% turn-1 dispatch, so in practice this
    loop should rarely need more than its first attempt whenever `tools`
    resolves to a valid MCP manifest. Retries remain as a safety net for
    the prose-fallback path (tools that can't be represented as MCP tool
    names) and for any future Claude Code regression.

    `stream_callback`, if given, should only ever be passed by the caller
    when tools_requested is False -- with tools in play, a failed
    attempt's narration text must NOT reach the client before the shim
    knows to discard it and retry (see call_claude_streaming's own
    docstring). Enforced here defensively too: the callback is only ever
    forwarded to call_claude_streaming when tools_requested is False,
    regardless of what the caller passed in."""
    result = None
    cur_mode, cur_id, cur_messages = session_mode, session_id, delta_claude_messages
    total_attempts = 1 + TOOL_CALL_MAX_RETRIES
    for attempt in range(total_attempts):
        cb = stream_callback if not tools_requested else None
        result = call_claude_streaming(
            cur_messages, system_prompt, model,
            session_mode=cur_mode, session_id=cur_id,
            tools_requested=tools_requested,
            tools=tools,
            stop=stop,
            stream_callback=cb,
            effort=effort,
        )
        if result["tool_calls"] or not tools_requested:
            return result, cur_mode, cur_id
        if attempt < total_attempts - 1:
            time.sleep(_backoff_delay_s(attempt))
            cur_mode, cur_id, cur_messages = "fresh", str(uuid.uuid4()), full_claude_messages
    return result, cur_mode, cur_id


# Sampling/formatting parameters that have NO equivalent anywhere in the
# Claude Code CLI (checked `claude --help` and the official env-vars
# docs, 2026-08-13): no flag, no env var, nothing. `--effort
# <low|medium|high|max>` (Claude Code's reasoning-effort knob) IS now
# mapped, but only from OpenRouter's `reasoning` parameter (see
# resolve_reasoning_effort) -- a genuinely different axis (how much
# internal deliberation the model does) from output randomness/
# diversity, so it is NOT treated as a substitute for temperature/top_p
# below.
#
# These are silently accepted (never hard-errored) because real OpenAI
# clients routinely send explicit defaults on every request (e.g.
# `temperature: 1.0`, which IS the OpenAI default and carries no signal
# that the caller actually wants non-default sampling behavior) -- hard
# erroring on presence-of-key rather than meaningfully-different-value
# would break compatibility with a huge fraction of well-behaved clients
# for zero practical benefit. A stderr warning (not a client-visible
# error) is emitted once per parameter name observed, so operators running
# this shim can tell that a request asked for something it can't honor,
# without breaking the request.
_UNSUPPORTED_SAMPLING_PARAMS = (
    "temperature", "top_p", "seed", "logprobs", "top_logprobs",
    "presence_penalty", "frequency_penalty", "logit_bias",
)
_warned_sampling_params = set()


def _warn_unsupported_sampling_params(payload):
    """Emit a one-time-per-parameter-name stderr warning when a request
    includes a sampling parameter Claude Code has no way to honor. Does
    NOT reject the request -- see the module-level comment above
    _UNSUPPORTED_SAMPLING_PARAMS for why a hard error would be wrong
    here."""
    for name in _UNSUPPORTED_SAMPLING_PARAMS:
        if name in payload and payload[name] is not None and name not in _warned_sampling_params:
            _warned_sampling_params.add(name)
            sys.stderr.write(
                f"claudecode-as-openai: warning: '{name}' was requested but Claude "
                f"Code has no equivalent (no CLI flag, no env var) -- ignored, not "
                f"applied. See README \"Capability audit\".\n"
            )


def _apply_stop_sequences(text, stop):
    """Client-side emulation of OpenAI's `stop` parameter -- Claude Code
    has no native stop-sequence flag, so truncate the text ourselves at
    the earliest match. Returns (truncated_text, matched: bool)."""
    if not text or not stop:
        return text, False
    if isinstance(stop, str):
        stop = [stop]
    earliest = None
    for seq in stop:
        if not seq:
            continue
        idx = text.find(seq)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text, False
    return text[:earliest], True


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, err: ClaudeCliError):
        self._send_json(err.to_openai_body(), err.http_status)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(
                {
                    "object": "list",
                    "data": fetch_model_list(),
                }
            )
            return
        self._send_json({"status": "ok"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            self._send_error(ClaudeCliError(400, "invalid_request_error", f"Invalid JSON body: {e}"))
            return

        try:
            self._handle_chat_completion(payload)
        except ClaudeCliError as e:
            self._send_error(e)
        except Exception as e:
            self._send_error(ClaudeCliError(500, "api_error", str(e)))

    def _handle_chat_completion(self, payload):
        messages = payload.get("messages", [])
        tools = payload.get("tools")
        model = normalize_model_name(payload.get("model") or DEFAULT_MODEL)
        stream = bool(payload.get("stream"))
        tool_choice = payload.get("tool_choice")
        n = payload.get("n") or 1
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens")
        stop = payload.get("stop")
        response_format = payload.get("response_format") or {}
        effort = resolve_reasoning_effort(payload)

        if n > MAX_N_CHOICES:
            raise ClaudeCliError(
                400, "invalid_request_error",
                f"n={n} exceeds this proxy's limit of {MAX_N_CHOICES} (each choice is a "
                "separate full Claude Code subprocess call; unbounded n would be unbounded cost).",
                code="n_too_large", param="n",
            )

        # tool_choice: "none" is real and correctly implementable -- treat
        # exactly like "no tools requested" so the full MCP+built-in
        # lockdown applies and no tool description is added to the prompt.
        # tool_choice: "required" / forcing a specific function name has NO
        # reliable mechanism in this harness (no raw Anthropic tool_choice
        # equivalent exposed) -- accepted but NOT enforced, same as before.
        effective_tools = tools
        if tool_choice == "none":
            effective_tools = None

        json_schema = None
        if response_format.get("type") == "json_schema":
            json_schema = (response_format.get("json_schema") or {}).get("schema")
        elif response_format.get("type") == "json_object":
            json_schema = {"type": "object"}

        system_prompt = system_prompt_from_messages(messages)
        # Native MCP tool registration (see build_mcp_tool_config) is the
        # real fix for tool-call reliability and is always attempted first
        # inside call_claude_streaming/_run_one_completion. The prose
        # fallback below is only actually NEEDED when a tool name can't be
        # represented as a valid MCP tool name (see
        # build_mcp_tool_manifest) -- checked here up front so the prompt
        # doesn't carry redundant tool descriptions on the common path
        # where MCP registration succeeds (call_claude_streaming would
        # otherwise silently prefer MCP anyway, but doubling up the prose
        # description wastes prompt-cache-relevant tokens for no benefit).
        if effective_tools and build_mcp_tool_manifest(effective_tools) is None:
            system_prompt = render_tools_into_system_prompt(effective_tools, system_prompt)

        env_overrides = {}
        if max_tokens:
            try:
                env_overrides["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(max_tokens))
            except (TypeError, ValueError):
                pass

        _warn_unsupported_sampling_params(payload)

        # Real token-level streaming is only safe for the single-choice,
        # no-tools-requested, no-json_schema, no-reasoning path: with tool
        # retry in play, a failed attempt's narration text must not reach
        # the client before the shim knows to discard it and retry,
        # --json-schema's multi-turn corrective mechanism doesn't map
        # cleanly onto a single token stream either, and reasoning
        # (--effort) requires capturing the `thinking` block BEFORE the
        # `text` block starts (see build_reasoning_details), which
        # _handle_streaming_completion's stream_callback (text_delta only)
        # has no hook for. Every other combination falls back to the
        # existing buffered-then-emit behavior (SSE framing is still
        # correct, just not real-time).
        can_stream_live = stream and n == 1 and not effective_tools and json_schema is None and effort is None
        if can_stream_live:
            self._handle_streaming_completion(
                messages, model, system_prompt, max_tokens, env_overrides, stop,
            )
            return

        choices = []
        usage_totals = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        }

        for choice_index in range(n):
            if n == 1:
                # Single-choice path: use session caching (fingerprint +
                # resume-with-delta) for the common case.
                session_mode, session_id, delta_messages, conv_key = resolve_session(messages)
            else:
                # n>1 fan-out: each choice is an independent fresh
                # completion -- parallel-choice semantics don't fit clean
                # single-session continuity, so caching is skipped here.
                session_mode, session_id, delta_messages, conv_key = "fresh", str(uuid.uuid4()), messages, None

            delta_claude_messages = build_claude_messages(delta_messages)
            full_claude_messages = build_claude_messages(messages)

            result, final_mode, final_id = self._run_one_completion(
                delta_claude_messages, full_claude_messages, system_prompt, model,
                tools_requested=bool(effective_tools), session_mode=session_mode,
                session_id=session_id, json_schema=json_schema, env_overrides=env_overrides,
                tools=effective_tools, stop=stop, effort=effort,
            )

            text = result["text"]
            # The subprocess is now killed the instant a stop sequence
            # appears mid-stream (see call_claude_streaming's stop=
            # handling) -- result["stop_matched"] reflects that. This
            # second pass is now just a safety net for edge cases where
            # early termination didn't apply (e.g. the CLAUDE_TIMEOUT_S
            # deadline fired first, or --json-schema's structured_json
            # path bypasses stop-sequence checking entirely): idempotent
            # on already-truncated text, matches nothing, no-op.
            stop_matched = result.get("stop_matched", False)
            if stop and text and not stop_matched:
                text, stop_matched = _apply_stop_sequences(text, stop)

            raw_tool_calls = result["tool_calls"]
            tool_calls = None
            if raw_tool_calls:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])},
                    }
                    for tc in raw_tool_calls
                ]

            message = {"role": "assistant", "content": text}
            if tool_calls:
                message["tool_calls"] = tool_calls

            reasoning_details = build_reasoning_details(
                result.get("reasoning"), result.get("reasoning_signature"),
            )
            if reasoning_details:
                # OpenRouter's two supported shapes
                # (https://openrouter.ai/docs/guides/best-practices/
                # reasoning-tokens): `reasoning` (plaintext string) for
                # simple consumers, `reasoning_details` (structured array,
                # preserves the signature) for consumers that round-trip
                # reasoning back into a follow-up request. Both point at
                # the same captured thinking text.
                message["reasoning"] = result["reasoning"]
                message["reasoning_details"] = reasoning_details

            finish_reason = result["finish_reason"]
            if tool_calls:
                finish_reason = "tool_calls"
            elif stop_matched:
                finish_reason = "stop"

            choices.append({"index": choice_index, "message": message, "finish_reason": finish_reason})

            usage = result["usage"]
            usage_totals["prompt_tokens"] += usage.get("input_tokens", 0)
            usage_totals["completion_tokens"] += usage.get("output_tokens", 0)
            usage_totals["total_tokens"] += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            usage_totals["cache_read_input_tokens"] += usage.get("cache_read_input_tokens", 0)
            usage_totals["cache_creation_input_tokens"] += usage.get("cache_creation_input_tokens", 0)

            if n == 1 and conv_key is not None:
                record_session(conv_key, final_id, messages, message)

        if stream:
            self._send_stream_chunks(model, choices)
            return

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": _build_openai_usage(usage_totals),
        }
        self._send_json(response)

    def _handle_streaming_completion(self, messages, model, system_prompt, max_tokens, env_overrides, stop):
        """Real token-level SSE streaming for the safe case (single choice,
        no tools requested, no json_schema -- see the can_stream_live
        gate in _handle_chat_completion). Sends SSE headers immediately,
        then forwards each real text_delta from
        call_claude_streaming(stream_callback=...) to the client as it
        arrives, instead of buffering the full response first. Falls back
        to the same session-caching/resume logic as the buffered path."""
        session_mode, session_id, delta_messages, conv_key = resolve_session(messages)
        delta_claude_messages = build_claude_messages(delta_messages)

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        role_sent = False

        def emit(delta, finish=None):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def on_text_delta(text_piece):
            nonlocal role_sent
            if not role_sent:
                emit({"role": "assistant", "content": text_piece})
                role_sent = True
            else:
                emit({"content": text_piece})

        original_popen = subprocess.Popen
        if env_overrides:
            def scoped_popen(*args, **kwargs):
                env = dict(os.environ)
                env.update(env_overrides)
                kwargs["env"] = env
                return original_popen(*args, **kwargs)
            subprocess.Popen = scoped_popen

        try:
            result = call_claude_streaming(
                delta_claude_messages, system_prompt, model,
                session_mode=session_mode, session_id=session_id,
                tools_requested=False, stop=stop,
                stream_callback=on_text_delta,
            )
        except ClaudeCliError as e:
            # Headers are already sent by this point (SSE has to commit to
            # a 200 before any content is known) -- an OpenAI SSE client
            # expects an error surfaced as a final chunk, not a fresh HTTP
            # error status, since the status line is long gone.
            emit({"content": f"\n\n[error: {e.message}]"}, finish="stop")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        finally:
            if env_overrides:
                subprocess.Popen = original_popen

        if not role_sent:
            # No text ever arrived (e.g. immediate stop-sequence match on
            # an empty prefix, or a same-turn error): still send the role
            # delta so the client sees a well-formed message shape.
            emit({"role": "assistant"})

        finish_reason = result["finish_reason"]
        if result.get("stop_matched"):
            finish_reason = "stop"
        emit({}, finish=finish_reason)
        self.wfile.write(b"data: [DONE]\n\n")

        if conv_key is not None:
            message = {"role": "assistant", "content": result["text"]}
            record_session(conv_key, session_id, messages, message)

    def _run_one_completion(self, delta_messages, full_messages, system_prompt, model,
                             tools_requested, session_mode, session_id, json_schema, env_overrides,
                             tools=None, stop=None, effort=None):
        """Single entry point for producing one completion, regardless of
        whether env overrides (max_tokens), structured output (json_schema),
        and/or tool retry apply. Returns (result_dict, final_session_mode,
        final_session_id). Consolidating this here (rather than branching
        across several call sites) keeps the env-var scoping and retry
        logic each applied exactly once, in a well-defined order."""

        def _do_call(msgs, mode, sid):
            return call_claude_streaming(
                msgs, system_prompt, model,
                session_mode=mode, session_id=sid,
                tools_requested=tools_requested, tools=tools, json_schema=json_schema,
                stop=stop, effort=effort,
            )

        original_popen = subprocess.Popen
        if env_overrides:
            def scoped_popen(*args, **kwargs):
                env = dict(os.environ)
                env.update(env_overrides)
                kwargs["env"] = env
                return original_popen(*args, **kwargs)
            subprocess.Popen = scoped_popen

        try:
            if json_schema is not None:
                # Structured output doesn't go through the tool-call retry
                # loop -- --json-schema uses its own internal mechanism
                # (a StructuredOutput tool + corrective turn) that isn't
                # the "narrated instead of dispatching" failure mode the
                # retry loop targets.
                result = _do_call(delta_messages, session_mode, session_id)
                return result, session_mode, session_id
            return call_claude_with_tool_retry(
                delta_messages, full_messages, system_prompt, model,
                tools_requested=tools_requested, session_mode=session_mode, session_id=session_id,
                tools=tools, stop=stop, effort=effort,
            )
        finally:
            if env_overrides:
                subprocess.Popen = original_popen


    def _send_stream_chunks(self, model, choices):
        """Emits SSE chunks. NOTE: this is protocol-shaped streaming, not
        real token streaming -- each choice's full text/tool_calls are
        already fully computed by the time this runs (Claude Code's
        stream-json mode streams messages, not token deltas within a
        message), so each choice arrives as a single content delta after
        the full latency. See README "Capability audit" for measurements."""
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(index, delta, finish=None):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

        for c in choices:
            idx = c["index"]
            message = c["message"]
            emit(idx, {"role": "assistant"})
            if message.get("tool_calls"):
                for i, tc in enumerate(message["tool_calls"]):
                    emit(idx, {"tool_calls": [{"index": i, "id": tc["id"], "type": "function", "function": tc["function"]}]})
            elif message.get("content"):
                emit(idx, {"content": message["content"]})
            emit(idx, {}, finish=c["finish_reason"])
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8977
    except ValueError:
        print(f"Invalid port: {sys.argv[1]!r}", file=sys.stderr)
        sys.exit(2)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Claude Code shim (native tool-calling + session caching) listening on http://127.0.0.1:{port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
