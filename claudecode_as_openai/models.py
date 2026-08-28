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
from claudecode_as_openai.constants import DEFAULT_MODEL, _OPENROUTER_ALIASES
from claudecode_as_openai.errors import ClaudeCliError

def normalize_model_name(model):
    """Translate an OpenRouter-style Anthropic model slug into the
    equivalent Claude Code `--model` value. See CHANGELOG for the live
    verification behind each transform:

    - `~anthropic/claude-sonnet-latest`, `anthropic/claude-sonnet-latest`,
      `claude-sonnet-latest` -> `sonnet` (bare alias; same for opus/haiku).
    - `anthropic/claude-sonnet-4.5`, `claude-sonnet-4.5` ->
      `claude-sonnet-4-5` (dot-to-dash; Claude Code's `--model` flag
      rejects the dotted form outright).
    - A `-fast` suffix (OpenRouter's Fast-mode variant naming) is
      stripped with a one-time stderr warning -- Claude Code's `-p` mode
      has no reachable fast-mode equivalent, so this degrades to the
      normal (non-fast) model rather than erroring the whole request.
    - Anything else (already-native Claude Code model strings, bare
      aliases, full dated IDs) passes through unchanged.
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


# OpenRouter reasoning-tokens compatibility (see CHANGELOG). Maps
# OpenRouter's `reasoning` request parameter onto Claude Code's
# `--effort <low|medium|high|max>` flag, which genuinely produces real
# `thinking` content blocks. `--effort` only accepts exactly those four
# values, so OpenRouter's wider vocabulary is clamped to the nearest one
# rather than passed through and erroring the whole request.
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

    Supports OpenRouter's real request shapes
    (https://openrouter.ai/docs/guides/best-practices/reasoning-tokens):
    - `{"reasoning": {"effort": "high"}}` -- clamped via _REASONING_EFFORT_MAP.
    - `{"reasoning": {"max_tokens": N}}` -- approximated to an effort
      level using OpenRouter's documented percentage bands, since Claude
      Code has no raw token-budget equivalent.
    - `{"reasoning": {"enabled": true}}` alone -> "medium" (OpenRouter's
      own documented default).
    - `{"reasoning": {"exclude": true}}` -- accepted but NOT enforced:
      Claude Code has no way to reason internally while withholding the
      thinking block, so this shim's reasoning is always returned when
      requested (see build_reasoning_details()).

    Returns None when no `reasoning` key is present, or when
    `reasoning.effort` is explicitly "none".

    Falls back to `_DEFAULT_EFFORT_ENV` (CLAUDE_OPENAI_DEFAULT_EFFORT)
    when the payload has no `reasoning` key at all -- lets callers that
    skip `extra_body.reasoning` for custom/localhost providers (e.g.
    Hermes on a 127.0.0.1 base_url) still get thinking blocks when a
    default is configured.
    """
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, dict):
        # No reasoning key in payload -- apply env default (None if unset).
        if _DEFAULT_EFFORT_ENV and _DEFAULT_EFFORT_ENV != "none":
            return _REASONING_EFFORT_MAP.get(_DEFAULT_EFFORT_ENV, "medium")
        return None
    if reasoning.get("enabled") is False:
        return None
    effort = reasoning.get("effort")
    if effort is not None:
        return _REASONING_EFFORT_MAP.get(str(effort).lower(), "medium")
    max_tokens = reasoning.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and max_tokens > 0:
        # OpenRouter's documented effort/token-budget percentage bands:
        # low ~20%, medium ~50%, high ~80%, max ~95%. Without a concrete
        # overall budget, treat the raw token count against the same
        # thresholds as a best-effort proxy.
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
# fetch_model_list() so it can return the actual current model list
# without requiring a separate ANTHROPIC_API_KEY. This is a free
# metadata call, not a billed completion.
_CLAUDE_CREDENTIALS_PATH = os.path.expanduser("~/.claude/.credentials.json")
_MODEL_LIST_CACHE_TTL_S = 300
_model_list_cache = {"data": None, "fetched_at": 0.0}
_model_list_cache_lock = threading.Lock()

# Quota state captured from rate_limit_event chunks during completions.
# Updated on every completion that receives a rate_limit_event, then read
# by /v1/key, /v1/credits, and /health endpoints for monitoring. Initialized
# to None; stays None until the first completion.
_rate_limit_cache = None

# Every spawned `claude` subprocess inherits whatever directory the shim
# process happens to be running from, and Claude Code can read real
# files there. Spawning every `claude` call from a dedicated, empty,
# per-run temp directory closes this off.
_CLAUDE_CWD = tempfile.mkdtemp(prefix="claudecode-as-openai-sandbox-")
MAX_N_CHOICES = 5

# Applied to every spawned `claude` subprocess (see _scoped_env_overrides).
# CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC bundles DISABLE_AUTOUPDATER,
# DISABLE_TELEMETRY, DISABLE_ERROR_REPORTING, and DISABLE_FEEDBACK_COMMAND
# into one flag. Since this shim spawns a fresh `claude -p` process per
# request, the startup work those four disable (autoupdater version
# check, telemetry, error reporting, feedback prompts) is otherwise paid
# on every single call -- disabling it is a real per-request latency win.
# This also means `claude` will never self-update here; the user updates
# it manually on their own schedule instead.
_BASE_ENV_OVERRIDES = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}

# Claude Code's built-in tool names, PLUS the MCP-adjacent helper tools
# that also leak custom-tool dispatch surface: RemoteTrigger (a generic
# dispatcher that can invoke ANY declared custom tool by name), and
# ListMcpResourcesTool/ReadMcpResourceTool (MCP resource browsers, not
# server-specific so --strict-mcp-config doesn't touch them). This does
# NOT fully close the leak when tools ARE requested -- see README "MCP
# tool leakage". When NO tools are requested, the no-tools path below
# (--tools "" --strict-mcp-config) closes this completely instead.
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


def _gen_tool_id():
    return f"toolu_{uuid.uuid4().hex[:20]}"


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
                        "id": tc.get("id", _gen_tool_id()),
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
    credentials file (subscription auth, NOT an API key). Returns the
    token string, or None if missing, unreadable, malformed, or expired.
    Never raises -- every failure mode falls through to the next auth
    method in fetch_model_list()."""
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
    merged in. Free metadata call, not a billed completion. Returns the
    parsed `data` list, or None on any failure so the caller can fall
    through to the next auth method or the hardcoded fallback."""
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


def _anthropic_id_to_openrouter(model_id):
    """Convert a Claude Code / Anthropic API model ID to the OpenRouter
    canonical form: `anthropic/<family>-<major>.<minor>[-suffix]`.

    Anthropic API returns IDs like `claude-opus-4-8-20260528` (dashes,
    date suffix). OpenRouter advertises them as `anthropic/claude-opus-4.8`
    (dots, no date suffix, prefixed). We emit BOTH so clients using either
    naming convention see a match in our catalog.

    Returns the OpenRouter-style ID, or None if the pattern doesn't match
    (already-alias or unrecognised shape -- caller skips ORouter form)."""
    import re as _re
    # e.g. "claude-opus-4-8-20260528" or "claude-haiku-4-5-20251001"
    m = _re.match(
        r"^claude-(opus|sonnet|haiku|fable)-(\d+)-(\d+)(?:-\d{8})?$",
        model_id,
    )
    if m:
        family, major, minor = m.group(1), m.group(2), m.group(3)
        return f"anthropic/claude-{family}-{major}.{minor}"
    # e.g. "claude-opus-5" or "claude-fable-5" (no minor)
    m2 = _re.match(r"^claude-(opus|sonnet|haiku|fable)-(\d+)$", model_id)
    if m2:
        family, major = m2.group(1), m2.group(2)
        return f"anthropic/claude-{family}-{major}"
    return None


# OpenRouter-compatible supported_parameters for Claude models.
# Older Claude 3 models don't support reasoning; everything 4+ does.
# We tag all models generically with the superset -- the shim does not
# distinguish per-model version here, so consumers should treat this as
# "this endpoint supports reasoning" rather than per-model granularity.
_SUPPORTED_PARAMS_WITH_REASONING = [
    # Fully supported: honored on every request.
    "include_reasoning",  # OpenRouter: return reasoning tokens in response
    "max_tokens",         # -> CLAUDE_CODE_MAX_OUTPUT_TOKENS env var
    "reasoning",          # -> --effort flag (see resolve_reasoning_effort)
    "response_format",    # -> --json-schema (json_schema/json_object types)
    "stop",               # client-side emulation via _apply_stop_sequences
    "tools",              # native MCP tool registration (see build_mcp_tool_config)
    "tool_choice",        # "none" fully honored; "required"/named: accepted, not enforced
]
_SUPPORTED_PARAMS_NO_REASONING = [
    # Same as WITH_REASONING minus the reasoning-specific entries.
    "max_tokens",
    "response_format",
    "stop",
    "tools",
    "tool_choice",
]
_LEGACY_NO_REASONING_PREFIXES = ("claude-3-",)


def _supports_reasoning_params(model_id):
    """Return True when the model ID looks like a reasoning-capable Claude
    model (Claude 4+ family). Claude 3.x and older return False."""
    import re as _re
    lower = model_id.lower()
    # Any Claude 3 model (3-haiku, 3.5-haiku, etc.)
    if _re.search(r"claude-3", lower):
        return False
    # Claude 4+ opus/sonnet/haiku/fable all support reasoning
    return True


def _model_entry(model_id, context_length=None):
    """Build an OpenRouter-compatible model entry dict for a single model.

    Emits `supported_parameters` and a minimal `reasoning` object so
    consumers (e.g. Hermes) can auto-detect reasoning support from the
    catalog without per-model config. See README 'OpenRouter model-name
    compatibility' for the schema source."""
    has_reasoning = _supports_reasoning_params(model_id)
    entry = {
        "id": model_id,
        "object": "model",
        "supported_parameters": (
            _SUPPORTED_PARAMS_WITH_REASONING if has_reasoning
            else _SUPPORTED_PARAMS_NO_REASONING
        ),
    }
    if context_length:
        entry["context_length"] = context_length
    if has_reasoning:
        # Minimal OpenRouter-schema reasoning object. `mandatory: false`
        # means reasoning can be disabled; `supported_efforts` matches the
        # four values Claude Code's --effort flag accepts.
        entry["reasoning"] = {
            "mandatory": False,
            "default_enabled": False,
            "supports_max_tokens": True,
            "supported_efforts": ["low", "medium", "high", "max"],
            "default_effort": "medium",
        }
    return entry


def _build_key_response():
    """Return an OpenRouter-compatible /v1/key response backed by the last
    rate_limit_event captured from the claude subprocess.

    Uses the 7-day window as the primary quota because it is the limit
    most likely to constrain a day's work.  The 5-hour window is included
    in the `rate_limits` extension field so callers can surface it.

    Returns null values before the first turn completes (no data yet)."""
    info = _rate_limit_cache
    if not info:
        return {
            "data": {
                "label": "claude-code-subscription",
                "limit": None,
                "limit_remaining": None,
                "is_free_tier": False,
            }
        }
    windows = info.get("unifiedWindows", {})
    five_h = windows.get("five_hour", {})
    seven_d = windows.get("seven_day", {})

    # Use the 5-hour window as the primary quota: it is the current/active
    # limit users hit first. The 7-day window is included in the extension
    # field for informational display.
    five_h_used = five_h.get("utilization", 0.0)
    limit = 100
    usage = round(five_h_used * limit, 2)
    limit_remaining = round(limit - usage, 2)

    return {
        "data": {
            "label": "claude-code-subscription",
            "limit": limit,
            "usage": usage,
            "limit_remaining": limit_remaining,
            "is_free_tier": False,
            "rate_limit_status": info.get("status"),
            "rate_limits": {
                "5h": {
                    "percent_used": round(five_h.get("utilization", 0.0) * 100),
                    "resets_at": five_h.get("resetsAt"),
                },
                "7d": {
                    "percent_used": round(seven_d.get("utilization", 0.0) * 100),
                    "resets_at": seven_d.get("resetsAt"),
                },
            },
        }
    }


def _build_credits_response():
    """Return an OpenRouter-compatible /v1/credits response.

    Maps the 5-hour (current/active) usage window to a 0-100 credit scale,
    consistent with /v1/key so that clients reading total_credits/total_usage
    get a coherent view."""
    info = _rate_limit_cache
    if not info:
        return {"data": {"total_credits": None, "total_usage": None}}
    five_h = info.get("unifiedWindows", {}).get("five_hour", {})
    total_credits = 100
    total_usage = round(five_h.get("utilization", 0.0) * total_credits, 2)
    return {"data": {"total_credits": total_credits, "total_usage": total_usage}}


def fetch_model_list():
    """Returns the current Anthropic model list for /v1/models, shaped as
    OpenRouter-compatible entries with reasoning capability signals.

    Each model is emitted TWICE:
      1. In native Claude Code format (e.g. `claude-opus-4-8`) -- so a
         client using Claude Code model names finds an exact match.
      2. In OpenRouter format (e.g. `anthropic/claude-opus-4.8`, dotted,
         prefixed) -- so a client configured for OpenRouter model names
         also finds a match.

    Both entries carry `supported_parameters` (including `"reasoning"` for
    Claude 4+ models) and a `reasoning` object, so consumers like Hermes
    can auto-detect reasoning capability from this catalog response without
    any per-model config override.

    Auth order: (1) Claude Code OAuth token; (2) ANTHROPIC_API_KEY env var;
    (3) KNOWN_MODEL_ALIASES hardcoded fallback. Cached for
    _MODEL_LIST_CACHE_TTL_S. Never raises."""
    with _model_list_cache_lock:
        cached = _model_list_cache["data"]
        if cached is not None and (time.time() - _model_list_cache["fetched_at"]) < _MODEL_LIST_CACHE_TTL_S:
            return cached

    raw_data = None
    oauth_token = _read_claude_oauth_token()
    if oauth_token:
        raw_data = _fetch_models_from_anthropic_api({"Authorization": f"Bearer {oauth_token}"})

    if raw_data is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            raw_data = _fetch_models_from_anthropic_api({"x-api-key": api_key})

    seen_ids = set()
    result = []

    def add_entry(model_id, context_length=None):
        if model_id and model_id not in seen_ids:
            seen_ids.add(model_id)
            result.append(_model_entry(model_id, context_length))

    if raw_data is not None:
        for m in raw_data:
            native_id = m.get("id")
            if not native_id:
                continue
            ctx = m.get("context_window") or m.get("context_length")
            # Native Claude Code / Anthropic API ID (e.g. claude-opus-4-8-20260528)
            add_entry(native_id, ctx)
            # OpenRouter-format alias (e.g. anthropic/claude-opus-4.8)
            or_id = _anthropic_id_to_openrouter(native_id)
            add_entry(or_id, ctx)
    else:
        # Hardcoded fallback: bare aliases only, no OpenRouter duplicates
        for alias in KNOWN_MODEL_ALIASES:
            add_entry(alias)

    with _model_list_cache_lock:
        _model_list_cache["data"] = result
        _model_list_cache["fetched_at"] = time.time()
    return result




