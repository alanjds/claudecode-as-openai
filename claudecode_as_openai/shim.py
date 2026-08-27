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


def render_tools_into_system_prompt(tools, base_system_prompt):
    """LEGACY prose-based tool-description fallback, superseded by
    build_mcp_tool_manifest() (real MCP tool schemas). Kept as an automatic
    fallback for tool names that can't be represented as a valid MCP
    tool name -- see call_claude_streaming's use of this function only
    when build_mcp_tool_manifest() returns None. Framing tools as
    already-wired, real, harness-implemented tools (not "custom"/
    hypothetical) is what makes Claude actually call them instead of
    hedging; this path still only reaches ~40% single-shot reliability
    (see TOOL_CALL_MAX_RETRIES) -- the MCP path is the real fix."""
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
# tool-call reliability problem. Verified live: Claude Code's
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


def _normalize_stop_sequences(stop):
    """Shared with WarmProcess.send_turn: OpenAI's `stop` is a string or
    list of up to 4 strings; Claude Code has no native stop-sequence
    flag, so this shim always emulates it client-side against the
    accumulated text (see _apply_stop_sequences)."""
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else [s for s in stop if s]


# ---------------------------------------------------------------------------
# Persistent warm-pool: keeps ONE already-spawned, already-bootstrapped
# `claude -p --input-format stream-json` process parked per conversation,
# ready to take the NEXT turn without paying the ~5s process-spawn/
# bootstrap cost a fresh `-p --resume` invocation pays on every call
# (measured live: cold `-p --resume` wall time vs the CLI's own
# self-reported duration_ms showed a ~5.2s unaccounted gap -- pure
# process-lifecycle overhead outside the API call itself -- that
# collapses to ~5ms once a process is already resident and warm).
#
# Design, per explicit user direction: at most ONE parked (idle, already
# spawned) process at a time, never a process per concurrent conversation.
# A "new" conversation here means "different from the immediately
# preceding one" -- there is no attempt to keep N conversations warm
# simultaneously.
#
#   1. A turn for a brand-new conversation arrives -> served cold (no
#      warm process can exist for a conversation that didn't exist yet).
#      In parallel, once that reply's real Claude session_id is known,
#      spawn a WarmProcess pre-resuming that exact session, parked
#      waiting for turn 2.
#   2. The next turn for the SAME conversation (matching fingerprint +
#      session_id) arrives -> claim the parked WarmProcess, feed it the
#      turn directly (no spawn), and spawn a fresh WarmProcess to park
#      for turn 3 once this reply is known.
#   3. A turn for a DIFFERENT conversation arrives (new fingerprint, or a
#      continuation whose synced history no longer matches this parked
#      process's session) -> the stale parked process is useless (its
#      --resume target is for the wrong conversation) and is killed
#      immediately; that turn is served cold, and a new WarmProcess is
#      parked for whatever comes next.
#
# Cross-compatible with the existing cold path by construction: a warm
# process's Claude session_id is a completely normal Claude Code session
# (created with plain --session-id / --resume, just kept alive across
# turns via --input-format stream-json instead of exiting after one).
# Verified live: a session created and advanced by a WarmProcess resumes
# correctly via a totally separate one-shot `-p --resume` call after the
# warm process is killed, and vice versa -- either side can pick up
# where the other left off with no special handling needed.
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


class WarmPool:
    """Holds at most one parked WarmProcess at a time (see module-level
    comment above for the full protocol). Thread-safe: HTTP requests are
    served from a ThreadingHTTPServer, so claiming/replacing the parked
    process must be atomic against a concurrent request doing the same."""

    # A parked process that hasn't been claimed within this window is
    # evicted: any gap longer than this means the conversation probably
    # ended or moved on, so the next request will bring a different
    # fingerprint and kill it anyway. Killing proactively here prevents
    # the process from sitting alive indefinitely if the user never
    # returns -- which is exactly what caused the quota-burndown incident
    # (a parked --resume process kept retrying an invalid session for hours).
    _PARK_IDLE_TIMEOUT_S = 300   # 5 minutes idle (parked, unclaimed)
    # Hard upper bound on a process's entire lifetime regardless of idle
    # state: caps the worst-case quota burn from a stuck parked subprocess.
    _LIFETIME_TIMEOUT_S = 600    # 10 minutes total

    def __init__(self):
        self._lock = threading.Lock()
        self._parked = None  # WarmProcess | None
        self._reaper_thread = None  # Background thread that evicts stale processes

    def take_if_matching(self, fingerprint):
        """Returns the parked WarmProcess if it exists, is alive, not
        stale (idle or lifetime timeout -- see class constants), and
        matches `fingerprint` (same conversation), claiming it
        atomically so no other request can also take it. Returns None
        otherwise -- including when a parked process exists but is for
        a DIFFERENT conversation, in which case it's killed here (a
        stale parked process is useless once the conversation moves on,
        per the "at most one parked process, discarded on mismatch"
        design)."""
        with self._lock:
            candidate = self._parked
            if candidate is None:
                return None
            now = time.time()
            # Kill stale processes before even checking fingerprint.
            # Idle check: parked too long without being claimed.
            if (candidate.parked_at is not None
                    and now - candidate.parked_at > self._PARK_IDLE_TIMEOUT_S):
                self._parked = None
                idle_s = now - candidate.parked_at
                sys.stderr.write(
                    f"claudecode-as-openai: warm-pool: evicting"
                    f" {candidate.session_id} -- idle {idle_s:.0f}s"
                    f" > {self._PARK_IDLE_TIMEOUT_S}s limit\n"
                )
                candidate.kill()
                return None
            # Lifetime check: total age since spawn.
            if now - candidate.spawned_at > self._LIFETIME_TIMEOUT_S:
                self._parked = None
                age_s = now - candidate.spawned_at
                sys.stderr.write(
                    f"claudecode-as-openai: warm-pool: evicting"
                    f" {candidate.session_id} -- lifetime {age_s:.0f}s"
                    f" > {self._LIFETIME_TIMEOUT_S}s limit\n"
                )
                candidate.kill()
                return None
            if candidate.fingerprint != fingerprint:
                self._parked = None
                candidate.kill()
                return None
            self._parked = None
        if not candidate.claim():
            return None
        return candidate

    def park(self, warm_process):
        """Installs `warm_process` as the new parked process, killing
        whatever was parked before it (there is only ever one). Also
        checks if the process is already stale (shouldn't happen in normal
        flow, but defensive against race conditions) and starts the reaper
        thread if it's not already running."""
        with self._lock:
            stale = self._parked
            now = time.time()
            # Defensive check: if somehow the new process is already stale,
            # don't park it, just kill it immediately.
            if (now - warm_process.spawned_at > self._LIFETIME_TIMEOUT_S
                    or (warm_process.parked_at is not None
                        and now - warm_process.parked_at > self._PARK_IDLE_TIMEOUT_S)):
                sys.stderr.write(
                    f"claudecode-as-openai: warm-pool: not parking {warm_process.session_id}"
                    f" -- already stale on arrival\n"
                )
                warm_process.kill()
                return
            self._parked = warm_process
            warm_process.parked_at = now
            # Start the reaper thread if it's not already running
            if self._reaper_thread is None or not self._reaper_thread.is_alive():
                self._reaper_thread = threading.Thread(target=self._reap_loop, daemon=True)
                self._reaper_thread.start()
        if stale is not None:
            stale.kill()

    def discard(self, fingerprint):
        """Kills and clears the parked process if it matches
        `fingerprint`. Used when a conversation ends up not continuing
        the way the parked process assumed (e.g. the caller's next
        request diverged before ever reaching take_if_matching)."""
        with self._lock:
            candidate = self._parked
            if candidate is None or candidate.fingerprint != fingerprint:
                return
            self._parked = None
        candidate.kill()

    def _reap_loop(self):
        """Background thread that periodically evicts stale parked processes.
        Runs once the first process is parked, sleeps 60 seconds between checks.
        Exits cleanly if _parked becomes None."""
        while True:
            time.sleep(60)  # Check every 60 seconds for stale processes
            with self._lock:
                candidate = self._parked
                if candidate is None:
                    # No process parked; reaper can exit
                    return
                now = time.time()
                # Check if stale
                if now - candidate.spawned_at > self._LIFETIME_TIMEOUT_S:
                    age = now - candidate.spawned_at
                    sys.stderr.write(
                        f"claudecode-as-openai: warm-pool: reaper evicting"
                        f" {candidate.session_id} -- lifetime {age:.0f}s"
                        f" > {self._LIFETIME_TIMEOUT_S}s limit\n"
                    )
                    self._parked = None
                    candidate.kill()
                    return
                elif (candidate.parked_at is not None
                        and now - candidate.parked_at > self._PARK_IDLE_TIMEOUT_S):
                    idle = now - candidate.parked_at
                    sys.stderr.write(
                        f"claudecode-as-openai: warm-pool: reaper evicting"
                        f" {candidate.session_id} -- idle {idle:.0f}s"
                        f" > {self._PARK_IDLE_TIMEOUT_S}s limit\n"
                    )
                    self._parked = None
                    candidate.kill()
                    return


_WARM_POOL = WarmPool()


def _log_quota_snapshot():
    """Log current quota state to stderr if available. Called after every
    completion to give operators visibility into quota burn rate and
    remaining headroom."""
    info = _rate_limit_cache
    if not info:
        return
    status = info.get("status", "unknown")
    five_h = info.get("unifiedWindows", {}).get("five_hour", {})
    util = five_h.get("utilization", 0.0)
    resets_at = five_h.get("resetsAt")
    # Only log at WARNING/CRITICAL levels to reduce noise on normal operations
    if util >= 0.90:
        severity = "CRITICAL" if util >= 0.95 else "WARNING"
        sys.stderr.write(
            "claudecode-as-openai: quota_snapshot status=%s util_5h=%.1f%% "
            "severity=%s resets_at=%s\n" % (status, util * 100, severity, resets_at)
        )


def _parking_fingerprint(conv_key, model, tools_requested, tools, effort, env_overrides=None):
    """What must stay IDENTICAL between the turn a WarmProcess was
    parked for and the turn that claims it. Deliberately includes
    everything _build_claude_cmd branches on for this call shape
    (model/tools/effort change the spawned command line) plus the
    session-cache conv_key (a different conv_key is, by definition, a
    different conversation) and env_overrides (max_tokens changes
    CLAUDE_CODE_MAX_OUTPUT_TOKENS, which is baked into a warm process's
    environment at spawn time -- see WarmProcess._start) -- any mismatch
    here means the parked process's command line, environment, or
    --resume target are wrong for the incoming turn, so per the user's
    explicit design it must be discarded and respawned, never reused
    with different parameters."""
    tools_key = None
    if tools:
        try:
            tools_key = json.dumps(tools, sort_keys=True, default=str)
        except Exception:
            tools_key = str(tools)
    env_key = json.dumps(env_overrides, sort_keys=True) if env_overrides else None
    return (conv_key, model, bool(tools_requested), tools_key, effort, env_key)


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


def _build_claude_cmd(model, session_mode, session_id, tools_requested, max_turns, json_schema, want_partial_messages=False, mcp_tool_config=None, effort=None, input_format=None):
    cmd = [CLAUDE_BIN]
    # Excludes user/project/local settings.json entirely, preventing
    # locally-configured SessionStart/other hooks from injecting extra
    # context into every completion -- a stateless API shim should never
    # be silently subject to that. Not --bare: --bare strictly requires
    # ANTHROPIC_API_KEY and never reads OAuth, which would break this
    # shim's whole premise of riding the user's Claude subscription.
    cmd += ["--setting-sources", ""]
    if mcp_tool_config is not None:
        # Native MCP tool registration path (see build_mcp_tool_config):
        # --strict-mcp-config restricts MCP servers to ONLY the
        # just-built manifest server, --allowedTools further restricts
        # to exactly this session's declared tool names, and
        # --disallowedTools blocks every Claude Code built-in by name.
        # This combination exposes ONLY the intended
        # mcp__shim_tools__<name> entries -- no built-ins, no leakage.
        cmd += [
            "--disallowedTools", EXTENDED_DISALLOWED_TOOLS,
            "--strict-mcp-config", "--mcp-config", json.dumps(mcp_tool_config["mcp_config"]),
            "--allowedTools", ",".join(mcp_tool_config["allowed_tools"]),
        ]
    elif tools_requested:
        cmd += ["--disallowedTools", EXTENDED_DISALLOWED_TOOLS]
    else:
        # No tools requested: fully lock down BOTH built-ins and any
        # locally-configured MCP servers (see EXTENDED_DISALLOWED_TOOLS
        # and README "MCP tool leakage").
        cmd += ["--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    cmd += ["--output-format", "stream-json", "--verbose", "--max-turns", str(max_turns)]
    if input_format == "stream-json":
        # Used only by the persistent warm-pool process (see
        # WarmProcess): keeps the underlying `claude -p` subprocess
        # alive across multiple turns of the SAME conversation, reading
        # one JSON line per turn from stdin instead of exiting after a
        # single message array. --replay-user-messages makes the CLI
        # echo each user turn back on stdout so the reader can align
        # results to the right turn without a separate side channel.
        cmd += ["--input-format", "stream-json", "--replay-user-messages"]
    if want_partial_messages:
        # Adds token-level "stream_event"/"content_block_delta" lines
        # interleaved with the existing full "assistant" message chunks.
        # Needed because Claude Code's default stream-json granularity is
        # per-message, not per-token: without this, a stop sequence can
        # only be detected after an entire text block has already been
        # generated and billed, and `stream: true` would only deliver one
        # giant content delta instead of a real typing effect.
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


def _consume_claude_response(chunk_source, deadline, stop_sequences, stream_callback):
    """Shared NDJSON-chunk parsing loop, used by both a one-shot `claude
    -p` invocation (call_claude_streaming) and a persistent warm-pool
    process serving one turn at a time (WarmProcess.send_turn). Reads
    from `chunk_source` (any iterable of parsed NDJSON dicts) until a
    "result" chunk, a tool_use/structured-output block, a stop-sequence
    match, or the deadline ends it. Returns the same result dict shape
    call_claude_streaming has always returned; raises ClaudeCliError on
    the same conditions as before. Does NOT touch process lifecycle --
    callers own spawning/terminating whatever produced chunk_source."""
    text_parts = []
    reasoning_parts = []
    reasoning_signature = None
    tool_calls = []
    structured_json = None
    usage = {}
    finish_reason = "stop"
    stop_matched = False
    streaming_partial_text = ""

    for chunk in chunk_source:
        if time.time() > deadline:
            break
        ctype = chunk.get("type")

        if ctype == "rate_limit_event":
            global _rate_limit_cache
            info = chunk.get("rate_limit_info") or {}
            _rate_limit_cache = info
            status = info.get("status")
            if status == "rejected":
                # Quota window exhausted: failing immediately avoids
                # burning tokens on retries that will all be rejected
                # anyway (this was the root cause of the quota-burndown
                # incident -- a stuck --resume kept retrying on 429s).
                overage = info.get("overageStatus", "unknown")
                five_h_util = (
                    info.get("unifiedWindows", {})
                    .get("five_hour", {})
                    .get("utilization", 1.0)
                )
                sys.stderr.write(
                    f"claudecode-as-openai: quota rejected"
                    f" (5h={five_h_util*100:.1f}%, overage={overage});"
                    f" failing immediately\n"
                )
                raise ClaudeCliError(
                    429, "rate_limit_error",
                    f"Claude Code quota rejected: 5-hour window at"
                    f" {five_h_util*100:.1f}% ({overage}).",
                    code="quota_rejected",
                )
            # Log when quota is getting high (once per function call, not
            # once per session -- rate_limit_events arrive roughly once
            # per API round-trip so this doesn't spam in normal usage).
            five_h = info.get("unifiedWindows", {}).get("five_hour", {})
            util = five_h.get("utilization", 0.0)
            if util >= 0.9:
                severity = "CRITICAL" if util >= 0.95 else "WARNING"
                resets_at = five_h.get("resetsAt", "unknown")
                sys.stderr.write(
                    f"claudecode-as-openai: QUOTA_{severity}"
                    f" 5h={util*100:.1f}% status={status}"
                    f" resets_at={resets_at}\n"
                )
            continue

        if ctype == "stream_event" and (stop_sequences or stream_callback):
            # Real token-level deltas, only present when
            # --include-partial-messages was passed. These interleave
            # with, and arrive BEFORE, the full "assistant" chunk for
            # the same content block: content_block_start ->
            # content_block_delta(s) -> the full "assistant" chunk ->
            # content_block_stop -> next content_block_start. That
            # ordering is what makes resetting the per-block
            # accumulator on content_block_start safe. This lets a
            # stop sequence be caught mid-block instead of only after
            # a full text block finishes, and is what delivers real
            # client-facing streaming when stream_callback is set.
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
                    # NDJSON message and dispatches the actual tool_use
                    # in a SEPARATE, later assistant message within the
                    # same turn ([thinking, text, tool_use] is common).
                    # Breaking on the text-only message silently
                    # swallows the tool_use that follows. Keep reading;
                    # only stop early on an actual tool_use/structured-
                    # output block, a stop-sequence match, an error, or
                    # the stream ending naturally (the "result" chunk
                    # case further down).
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
                                "id": content.get("id", _gen_tool_id()),
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
                # the caller tears down (or, for a warm process, simply
                # stops reading -- the process itself stays alive for
                # the next turn) immediately, which is the actual
                # latency/cost win over letting `claude` keep generating
                # a response nobody will see.
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
# text instead of a real tool_use block (~40% single-shot native tool_use
# reliability with --disallowedTools + strong framing). Retrying turns
# that into a much higher practical success rate at the cost of extra
# latency/spend only on the failing path. No deterministic fix exists for
# this failure mode anywhere in the Claude Code ecosystem (checked
# @anthropic-ai/claude-agent-sdk: retries transport/API errors only, no
# tool_choice:required-equivalent); every implementation handles it with
# a retry loop.
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


# Sampling/formatting parameters with no equivalent anywhere in the
# Claude Code CLI: no flag, no env var. `--effort <low|medium|high|max>`
# (reasoning-effort) IS mapped, but only from OpenRouter's `reasoning`
# parameter (see resolve_reasoning_effort) -- a different axis (internal
# deliberation) from output randomness/diversity, so it's not treated as
# a substitute for temperature/top_p below.
#
# These are silently accepted (never hard-errored) because real OpenAI
# clients routinely send explicit defaults on every request (e.g.
# `temperature: 1.0`, the OpenAI default itself, carries no signal the
# caller wants non-default behavior) -- hard-erroring on presence-of-key
# would break compatibility with well-behaved clients for no benefit. A
# one-time stderr warning per parameter name is emitted instead.
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


def _build_sse_chunk(chat_id, created, model, index, delta, finish=None):
    """Shared shape for an OpenAI `chat.completion.chunk` SSE event, used
    by both the real-token streaming path and the buffered-then-emit
    streaming path."""
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish}],
    }


@contextmanager
def _scoped_env_overrides(env_overrides):
    """Temporarily patches subprocess.Popen so any process it spawns
    while this context is active inherits `env_overrides` merged into
    the current environment. Used to scope CLAUDE_CODE_MAX_OUTPUT_TOKENS
    (from `max_tokens`) to a single completion's subprocess call without
    mutating the shim's own process-wide environment."""
    if not env_overrides:
        yield
        return
    original_popen = subprocess.Popen

    def scoped_popen(*args, **kwargs):
        env = dict(os.environ)
        env.update(env_overrides)
        kwargs["env"] = env
        return original_popen(*args, **kwargs)

    subprocess.Popen = scoped_popen
    try:
        yield
    finally:
        subprocess.Popen = original_popen


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

    def _check_quota_headroom(self):
        """Early check: if quota is rejected and won't reset soon, fail fast
        without spawning claude. Saves subprocess overhead when we know it
        will fail anyway."""
        info = _rate_limit_cache
        if not info:
            # No quota data yet; proceed normally
            return
        status = info.get("status")
        if status != "rejected":
            # Quota is not currently exhausted; proceed
            return
        # Quota is rejected. Check if the 5h window has reset yet.
        resets_at = info.get("unifiedWindows", {}).get("five_hour", {}).get("resetsAt")
        if resets_at is None:
            # No reset time available; proceed (defensive)
            return
        now = time.time()
        if now < resets_at:
            # 5h window hasn't reset yet; fail immediately
            secs_until_reset = int(resets_at - now)
            raise ClaudeCliError(
                429, "rate_limit_error",
                f"Claude Code quota rejected. 5-hour window resets in {secs_until_reset}s.",
                code="quota_rejected_known_reset",
            )
        # Window has reset but status is still "rejected" (stale cache or
        # 7-day limit exhausted); proceed and let the subprocess discover
        # the real state

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(
                {
                    "object": "list",
                    "data": fetch_model_list(),
                }
            )
            return
        if self.path.rstrip("/") in ("/v1/key", "/key"):
            self._send_json(_build_key_response())
            return
        if self.path.rstrip("/") in ("/v1/credits", "/credits"):
            self._send_json(_build_credits_response())
            return
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send_json(self._build_health())
            return
        self._send_json({"status": "ok"})

    def _build_health(self):
        """Returns a dict with quota and warm-pool state for the /health
        endpoint.  Useful for monitoring: lets operators see quota
        utilization and whether a parked process is sitting alive without
        making a real completion call."""
        health = {"status": "ok"}
        # Quota info -- populated after the first completion that returned
        # a rate_limit_event; null before that.
        info = _rate_limit_cache
        if info:
            windows = info.get("unifiedWindows", {})
            five_h = windows.get("five_hour", {})
            seven_d = windows.get("seven_day", {})
            health["quota"] = {
                "status": info.get("status"),
                "5h_utilization": five_h.get("utilization"),
                "5h_resets_at": five_h.get("resetsAt"),
                "7d_utilization": seven_d.get("utilization"),
                "overage_status": info.get("overageStatus"),
            }
        else:
            health["quota"] = None
        # Warm-pool state.
        with _WARM_POOL._lock:
            parked = _WARM_POOL._parked
            if parked is not None and parked.is_alive():
                now = time.time()
                health["warm_pool"] = {
                    "active": True,
                    "session_id": parked.session_id,
                    "spawned_s_ago": round(now - parked.spawned_at, 1),
                    "parked_s_ago": (
                        round(now - parked.parked_at, 1)
                        if parked.parked_at is not None else None
                    ),
                }
            else:
                health["warm_pool"] = {"active": False}
        return health

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
        # include_reasoning: OpenRouter's flag to include/suppress reasoning
        # tokens in the response. True or absent = include (default); False =
        # suppress reasoning/reasoning_details even when --effort was set.
        include_reasoning = payload.get("include_reasoning", True)

        # Short-circuit if quota is exhausted and won't reset for a while.
        # Check before any subprocess spawn to save time/resources.
        self._check_quota_headroom()

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
        # Native MCP tool registration is the real fix for tool-call
        # reliability and is always attempted first inside
        # call_claude_streaming/_run_one_completion. If any tool name can't
        # be represented as a valid MCP tool name, it will be silently dropped
        # (MCP is the only supported path now; prose fallback removed).
        if effective_tools and build_mcp_tool_manifest(effective_tools) is None:
            # Tool name(s) fail MCP constraint -- drop them silently rather
            # than falling back to prose (removed for cost: max_turns=1 now)
            effective_tools = None

        env_overrides = dict(_BASE_ENV_OVERRIDES)
        if max_tokens:
            try:
                env_overrides["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(max_tokens))
            except (TypeError, ValueError):
                pass

        _warn_unsupported_sampling_params(payload)

        # Real token-level streaming is only safe for the single-choice,
        # no-tools, no-json_schema, no-reasoning path: with tool retry in
        # play, a failed attempt's narration text must not reach the
        # client early; --json-schema's multi-turn corrective mechanism
        # doesn't map onto a single token stream; and reasoning requires
        # capturing the `thinking` block before `text` starts (see
        # build_reasoning_details), which the buffered path's
        # text_delta-only stream_callback has no hook for. Everything
        # else falls back to buffered-then-emit (SSE framing still
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
                tools=effective_tools, stop=stop, effort=effort, conv_key=conv_key,
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

            if include_reasoning is not False:
                # include_reasoning=False suppresses thinking blocks from the
                # response even when --effort produced them. Default (True or
                # absent) always includes them when present.
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
            _log_quota_snapshot()
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
        _log_quota_snapshot()

    def _handle_streaming_completion(self, messages, model, system_prompt, max_tokens, env_overrides, stop):
        """Real token-level SSE streaming for the safe case (single choice,
        no tools requested, no json_schema -- see the can_stream_live
        gate in _handle_chat_completion). Sends SSE headers immediately,
        then forwards each real text_delta from
        call_claude_streaming(stream_callback=...) to the client as it
        arrives, instead of buffering the full response first. Falls back
        to the same session-caching/resume logic as the buffered path.

        Also checks the warm pool (see WarmPool/WarmProcess) for a parked
        process matching this turn's fingerprint before spawning cold --
        a warm process is created with use_pty=True specifically so it
        can serve a LATER streaming turn just as well as a buffered one;
        verified live that token-level deltas remain correctly paced
        (not bursty) when streamed off an already-resident --resume'd
        process, same signature as the cold path."""
        session_mode, session_id, delta_messages, conv_key = resolve_session(messages)
        delta_claude_messages = build_claude_messages(delta_messages)

        fingerprint = None
        if conv_key is not None:
            fingerprint = _parking_fingerprint(conv_key, model, False, None, None, env_overrides)

        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        role_sent = False

        def emit(delta, finish=None):
            chunk = _build_sse_chunk(chat_id, created, model, 0, delta, finish)
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

        def _park_next(final_sid):
            # Same best-effort contract as _run_one_completion's
            # _park_next: a spawn failure here must never surface as a
            # request failure, since the completion already succeeded.
            try:
                warm = WarmProcess(
                    fingerprint, final_sid, model, system_prompt, False, None,
                    None, None, use_pty=True, env_overrides=env_overrides,
                )
                _WARM_POOL.park(warm)
            except Exception:
                pass

        try:
            with _scoped_env_overrides(env_overrides):
                # Guarded on session_mode == "resume", matching
                # _run_one_completion's warm-pool gate exactly: a
                # "fresh" session_mode means delta_claude_messages is
                # the FULL conversation history (resolve_session found
                # no matching prior sync), not a true delta -- sending
                # that to an already-`--resume`d warm process would
                # duplicate/corrupt its session transcript. A parked
                # process can only ever be correctly claimed for the
                # NEXT turn of a conversation it already knows about.
                warm = (
                    _WARM_POOL.take_if_matching(fingerprint)
                    if fingerprint is not None and session_mode == "resume"
                    else None
                )
                if warm is not None:
                    try:
                        result = warm.send_turn(delta_claude_messages, stop=stop, stream_callback=on_text_delta)
                    except ClaudeCliError:
                        warm.kill()
                        raise
                    else:
                        # A WarmProcess serves exactly one turn, ever
                        # (see its docstring) -- kill it as soon as
                        # that turn is done, whether or not this reply
                        # ends up parking a fresh replacement below.
                        # Without this, a successfully-served warm
                        # process is simply abandoned: still alive,
                        # still holding an --include-partial-messages
                        # PTY subprocess open, forever (verified live:
                        # ps showed the spent process sitting in S
                        # state indefinitely after being claimed).
                        warm.kill()
                else:
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
            # fingerprint is set whenever conv_key is (see above), so
            # this always parks a fresh WarmProcess for this
            # conversation's next turn -- whether this turn was itself
            # served warm or cold.
            _park_next(session_id)

    def _run_one_completion(self, delta_messages, full_messages, system_prompt, model,
                             tools_requested, session_mode, session_id, json_schema, env_overrides,
                             tools=None, stop=None, effort=None, conv_key=None):
        """Single entry point for producing one completion, regardless of
        whether env overrides (max_tokens), structured output (json_schema),
        and/or tool retry apply. Returns (result_dict, final_session_mode,
        final_session_id). Consolidating this here (rather than branching
        across several call sites) keeps the env-var scoping and retry
        logic each applied exactly once, in a well-defined order.

        Also owns the warm-pool fast path (see WarmPool/WarmProcess): when
        `conv_key` is given, `session_mode == "resume"` (a brand-new
        conversation's first turn can never have a parked process, by
        definition), and a parked process matches this exact call's
        fingerprint, the turn is served by that already-running process
        instead of spawning a fresh one -- skipping the ~5s cold
        process-spawn/bootstrap cost measured live between a fresh `-p
        --resume` call and an already-resident warm process. Whether
        served warm or cold, a new WarmProcess is parked afterward for
        this conversation's NEXT turn, per the user's explicit "one
        parked process, replaced on any mismatch" design -- never one
        process per concurrent conversation."""
        mcp_tool_config = build_mcp_tool_config(tools) if tools else None
        fingerprint = None
        if conv_key is not None:
            fingerprint = _parking_fingerprint(
                conv_key, model, tools_requested, tools, effort, env_overrides,
            )

        def _do_call(msgs, mode, sid):
            return call_claude_streaming(
                msgs, system_prompt, model,
                session_mode=mode, session_id=sid,
                tools_requested=tools_requested, tools=tools, json_schema=json_schema,
                stop=stop, effort=effort,
            )

        def _park_next(final_sid):
            # Best-effort: a warm-process spawn failure here must never
            # surface as a request failure -- the completion this turn
            # produced is already valid and about to be returned. Worst
            # case, the next turn just falls back to the cold path.
            try:
                warm = WarmProcess(
                    fingerprint, final_sid, model, system_prompt, tools_requested, tools,
                    mcp_tool_config, effort, use_pty=False, env_overrides=env_overrides,
                )
                _WARM_POOL.park(warm)
            except Exception:
                pass

        with _scoped_env_overrides(env_overrides):
            if json_schema is not None:
                # Structured output doesn't go through the tool-call retry
                # loop, or the warm pool -- --json-schema uses its own
                # internal mechanism (a StructuredOutput tool + corrective
                # turn) that isn't the "narrated instead of dispatching"
                # failure mode the retry loop targets, and its multi-turn
                # correction doesn't fit the warm pool's one-turn-per-park
                # model cleanly. Kept on the cold path unconditionally.
                result = _do_call(delta_messages, session_mode, session_id)
                return result, session_mode, session_id

            if fingerprint is not None and session_mode == "resume":
                warm = _WARM_POOL.take_if_matching(fingerprint)
                if warm is not None:
                    try:
                        result = warm.send_turn(delta_messages, stop=stop)
                    except ClaudeCliError:
                        warm.kill()
                        raise
                    if result["tool_calls"] or not tools_requested:
                        # A WarmProcess serves exactly one turn, ever --
                        # kill it now that this turn is done, same as
                        # the streaming path (see there for the leak
                        # this fixes: an unkilled spent process just
                        # sits alive indefinitely).
                        warm.kill()
                        _park_next(session_id)
                        return result, session_mode, session_id
                    # Warm-served turn wanted a tool call but didn't get
                    # one: fall through to the cold retry loop exactly
                    # like a first cold attempt would (see
                    # call_claude_with_tool_retry) -- the warm process is
                    # already spent (one turn each) and not reused.
                    warm.kill()

            result, final_mode, final_id = call_claude_with_tool_retry(
                delta_messages, full_messages, system_prompt, model,
                tools_requested=tools_requested, session_mode=session_mode, session_id=session_id,
                tools=tools, stop=stop, effort=effort,
            )
            if fingerprint is not None:
                _park_next(final_id)
            return result, final_mode, final_id


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
            chunk = _build_sse_chunk(chat_id, created, model, index, delta, finish)
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
        port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    except ValueError:
        print(f"Invalid port: {sys.argv[1]!r}", file=sys.stderr)
        sys.exit(2)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Claude Code shim (native tool-calling + session caching) listening on http://127.0.0.1:{port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
