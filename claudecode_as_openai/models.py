#!/usr/bin/env python3
"""Model-name normalization (OpenRouter <-> Claude Code), reasoning-effort
resolution, and the /v1/models catalog (fetch_model_list)."""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from claudecode_as_openai.constants import _OPENROUTER_ALIASES

# When a caller doesn't pass `reasoning` at all (e.g. Hermes on a custom
# provider URL, which skips extra_body.reasoning to avoid 400s on unknown
# backends), fall back to this effort level rather than silently disabling
# thinking. Set to the effort level configured in Hermes's reasoning_effort
# config or any other value from _REASONING_EFFORT_MAP. Empty string or
# absent = no default effort (thinking disabled unless explicitly requested).
_DEFAULT_EFFORT_ENV = os.environ.get("CLAUDE_OPENAI_DEFAULT_EFFORT", "").strip().lower() or None

# Hardcoded last-resort fallback for /v1/models when neither OAuth nor
# an API key is available to query the real Anthropic /v1/models
# endpoint (see fetch_model_list()). Extracted from strings embedded
# in the compiled `claude` binary (2026-08-14) -- will go stale as new
# models ship, which is exactly why the live query is preferred.
KNOWN_MODEL_ALIASES = list(_OPENROUTER_ALIASES)

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
    # e.g. "claude-opus-4-8-20260528" or "claude-haiku-4-5-20251001"
    m = re.match(
        r"^claude-(opus|sonnet|haiku|fable)-(\d+)-(\d+)(?:-\d{8})?$",
        model_id,
    )
    if m:
        family, major, minor = m.group(1), m.group(2), m.group(3)
        return f"anthropic/claude-{family}-{major}.{minor}"
    # e.g. "claude-opus-5" or "claude-fable-5" (no minor)
    m2 = re.match(r"^claude-(opus|sonnet|haiku|fable)-(\d+)$", model_id)
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
    lower = model_id.lower()
    # Any Claude 3 model (3-haiku, 3.5-haiku, etc.)
    if re.search(r"claude-3", lower):
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
