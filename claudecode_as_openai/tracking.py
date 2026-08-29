"""Observability seam: pluggable usage tracking, DEBUG command-line
logging (with redaction), and optional OTEL/Logfire span tracing. Every
piece here is best-effort and MUST degrade to a true no-op when its
optional dependency (litellm, logfire) isn't installed or its env var
isn't set -- nothing in this module may ever slow down or fail a
request. Zero dependencies on any other claudecode_as_openai module, so
it can be imported from server.py, streaming.py, and warm_pool.py
without creating an import cycle."""

import contextlib
import json
import logging
import os
import shlex
import sys

logger = logging.getLogger("claudecode_as_openai")


# ---------------------------------------------------------------------------
# Usage tracking: a pluggable seam so operators can wire up cost tracking,
# a JSONL audit log, or a custom sink without touching the request path.
# ---------------------------------------------------------------------------

class UsageTracker:
    """Plugin contract: implement record() to receive one usage event per
    completion (see server.py's _handle_chat_completion/
    _handle_streaming_completion for the event shape)."""

    def record(self, event: dict) -> None:
        raise NotImplementedError


_TRACKERS = []


def register_tracker(tracker):
    _TRACKERS.append(tracker)


def emit_usage(event: dict) -> None:
    """Calls every registered tracker's record(event) in registration
    order. A tracker mutating `event` (e.g. attaching cost_usd) is visible
    to trackers registered after it -- see init_trackers_from_env for why
    registration order matters. Tracking must never fail or slow a
    request (same best-effort contract as warm-pool parking elsewhere in
    this codebase), so each tracker's failure is swallowed independently
    -- one broken tracker must not stop the others from recording."""
    for tracker in _TRACKERS:
        try:
            tracker.record(event)
        except Exception:
            pass


class LiteLLMCostTracker(UsageTracker):
    """Attaches event["cost_usd"] via litellm's model pricing table.
    litellm is an optional dependency (see pyproject.toml's `tracking`
    extra) -- imported lazily here so the shim runs fine without it."""

    def record(self, event: dict) -> None:
        try:
            import litellm
        except ImportError:
            return
        cost = litellm.completion_cost(
            model=event["model"],
            completion_response={"model": event["model"], "usage": event["usage"]},
        )
        event["cost_usd"] = cost


class JsonlFileTracker(UsageTracker):
    """Appends one JSON line per usage event to `path` (see
    CLAUDE_OPENAI_USAGE_LOG)."""

    def __init__(self, path):
        self.path = path

    def record(self, event: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")


def init_trackers_from_env():
    """Registers trackers based on env vars. CLAUDE_OPENAI_TRACK_COST is
    registered BEFORE CLAUDE_OPENAI_USAGE_LOG: trackers run in
    registration order and mutate the shared event dict, so the cost
    plugin must attach cost_usd before the file sink serializes the
    event, or the JSONL log would miss it."""
    if os.environ.get("CLAUDE_OPENAI_TRACK_COST", "").strip() == "1":
        register_tracker(LiteLLMCostTracker())
    usage_log_path = os.environ.get("CLAUDE_OPENAI_USAGE_LOG", "").strip()
    if usage_log_path:
        register_tracker(JsonlFileTracker(usage_log_path))


# ---------------------------------------------------------------------------
# DEBUG-level logging, with redaction for prompt/schema content.
# ---------------------------------------------------------------------------

def configure_logging_from_env():
    """Configures the shared `logger` from CLAUDE_OPENAI_LOG_LEVEL
    (default INFO). Call once, from main(), before serve_forever()."""
    level_name = os.environ.get("CLAUDE_OPENAI_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level, stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger.setLevel(level)


_REDACT_FLAGS = (
    "--system-prompt", "--json-schema",
    "--mcp-config", "--disallowedTools", "--allowedTools",
)


def redact_cmd_for_log(cmd):
    """Returns a shell-quoted string of `cmd` with the value following each
    of _REDACT_FLAGS replaced by <len=N>. --system-prompt/
    --json-schema are redacted so DEBUG logs never leak prompt/schema
    content; --mcp-config/--disallowedTools/--allowedTools are redacted
    purely to cut noise -- their values are long (a JSON blob, or the
    ~25-entry built-in tool list) and rarely the thing worth reading in a
    DEBUG line, unlike the flag's presence/position itself. --system-prompt-
    file is left alone (a path, not content). Never mutates the input list.
    User message content is never in argv at all (sent via stdin), so
    there's nothing to redact for that."""
    redacted = list(cmd)
    i = 0
    while i < len(redacted):
        if redacted[i] in _REDACT_FLAGS and i + 1 < len(redacted):
            value = redacted[i + 1]
            redacted[i + 1] = f"<len={len(value)}>"
            i += 2
        else:
            i += 1
    return shlex.join(redacted)


# ---------------------------------------------------------------------------
# Optional OTEL/Logfire span tracing.
# ---------------------------------------------------------------------------

_tracing_enabled = False


def configure_tracing_from_env():
    """Enables tracing only when CLAUDE_OPENAI_TRACING=1 AND both
    importing and configuring logfire succeed (LOGFIRE_TOKEN is read by
    logfire.configure() itself). Any failure here leaves tracing
    disabled -- span() then no-ops for the whole process lifetime."""
    global _tracing_enabled
    if os.environ.get("CLAUDE_OPENAI_TRACING", "").strip() != "1":
        return
    try:
        import logfire
        logfire.configure()
    except Exception:
        logger.debug("logfire configure failed; tracing disabled", exc_info=True)
        return
    _tracing_enabled = True


class _NoopSpan:
    """Stand-in span used whenever tracing is disabled/unavailable/broken.
    Any attribute access returns a no-op callable, so `span.set_attribute(...)`
    (or any other logfire span method) is always safe to call."""

    def __getattr__(self, name):
        return lambda *a, **kw: None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@contextlib.contextmanager
def span(name, **attrs):
    """Best-effort tracing span. Never raises on its own account: a
    logfire failure (misconfiguration, set_attribute erroring, whatever)
    is swallowed and never masks or replaces an exception raised by the
    wrapped code, which always propagates untouched. Yields a real
    logfire span (usable as `sp.set_attribute(key, value)`) when tracing
    is enabled and healthy, otherwise a _NoopSpan."""
    real_cm = None
    real_span = _NoopSpan()
    if _tracing_enabled:
        try:
            import logfire
            real_cm = logfire.span(name, **attrs)
            real_span = real_cm.__enter__()
        except Exception:
            real_cm = None
            real_span = _NoopSpan()

    try:
        yield real_span
    except BaseException:
        if real_cm is not None:
            try:
                real_cm.__exit__(*sys.exc_info())
            except Exception:
                pass
        raise
    else:
        if real_cm is not None:
            try:
                real_cm.__exit__(None, None, None)
            except Exception:
                pass
