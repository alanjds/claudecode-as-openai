#!/usr/bin/env python3
"""OpenAI-chat-completions-compatible shim over the local Claude Code CLI:
the HTTP request handler and server entry point. Named server.py, not
http.py, because a package submodule named `http` shadows the stdlib
`http` package for any code in this process that does `import http`
(urllib.request does exactly this) once this package's own directory
ends up on sys.path -- as it does when this file (or shim.py) is run
directly as a script rather than via `python -m`. Translates OpenAI's
`/v1/chat/completions` API onto `claude -p` (Claude Code's non-interactive
mode): native tool-calling via MCP tool registration, session caching via
--session-id/--resume, OpenAI-shaped error translation, response_format
via --json-schema, OpenRouter model-name and reasoning-effort
compatibility, and stop-sequence emulation.

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
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from claudecode_as_openai.constants import (
    DEFAULT_MODEL, MAX_N_CHOICES, _BASE_ENV_OVERRIDES, WARM_POOL_DISABLED, TOOL_CALL_MAX_RETRIES,
)
from claudecode_as_openai import state
from claudecode_as_openai.errors import ClaudeCliError
from claudecode_as_openai.models import fetch_model_list, normalize_model_name, resolve_reasoning_effort
from claudecode_as_openai.messages import build_claude_messages, system_prompt_from_messages
from claudecode_as_openai.tools import build_mcp_tool_manifest, build_mcp_tool_config
from claudecode_as_openai.sessions import resolve_session, record_session
from claudecode_as_openai.streaming import (
    call_claude_streaming, call_claude_with_tool_retry, build_reasoning_details,
    _build_openai_usage, _apply_stop_sequences,
)
from claudecode_as_openai.warm_pool import WarmProcess, _parking_fingerprint
from claudecode_as_openai.quota import _build_key_response, _build_credits_response, _log_quota_snapshot
from claudecode_as_openai import tracking
from claudecode_as_openai.tracking import logger

DEFAULT_PORT = int(os.environ.get("CLAUDE_OPENAI_PORT", "8977"))

# Eager on import (matching the original monolith's module-level
# `_CLAUDE_CWD = tempfile.mkdtemp(...)` / `_WARM_POOL = WarmPool()`):
# Handler methods assume both are already real values, not None, from the
# moment this module is importable -- see state.initialize().
state.initialize()

_UNSUPPORTED_SAMPLING_PARAMS = (
    "temperature", "top_p", "seed", "logprobs", "top_logprobs",
    "presence_penalty", "frequency_penalty", "logit_bias",
)
_warned_sampling_params = set()


def _warn_unsupported_sampling_params(payload):
    """Emit a one-time-per-parameter-name stderr warning when a request
    includes a sampling parameter Claude Code has no way to honor. Does
    NOT reject the request -- these are silently accepted (never
    hard-errored) because real OpenAI clients routinely send explicit
    defaults on every request (e.g. `temperature: 1.0`, the OpenAI
    default itself, carries no signal the caller wants non-default
    behavior) -- hard-erroring on presence-of-key would break
    compatibility with well-behaved clients for no benefit."""
    for name in _UNSUPPORTED_SAMPLING_PARAMS:
        if name in payload and payload[name] is not None and name not in _warned_sampling_params:
            _warned_sampling_params.add(name)
            sys.stderr.write(
                f"claudecode-as-openai: warning: '{name}' was requested but Claude "
                f"Code has no equivalent (no CLI flag, no env var) -- ignored, not "
                f"applied. See README \"Capability audit\".\n"
            )


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


def _build_usage_event(chat_id, model, usage, stream, n):
    """Shared shape for the usage event handed to tracking.emit_usage --
    see the two call sites in _handle_chat_completion and
    _handle_streaming_completion."""
    return {
        "id": chat_id, "model": model, "usage": usage,
        "created": int(time.time()), "stream": stream, "n": n,
    }


@contextmanager
def _scoped_env_overrides(env_overrides):
    """Temporarily patches subprocess.Popen so any process it spawns
    while this context is active inherits `env_overrides` merged into
    the current environment. Used to scope CLAUDE_CODE_MAX_OUTPUT_TOKENS
    (from `max_tokens`) to a single completion's subprocess call without
    mutating the shim's own process-wide environment."""
    import subprocess

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
        info = state._rate_limit_cache
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
        info = state._rate_limit_cache
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
        with state._WARM_POOL._lock:
            parked = state._WARM_POOL._parked
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
            if logger.isEnabledFor(10):  # 10 == logging.DEBUG
                import traceback
                logger.debug("unhandled exception in _handle_chat_completion\n%s",
                             traceback.format_exc())
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

        with tracking.span("chat_completion", model=model, n=n, stream=stream):
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
                with tracking.span("choice", index=choice_index):
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
                chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                tracking.emit_usage(_build_usage_event(
                    chat_id, model, _build_openai_usage(usage_totals), stream=True, n=n,
                ))
                self._send_stream_chunks(model, choices, chat_id=chat_id)
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
            tracking.emit_usage(_build_usage_event(
                response["id"], model, response["usage"], stream=False, n=n,
            ))
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
            if WARM_POOL_DISABLED:
                return
            try:
                warm = WarmProcess(
                    fingerprint, final_sid, model, system_prompt, False, None,
                    None, None, use_pty=True, env_overrides=env_overrides,
                )
                state._WARM_POOL.park(warm)
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
                    state._WARM_POOL.take_if_matching(fingerprint)
                    if fingerprint is not None and session_mode == "resume" and not WARM_POOL_DISABLED
                    else None
                )
                if warm is not None:
                    logger.debug("turn dispatch: path=warm session_id=%s", session_id)
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
                    logger.debug("turn dispatch: path=cold session_id=%s session_mode=%s", session_id, session_mode)
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

        # Live streaming's result["usage"] uses the same internal keys as
        # the buffered path's per-attempt usage -- remap into the
        # usage_totals shape so _build_openai_usage (already imported,
        # already the one source of truth for the cache/prompt_tokens_details
        # nesting logic) can build the same flat OpenAI shape here too,
        # rather than reimplementing it. n is always 1 on this path (see
        # the can_stream_live gate in _handle_chat_completion).
        raw_usage = result.get("usage") or {}
        usage_totals = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
            "cache_read_input_tokens": raw_usage.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": raw_usage.get("cache_creation_input_tokens", 0),
        }
        tracking.emit_usage(_build_usage_event(
            chat_id, model, _build_openai_usage(usage_totals), stream=True, n=1,
        ))

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
        process per concurrent conversation.

        Tool-result-continuation resumes (see sessions.py's delta
        widening) used to be excluded from the warm pool here: the old
        --input-format stream-json WarmProcess implementation was proven
        to reliably cause Claude to redo a tool call on this exact shape,
        independent of the widening fix that made the cold path safe.
        WarmProcess no longer uses that input mode at all (see
        warm_pool.py's module docstring) -- it's a plain one-shot `-p`
        process like the cold path, just pre-spawned with its stdin held
        open -- so that exclusion no longer applies and tool-continuation
        resumes now flow through this same warm-pool gate like any other
        resumed turn."""
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
            if WARM_POOL_DISABLED:
                return
            try:
                warm = WarmProcess(
                    fingerprint, final_sid, model, system_prompt, tools_requested, tools,
                    mcp_tool_config, effort, use_pty=False, env_overrides=env_overrides,
                )
                state._WARM_POOL.park(warm)
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

            if fingerprint is not None and session_mode == "resume" and not WARM_POOL_DISABLED:
                warm = state._WARM_POOL.take_if_matching(fingerprint)
                if warm is not None:
                    logger.debug("turn dispatch: path=warm session_id=%s", session_id)
                    try:
                        result = warm.send_turn(delta_messages, stop=stop)
                    except ClaudeCliError:
                        warm.kill()
                        raise
                    if result["tool_calls"] or not tools_requested or TOOL_CALL_MAX_RETRIES == 0:
                        # A WarmProcess serves exactly one turn, ever --
                        # kill it now that this turn is done, same as
                        # the streaming path (see there for the leak
                        # this fixes: an unkilled spent process just
                        # sits alive indefinitely).
                        #
                        # The TOOL_CALL_MAX_RETRIES==0 case here matters:
                        # without it, a tool-continuation's normal closing
                        # turn (which never calls a tool -- it just answers)
                        # would get its perfectly good warm-served text
                        # response THROWN AWAY and regenerated from scratch
                        # on the cold path, on every multi-round tool
                        # conversation, once tool-continuation resumes
                        # started being served warm at all. That's exactly
                        # the "no tool call != dispatch failure" case
                        # TOOL_CALL_MAX_RETRIES already exists to not retry
                        # by default (see constants.py) -- this warm/cold
                        # dispatch decision just hadn't been updated to
                        # agree with it. Only fall through to a cold retry
                        # when retries are deliberately enabled, matching
                        # what a first cold attempt would already do.
                        warm.kill()
                        _park_next(session_id)
                        return result, session_mode, session_id
                    # Warm-served turn wanted a tool call, didn't get one,
                    # and retries are deliberately enabled
                    # (TOOL_CALL_MAX_RETRIES > 0): fall through to the cold
                    # retry loop exactly like a first cold attempt would
                    # (see call_claude_with_tool_retry) -- the warm process
                    # is already spent (one turn each) and not reused.
                    warm.kill()

            logger.debug("turn dispatch: path=cold session_id=%s session_mode=%s", session_id, session_mode)
            result, final_mode, final_id = call_claude_with_tool_retry(
                delta_messages, full_messages, system_prompt, model,
                tools_requested=tools_requested, session_mode=session_mode, session_id=session_id,
                tools=tools, stop=stop, effort=effort,
            )
            if fingerprint is not None:
                _park_next(final_id)
            return result, final_mode, final_id


    def _send_stream_chunks(self, model, choices, chat_id=None):
        """Emits SSE chunks. NOTE: this is protocol-shaped streaming, not
        real token streaming -- each choice's full text/tool_calls are
        already fully computed by the time this runs (Claude Code's
        stream-json mode streams messages, not token deltas within a
        message), so each choice arrives as a single content delta after
        the full latency. See README "Capability audit" for measurements."""
        if chat_id is None:
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
    tracking.configure_logging_from_env()
    tracking.configure_tracing_from_env()
    tracking.init_trackers_from_env()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Claude Code shim (native tool-calling + session caching) listening on http://127.0.0.1:{port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
