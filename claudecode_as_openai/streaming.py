#!/usr/bin/env python3
"""Spawns `claude -p` and turns its NDJSON stream-json output into this
shim's internal result shape: one-shot cold calls (call_claude_streaming),
the shared chunk-parsing loop used by both the cold and warm-pool paths
(_consume_claude_response), and the bounded tool-dispatch retry loop
(call_claude_with_tool_retry)."""

import json
import os
import pty
import subprocess
import sys
import tempfile
import time
import uuid

from claudecode_as_openai.constants import (
    _CLAUDE_BIN, CLAUDE_TIMEOUT_S, EXTENDED_DISALLOWED_TOOLS, TOOL_CALL_MAX_RETRIES,
)
from claudecode_as_openai import state
from claudecode_as_openai.errors import ClaudeCliError, _classify_error_text
from claudecode_as_openai.tools import build_mcp_tool_config, _gen_tool_id, strip_mcp_tool_prefix
from claudecode_as_openai.messages import _flatten_content
from claudecode_as_openai.parsing import _iter_ndjson_lines, _iter_ndjson_lines_pty
from claudecode_as_openai import tracking
from claudecode_as_openai.tracking import logger


def _normalize_stop_sequences(stop):
    """Shared with WarmProcess.send_turn: OpenAI's `stop` is a string or
    list of up to 4 strings; Claude Code has no native stop-sequence
    flag, so this shim always emulates it client-side against the
    accumulated text (see _apply_stop_sequences)."""
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else [s for s in stop if s]


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


def _build_claude_cmd(model, session_mode, session_id, tools_requested, max_turns, json_schema, want_partial_messages=False, mcp_tool_config=None, effort=None, input_format=None):
    cmd = [_CLAUDE_BIN]
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
    elif session_mode == "fork":
        # Used by call_claude_with_tool_retry on retry: --fork-session
        # resumes `session_id`'s transcript (billed as a cache hit, not
        # a full resend -- verified live) into a CLI-assigned new session
        # id, leaving the original session's own transcript untouched.
        # See _consume_claude_response for where that new id is captured
        # back out of the "system"/"init" chunk.
        cmd += ["--resume", session_id, "--fork-session"]
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
    resolved_session_id = None
    finish_reason = "stop"
    stop_matched = False
    streaming_partial_text = ""

    for chunk in chunk_source:
        if time.time() > deadline:
            break
        ctype = chunk.get("type")

        if ctype == "system":
            if chunk.get("subtype") == "init":
                # The one place the CLI-assigned session id is ever
                # observable -- needed because --fork-session (see
                # _build_claude_cmd) means the caller doesn't get to
                # choose it. Echoes back whatever --session-id/--resume
                # value was passed for the non-forking modes too, so
                # callers can uniformly trust this over their own
                # locally-tracked id.
                resolved_session_id = chunk.get("session_id")
            continue

        if ctype == "rate_limit_event":
            info = chunk.get("rate_limit_info") or {}
            state._rate_limit_cache = info
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
                    # in server.py) -- verified live via a stream-json
                    # capture with a math prompt: setting --effort
                    # genuinely produces this block (with a real
                    # `signature` field) ahead of the final `text`
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
            "session_id": resolved_session_id,
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
        "session_id": resolved_session_id,
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
    path (see server.py's _handle_chat_completion): with tool retry in play,
    a failed attempt's narration text must not reach the client before the
    shim knows to discard it and retry.

    `tools`, if given, is registered as a real MCP tool server (see
    build_mcp_tool_config) instead of being described in prose -- the fix
    for the ~40% single-shot tool-call reliability problem, reaching 100%
    turn-1 dispatch.

    Returns a dict: {"text", "tool_calls", "usage", "finish_reason",
    "structured_json"}. Raises ClaudeCliError for conditions that should
    become an OpenAI-shaped error response -- see _classify_error_text.

    See EXTENDED_DISALLOWED_TOOLS and _build_claude_cmd for why
    --disallowedTools (not --tools "") is used when tools are requested,
    and why the no-tools path uses --tools "" + --strict-mcp-config
    instead."""
    if session_id is None:
        session_id = str(uuid.uuid4())

    stop_sequences = _normalize_stop_sequences(stop)

    system_prompt_file = None
    mcp_tool_config = build_mcp_tool_config(tools) if tools else None
    # --json-schema needs a couple of internal turns (an internal
    # "StructuredOutput" tool call + a corrective retry if the model
    # forgets it) to actually enforce the schema -- max_turns=1 leaves it
    # hanging at error_max_turns. MCP tool dispatch both break out on
    # tool_use before a second turn matters, so max_turns stays >= 2 only
    # when json_schema is set.
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

    redacted_cmd = tracking.redact_cmd_for_log(cmd)
    logger.debug(
        "spawning claude (cold): %s | session_id=%s session_mode=%s",
        redacted_cmd, session_id, session_mode,
    )
    with tracking.span("claude_spawn", cmd=redacted_cmd, path="cold") as sp:
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
                    cwd=state._CLAUDE_CWD,
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
                    cwd=state._CLAUDE_CWD,
                )
        except FileNotFoundError:
            if system_prompt_file:
                try:
                    os.unlink(system_prompt_file)
                except OSError:
                    pass
            raise ClaudeCliError(
                503, "api_error",
                f"'{_CLAUDE_BIN}' CLI not found on PATH. Install Claude Code "
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
            result = _consume_claude_response(chunk_source, deadline, stop_sequences, stream_callback)
            # Self-consistency fallback for the non-forking modes (and for
            # any test/mock chunk_source that omits the "system"/"init"
            # line): the CLI always echoes back whatever --session-id/
            # --resume value was passed, so this is a no-op in practice
            # except for session_mode="fork", where it's the only source
            # of truth for the id the CLI actually assigned.
            if not result.get("session_id"):
                result["session_id"] = session_id
            usage = result.get("usage") or {}
            sp.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
            sp.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
            return result
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
            # mcp_tool_config["manifest_path"], if any, is NOT unlinked here:
            # it's a cached, potentially-shared file owned by tools.py's
            # manifest cache (see _manifest_path_for) so a stable tool set
            # keeps a byte-identical --mcp-config across calls -- deleting
            # it after every single call would defeat that entirely.


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
# text instead of a real tool_use block. Retrying turns that into a much
# higher practical success rate at the cost of extra latency/spend only on
# the failing path. No deterministic fix exists for this failure mode
# anywhere in the Claude Code ecosystem (checked
# @anthropic-ai/claude-agent-sdk: retries transport/API errors only, no
# tool_choice:required-equivalent); every implementation handles it with
# a retry loop.
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
    """Wraps call_claude_streaming with a bounded, OFF-BY-DEFAULT retry for
    the documented tool-dispatch flakiness (TOOL_CALL_MAX_RETRIES defaults
    to 0 -- see constants.py for why). The FIRST attempt always uses
    whatever session mode was resolved by resolve_session() (a resumed
    session sending only the new delta messages, or a fresh session
    sending everything) so the common, successful case gets the full
    caching benefit, unaffected by whether retries are enabled at all.

    If a retry IS enabled and needed (tools were requested but no
    tool_use came back), it does NOT reuse the same --session-id (causes
    an "already in use" CLI error) or resume with more delta messages
    (would duplicate entries in that session's persisted transcript).
    Instead:
      - If the ORIGINAL call was session_mode="resume", every retry forks
        a brand-new session from that SAME original checkpoint (never
        from a previous retry's own session) via --fork-session, resending
        only the delta -- verified live: a fork is billed as a cache hit
        of the source's full context, not a resend, and leaves the
        source session's own transcript untouched. Every retry is
        therefore an equally "clean" re-ask, just cheap.
      - If the ORIGINAL call was session_mode="fresh" (this conversation's
        very first turn), there is no pre-existing Claude-side session to
        fork from -- retries fall back to a brand-new session with the
        full history resent, same as before this existed.
    Returns (result_dict, final_session_mode, final_session_id) so the
    caller knows which session to record for the next external turn --
    final_session_id is the CLI-CONFIRMED id from the winning attempt's
    "system"/"init" chunk (see _consume_claude_response), not necessarily
    the id this function itself picked, since --fork-session means the
    CLI chooses it.

    `tools`, if given, is forwarded to call_claude_streaming for native MCP
    tool registration (see build_mcp_tool_config) -- this is what actually
    fixes the tool-dispatch reliability this retry loop used to paper
    over pre-MCP; verified live to reach 100% turn-1 dispatch when a tool
    call is actually warranted. With MCP handling that, "no tool_call"
    overwhelmingly means "no tool was needed", not "dispatch failed" --
    which is why retrying is no longer the default. Kept as a safety net
    for tools that can't be represented as MCP tool names (dropped, see
    server.py) and for any future Claude Code regression: set
    TOOL_CALL_MAX_RETRIES > 0 to re-enable.

    `stream_callback`, if given, should only ever be passed by the caller
    when tools_requested is False -- with tools in play, a failed
    attempt's narration text must NOT reach the client before the shim
    knows to discard it and retry (see call_claude_streaming's own
    docstring). Enforced here defensively too: the callback is only ever
    forwarded to call_claude_streaming when tools_requested is False,
    regardless of what the caller passed in.

    Every attempt spawns a real `claude` subprocess and burns real,
    billed tokens -- including ones that only narrate instead of
    dispatching a tool and get discarded. `result["usage"]` on return is
    the SUM across every attempt this call made, not just the winning
    one, so callers (and usage-tracking events) see true billed usage
    rather than under-reporting whenever a retry happened."""
    result = None
    accumulated_usage = {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }
    origin_mode, origin_id = session_mode, session_id
    cur_mode, cur_id, cur_messages = session_mode, session_id, delta_claude_messages
    total_attempts = 1 + TOOL_CALL_MAX_RETRIES
    for attempt in range(total_attempts):
        cb = stream_callback if not tools_requested else None
        with tracking.span("tool_retry_attempt", attempt=attempt, session_mode=cur_mode):
            result = call_claude_streaming(
                cur_messages, system_prompt, model,
                session_mode=cur_mode, session_id=cur_id,
                tools_requested=tools_requested,
                tools=tools,
                stop=stop,
                stream_callback=cb,
                effort=effort,
            )
        actual_id = result.get("session_id") or cur_id
        attempt_usage = result.get("usage") or {}
        for key in accumulated_usage:
            accumulated_usage[key] += attempt_usage.get(key, 0)
        if result["tool_calls"] or not tools_requested:
            result["usage"] = dict(accumulated_usage)
            return result, cur_mode, actual_id
        if attempt < total_attempts - 1:
            time.sleep(_backoff_delay_s(attempt))
            if origin_mode == "resume":
                cur_mode, cur_id, cur_messages = "fork", origin_id, delta_claude_messages
            else:
                cur_mode, cur_id, cur_messages = "fresh", str(uuid.uuid4()), full_claude_messages
    result["usage"] = dict(accumulated_usage)
    actual_id = result.get("session_id") or cur_id
    return result, cur_mode, actual_id
