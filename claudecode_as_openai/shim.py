#!/usr/bin/env python3
"""OpenAI-chat-completions-compatible shim over the local Claude Code CLI,
built the way Cline's "Claude Code" provider actually does it (verified
against cline/cline's open-source ClaudeCodeHandler + runClaudeCode,
2026-08-12) rather than the ad-hoc JSON-envelope-in-prose approach this
file used previously.

Key differences from the old approach, and why they fix the flakiness:

1. Uses Claude Code's REAL native tool-calling protocol
   (`--output-format stream-json`, parsing genuine `tool_use` content
   blocks from the assistant message) instead of asking Claude to emit a
   custom JSON envelope in plain text. Claude never has to be talked into
   "pretending" to be an API backend -- it just does normal tool calling,
   which is reliable by construction.

2. Uses `--disallowedTools <full built-in list>` (not `--tools ""`) so
   Claude Code's own built-in tools (Bash, Read, Write, memory/skill tools,
   etc.) are blocked by name, but ANY custom tool names Hermes passes in
   the request's `tools` array remain callable. This is why the old
   `--tools ""` approach was flaky: Claude would still *attempt* a
   built-in tool_use (memory, skill, ...) despite the empty allowlist,
   which triggered `--max-turns 1` failures. Naming the built-ins
   explicitly leaves room for real, user-supplied tools to work.

3. Parses the NDJSON stream incrementally and returns as soon as a
   complete assistant message is seen (whether it's a `tool_use` content
   block or plain text) -- it does NOT wait for the subprocess to exit.
   If the CLI's own harness then tries to self-resolve a tool call it
   doesn't recognize and eventually hits `--max-turns`/exits non-zero,
   that happens after we've already extracted what we need, so it no
   longer surfaces as a shim-level failure.

4. Messages are passed as a JSON array on stdin (Claude Code's native
   message format), with the system prompt passed via `--system-prompt`
   (piped through a tempfile via `--system-prompt-file` when large, since
   `--system-prompt` alone can still hit ARG_MAX for big Hermes system
   prompts).

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
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLAUDE_BIN = "claude"
DEFAULT_MODEL = "sonnet"
CLAUDE_TIMEOUT_S = 300

# NOTE: previously used --disallowedTools with an enumerated built-in list
# (see BUILTIN_TOOLS_TO_DISALLOW below, kept as a documented dead-end).
# Switched to --tools "" per cline/cline#10336 -- see call_claude_streaming.
BUILTIN_TOOLS_TO_DISALLOW = ",".join(
    [
        "Task", "TaskOutput", "Bash", "Glob", "Grep", "Read", "Edit", "Write",
        "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch", "TaskStop",
        "AskUserQuestion", "Skill", "EnterPlanMode", "ExitPlanMode",
        "EnterWorktree", "ExitWorktree", "CronCreate", "CronDelete",
        "CronList", "ToolSearch",
    ]
)  # unused now -- kept only as a documented example of the approach that
   # did NOT reliably suppress native dispatch (0/8 in testing).


def _flatten_content(content):
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return content or ""


def build_claude_messages(openai_messages):
    """Translate OpenAI chat messages into Claude Code's native message
    array shape: [{"role": "user"|"assistant", "content": <str or blocks>}].
    Tool results become a user message with a tool_result content block
    (Claude's native format), and prior assistant tool_calls become a
    tool_use content block -- this keeps multi-turn tool loops coherent
    across separate `claude -p` invocations, since each call is otherwise
    stateless (no --resume; see the caching note in claude_code_shim skill)."""
    claude_messages = []
    system_prompt_parts = []

    for m in openai_messages:
        role = m.get("role", "user")
        if role == "system":
            system_prompt_parts.append(_flatten_content(m.get("content")))
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

    system_prompt = "\n\n".join(p for p in system_prompt_parts if p)
    return claude_messages, system_prompt


def render_tools_into_system_prompt(tools, base_system_prompt):
    """Custom tools aren't passed via a CLI flag (Claude Code's -p mode has
    no --tools-schema equivalent for arbitrary JSON-schema tools) -- they're
    described in the system prompt. Framing matters a lot here: describing
    them as "custom"/hypothetical made Claude hedge and refuse to call them
    (0/8 in testing) even with native tool_use active. Framing them as
    already-wired, real, implemented-by-the-harness tools fixed this
    completely (6/6) -- combined with --tools "" (see call_claude_streaming)
    removing Claude's own competing built-in dispatch options entirely."""
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


def _iter_ndjson_lines(proc):
    for raw_line in proc.stdout:
        line = raw_line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def call_claude_streaming(claude_messages, system_prompt, model):
    """Spawn `claude -p` in native stream-json mode, feed the message array
    on stdin, and return as soon as a usable assistant message (text and/or
    tool_use blocks) is seen -- mirrors Cline's approach of yielding from
    the stream incrementally rather than waiting for full completion.
    Terminates the subprocess immediately after extracting what's needed so
    a downstream max-turns/self-resolution failure never surfaces here.

    Tool suppression, and why --disallowedTools (not --tools ""):
    cline/cline#10336 fixes a DIFFERENT bug (Cline's own prompt-format text
    getting misparsed against Claude's *native* Bash/Read/Edit vocabulary)
    by going to `--tools "" --strict-mcp-config ...`, which leaves Claude
    with NO native dispatch at all -- it then emits tool calls as bare text
    (Cline's <function_calls>/<invoke> XML or a raw {"name":...} JSON
    blob), which is why Cline ships a whole text/XML tool-call parser.
    Verified directly here: with --tools "", tool calls come back as
    `<function_calls>/<invoke>` or `<tool_call>{"name":...}` TEXT, not a
    tool_use content block (0/8 native tool_use hits).

    This shim wants real native tool_use blocks instead of a text
    convention that needs its own parser, so it keeps --disallowedTools
    with the full enumerated built-in list (BUILTIN_TOOLS_TO_DISALLOW) --
    that leaves native dispatch live for any tool NOT in the denylist,
    which is exactly the custom/Hermes-supplied tools described in the
    system prompt below. Combined with strongly-worded ("already
    implemented, not custom/hypothetical") tool framing, this reliably
    produces real tool_use blocks (6/6 in manual testing) instead of
    either a refusal (0/8 with weak framing) or a text-only fallback (0/8
    with --tools "")."""
    system_prompt_file = None
    cmd = [
        CLAUDE_BIN,
        "--disallowedTools", BUILTIN_TOOLS_TO_DISALLOW,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", "1",
    ]
    if system_prompt:
        # A large Hermes system prompt can still hit ARG_MAX as a bare CLI
        # arg; write it to a tempfile and use --system-prompt-file instead
        # (same escape hatch Cline's `shouldUseFile` option models).
        if len(system_prompt) > 4000:
            fd, system_prompt_file = tempfile.mkstemp(prefix="hermes-acp-sysprompt-", suffix=".txt")
            with os.fdopen(fd, "w") as f:
                f.write(system_prompt)
            cmd += ["--system-prompt-file", system_prompt_file]
        else:
            cmd += ["--system-prompt", system_prompt]
    if model:
        cmd += ["--model", model]
    cmd += ["-p"]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        proc.stdin.write(json.dumps(claude_messages))
        proc.stdin.close()

        text_parts = []
        tool_calls = []
        usage = {}
        deadline = time.time() + CLAUDE_TIMEOUT_S
        saw_error = None

        for chunk in _iter_ndjson_lines(proc):
            if time.time() > deadline:
                break
            ctype = chunk.get("type")

            if ctype == "assistant" and "message" in chunk:
                message = chunk["message"]
                if message.get("error"):
                    saw_error = message["error"]
                    break
                got_tool_use = False
                got_text = False
                for content in message.get("content", []):
                    ctype2 = content.get("type")
                    if ctype2 == "text" and content.get("text"):
                        text_parts.append(content["text"])
                        got_text = True
                    elif ctype2 == "tool_use":
                        tool_calls.append(
                            {
                                "id": content.get("id", f"toolu_{uuid.uuid4().hex[:20]}"),
                                "name": content.get("name", ""),
                                "input": content.get("input", {}),
                            }
                        )
                        got_tool_use = True
                    # "thinking"/"redacted_thinking" blocks are intentionally
                    # ignored here -- they're extended-thinking output, not
                    # the final answer, and an assistant message can consist
                    # of ONLY a thinking block before the real tool_use/text
                    # message arrives on a later NDJSON line. Only stop once
                    # we've actually seen a tool_use or non-empty text block.
                if usage_data := message.get("usage"):
                    usage = usage_data
                if got_tool_use or got_text:
                    break
                continue

            if ctype == "result":
                if chunk.get("is_error") and chunk.get("subtype") not in ("error_max_turns",):
                    saw_error = chunk.get("result") or f"claude error: {chunk.get('subtype')}"
                if not usage and chunk.get("usage"):
                    usage = chunk["usage"]
                break

        if saw_error:
            raise RuntimeError(f"claude reported error: {saw_error}")

        return "".join(text_parts) or None, tool_calls, usage
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        if system_prompt_file:
            try:
                os.unlink(system_prompt_file)
            except OSError:
                pass


# Bounded retries when tools were requested but Claude answered with plain
# text instead of a real tool_use block (see call_claude_streaming
# docstring: --disallowedTools + strong framing gets ~40% single-shot
# native tool_use reliability, verified 6/15 over a fair sample on
# 2026-08-12). Retrying turns that into a much higher practical success
# rate at the cost of extra latency/spend only on the failing path.
#
# Parameters were tuned to match the retry decorator found in an OLD,
# now-deleted Cline commit (`@withRetry({maxRetries: 4, baseDelay: 2000,
# maxDelay: 15000})`, src/core/api/providers/claude-code.ts @ 9dea336c/
# 8a6441fd). Verified against Cline's CURRENT mainline (2026-08-12): that
# file no longer exists -- Cline now delegates entirely to the third-party
# `ai-sdk-provider-claude-code` npm package, which has no built-in retry
# decorator of its own. Checked the official `@anthropic-ai/claude-agent-sdk`
# too: it has a retry mechanism (SDKAPIRetryMessage) but only for
# transport/API errors, not "model narrated instead of dispatching a
# tool" -- no `tool_choice: required`-equivalent is exposed through the
# harness. Conclusion: no deterministic fix exists anywhere in this
# ecosystem for this exact failure mode; every implementation that
# handles it does so with a retry loop. Keeping these old-Cline-derived
# numbers is a reasonable choice since they were presumably tuned
# empirically, not because they're a known-current standard. See
# README.md "Investigated and ruled out" section for the full trail, and
# the (deliberately kept, unmerged) explore/text-tool-parser branch for
# the ruled-out alternative.
#
# (The `--tools "" --strict-mcp-config` + text-parser combo from
# cline/cline#10336 is a community workaround for a DIFFERENT, unrelated
# bug -- Cline's own agentic XML tool vocabulary colliding with Claude
# Code's native tools when routed through a different provider path -- it
# is not and never was Cline's Claude Code provider's actual approach.)
TOOL_CALL_MAX_RETRIES = 4  # matches Cline's maxRetries
TOOL_CALL_BASE_DELAY_S = 2.0  # matches Cline's baseDelay (ms -> s)
TOOL_CALL_MAX_DELAY_S = 15.0  # matches Cline's maxDelay (ms -> s)


def _backoff_delay_s(attempt_index):
    """Exponential backoff matching Cline's formula: baseDelay * 2^attempt,
    capped at maxDelay. attempt_index is 0-based (0 = first retry)."""
    return min(TOOL_CALL_BASE_DELAY_S * (2 ** attempt_index), TOOL_CALL_MAX_DELAY_S)


def call_claude_with_tool_retry(claude_messages, system_prompt, model, tools_requested):
    """Wraps call_claude_streaming with bounded retries specifically for the
    documented flakiness: when tools were requested and Claude comes back
    with plain text instead of a tool_use block, that's the known failure
    mode (not a real "no tool needed" answer -- Claude was asked to use a
    specific tool). Retry up to TOOL_CALL_MAX_RETRIES additional times
    (matching Cline's semantics: maxRetries is retries AFTER the first
    attempt, so total attempts = 1 + TOOL_CALL_MAX_RETRIES) with
    exponential backoff between attempts. Returns whatever the last
    attempt produced (including the failure case) so the caller still
    gets a valid OpenAI-shaped response either way."""
    text, tool_calls, usage = None, [], {}
    total_attempts = 1 + TOOL_CALL_MAX_RETRIES
    for attempt in range(total_attempts):
        text, tool_calls, usage = call_claude_streaming(claude_messages, system_prompt, model)
        if tool_calls or not tools_requested:
            # Either we got a real tool_use, or no tools were requested in
            # the first place (plain text is the correct, expected result).
            break
        # tools were requested but no tool_use came back -- the known
        # flakiness case. Retry (with backoff) unless this was the last attempt.
        if attempt < total_attempts - 1:
            time.sleep(_backoff_delay_s(attempt))
    return text, tool_calls, usage


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(
                {"object": "list", "data": [{"id": DEFAULT_MODEL, "object": "model"}]}
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
            messages = payload.get("messages", [])
            tools = payload.get("tools")
            model = payload.get("model") or DEFAULT_MODEL
            stream = bool(payload.get("stream"))

            claude_messages, system_prompt = build_claude_messages(messages)
            system_prompt = render_tools_into_system_prompt(tools, system_prompt)

            text, raw_tool_calls, usage = call_claude_with_tool_retry(
                claude_messages, system_prompt, model, tools_requested=bool(tools)
            )

            tool_calls = None
            if raw_tool_calls:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["input"]),
                        },
                    }
                    for tc in raw_tool_calls
                ]

            message = {"role": "assistant", "content": text}
            if tool_calls:
                message["tool_calls"] = tool_calls

            usage_obj = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            }
            finish_reason = "tool_calls" if tool_calls else "stop"

            if stream:
                self._send_stream_chunks(model, message, tool_calls, finish_reason, usage_obj)
                return

            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": usage_obj,
            }
            self._send_json(response)
        except Exception as e:
            self._send_json({"error": {"message": str(e), "type": "shim_error"}}, 500)

    def _send_stream_chunks(self, model, message, tool_calls, finish_reason, usage_obj):
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(delta, finish=None):
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

        emit({"role": "assistant"})
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                emit(
                    {
                        "tool_calls": [
                            {"index": i, "id": tc["id"], "type": "function", "function": tc["function"]}
                        ]
                    }
                )
        elif message.get("content"):
            emit({"content": message["content"]})
        emit({}, finish=finish_reason)
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
    print(f"Claude Code shim (Cline-style native tool-calling) listening on http://127.0.0.1:{port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
