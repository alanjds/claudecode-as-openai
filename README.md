# claudecode-as-openai

Exposes a local service with OpenAI-chat-completions compatible HTTP,
that uses shell calls to `claude` CLI (Claude Code),
so any OpenAI-client compatible tool/agent/harness can drive a Claude
subscription instead of using its metered API billing directly.

## Why this exists

Hermes's native `copilot-acp` provider can be pointed at `claude-agent-acp`
(see the `hermes-agent`/`claude-code-subscription-shim` skills), but that
path has no session caching and no real model-selection parameter. This shim
gets `--model`, `--session-id`/`--resume`, and several other OpenAI-shaped
API surfaces working reliably against a local Claude Code CLI/subscription.

## Installation

```console
pip install claudecode-as-openai
# or from source:
pip install .
```

## Quickstart

```console
# Start the shim (default port 8977; override with CLAUDE_OPENAI_PORT):
claudecode-as-openai

# Point Hermes at it:
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8977/v1
hermes config set model.api_key not-needed
hermes config set model.default sonnet
```

Or point any OpenAI-compatible client at `http://127.0.0.1:8977/v1`.

Alternate invocation forms:

```console
python3 -m claudecode_as_openai [port]
python3 -m claudecode_as_openai.shim [port]
claudecode-as-openai [port]
```

## Capabilities

Verified empirically against a live running shim.
Legend: ✅ works, ⚠️ partial/limited, ❌ not implemented.

| Capability | Status | Notes |
|---|---|---|
| Model selection | ✅ | `--model` passed per request. OpenRouter-style slugs (`anthropic/claude-sonnet-4.5`, `~anthropic/claude-sonnet-latest`, etc.) are translated automatically. |
| Session/conversation caching | ✅ | Fingerprints the conversation and resumes via `--resume`, sending only the new delta. Falls back to a fresh session on any history divergence. |
| Streaming (`stream: true`) | ✅ | Real token-level streaming for the common case (single choice, no tools, no `json_schema`). Falls back to buffered-then-emit for other combinations. |
| Tool calling (`tools`/`tool_calls`) | ✅ | Registered as real MCP tool schemas -- 100% turn-1 dispatch reliability. Tool names that violate MCP naming rules are silently dropped (no prose fallback). No retry by default when a turn declares tools but doesn't call one -- see "Native MCP tool registration" below. |
| `tool_choice: "none"` | ✅ | Full lockdown: built-ins and all MCP servers blocked, no tool descriptions added. |
| `tool_choice: "required"` / forced function | ❌ | No equivalent in the Claude Code CLI harness. |
| `max_tokens` / `max_completion_tokens` | ✅ | Mapped to `CLAUDE_CODE_MAX_OUTPUT_TOKENS` per subprocess call. |
| `stop` sequences | ✅ | Real early termination: subprocess killed the instant a stop sequence appears in the token stream, not post-hoc truncation. |
| `n` (multiple choices) | ⚠️ | Up to 5 independent subprocess calls. Requests above the limit return a 400. |
| `response_format: json_object` / `json_schema` | ✅ | Mapped to `--json-schema`. |
| `temperature`, `top_p`, `seed`, etc. | ⚠️ | No Claude Code CLI equivalent. Accepted silently; a one-time stderr warning is emitted per unknown parameter name. |
| `reasoning` (OpenRouter) | ✅ | Mapped to `--effort <low\|medium\|high\|max>`. Returned as `message.reasoning` (plaintext) and `message.reasoning_details` (OpenRouter's structured shape, signature preserved). |
| `usage.prompt_tokens_details.cached_tokens` | ✅ | Real cache-hit counts in OpenAI's nested shape and as flat custom keys (`cache_read_input_tokens`, `cache_creation_input_tokens`). |
| Vision / image blocks | ❌ | Silently dropped. |
| Error responses | ✅ | Translated to proper OpenAI `{"error": {"message", "type", "code"}}` with matching HTTP status (400/404/429/500/503). |
| `/v1/models` | ✅ | Queries the real Anthropic model list, using Claude Code's own OAuth token, then `ANTHROPIC_API_KEY`, then a hardcoded fallback. Cached 5 minutes. |
| Authentication | ❌ | Not needed. Ignored if provided. Fine as `127.0.0.1` binding is hardcoded. |

## Tests

```console
# Offline unit tests (no live claude needed):
python3 -m unittest tests.test_shim_unit -v

# Live integration tests (need a running shim and authenticated claude):
claudecode-as-openai &
python3 tests/test_plain_chat.py
python3 tests/test_tool_call_reliability.py --attempts 15
```

## Observability

All optional -- the shim runs exactly as before with none of these set, and
with none of the `tracking` extra's dependencies installed.

| Env var | Effect |
|---|---|
| `CLAUDE_OPENAI_LOG_LEVEL` | Sets the shim's log level (default `INFO`). Set to `DEBUG` to see the exact redacted `claude` command line spawned for every request (cold and warm-pool-init), plus which path (warm vs. cold) served each turn. |
| `CLAUDE_OPENAI_USAGE_LOG` | Path to a file. When set, one JSON line per completion (id, model, usage, timestamp, `stream`, `n`, and `cost_usd` if `CLAUDE_OPENAI_TRACK_COST=1`) is appended there. |
| `CLAUDE_OPENAI_TRACK_COST` | Set to `1` to attach a `cost_usd` field to every usage event via [LiteLLM](https://github.com/BerriAI/litellm)'s pricing table. Requires the `tracking` extra (`pip install "claudecode-as-openai[tracking]"`); a no-op otherwise. |
| `CLAUDE_OPENAI_TRACING` | Set to `1` to emit OTEL spans via [Logfire](https://logfire.pydantic.dev/), one nested per request / per `n`-choice / per tool-retry-attempt / per `claude` subprocess spawn -- exposes the retry loop, `n` fan-out, and warm-vs-cold amplification that's otherwise invisible from outside a single request. Requires the `tracking` extra; a no-op otherwise. |
| `LOGFIRE_TOKEN` | Read by Logfire itself when `CLAUDE_OPENAI_TRACING=1`; see Logfire's own docs for where to get one. |
| `CLAUDE_OPENAI_DISABLE_WARM_POOL` | Set to `1` to force every turn onto the cold path (a fresh `claude` subprocess per call), skipping the warm pool entirely -- an operational escape hatch for ruling out the warm pool while debugging, and how the warm-vs-cold redundant-tool-call comparison in "`--resume` and the redundant-tool-call bias" (below) was measured. |

**Redaction**: at `DEBUG` log level, the value following `--system-prompt`,
`--json-schema`, `--mcp-config`, `--disallowedTools`, and `--allowedTools` in
a logged command line is replaced with `<len=N>` -- the first two
because prompt/schema content should never be written to logs, the latter
three purely to cut noise (their values are a JSON blob or a long tool list,
rarely what you're looking for in a DEBUG line). User message content is
never logged at all (it's sent to `claude` over
stdin, never as a command-line argument).

Usage tracking is a pluggable seam (`claudecode_as_openai/tracking.py`): a
custom sink just needs to subclass `UsageTracker` and call
`register_tracker(...)` before serving requests.

```console
pip install "claudecode-as-openai[tracking]"
CLAUDE_OPENAI_USAGE_LOG=/var/log/claudecode-usage.jsonl \
CLAUDE_OPENAI_TRACK_COST=1 \
CLAUDE_OPENAI_LOG_LEVEL=DEBUG \
claudecode-as-openai
```

---

## Internals Lore

The sections below are for contributors and the curious. Users can stop here.

### OpenRouter model-name compatibility

The `model` field accepts Claude Code's native naming (`sonnet`,
`claude-sonnet-4-5`) and OpenRouter-style Anthropic slugs:

| Input | Normalized to |
|---|---|
| `~anthropic/claude-sonnet-latest` | `sonnet` |
| `anthropic/claude-opus-latest` | `opus` |
| `anthropic/claude-sonnet-4.5` | `claude-sonnet-4-5` (dot-to-dash; Claude Code rejects the dotted form) |
| `anthropic/claude-opus-4.8-fast` | `claude-opus-4-8` (fast-mode suffix stripped with a stderr warning) |
| anything else | passed through unchanged |

### OpenRouter reasoning-tokens compatibility

`reasoning` maps to Claude Code's `--effort` flag. Accepted shapes:

- `{"reasoning": {"effort": "high"}}` - mapped directly. OpenRouter's wider vocabulary (`none`, `minimal`, `xhigh`) is clamped to the nearest accepted value.
- `{"reasoning": {"max_tokens": N}}` - approximated to an effort level via banding.
- `{"reasoning": {"enabled": true}}` - maps to `medium`.

`reasoning.exclude: true` is accepted but not enforced: Claude Code always
includes thinking content when reasoning is active.

Reasoning is not compatible with live token-level streaming; those calls fall
back to the buffered-then-emit path.

### Native MCP tool registration

Claude Code's `-p` mode has no flag for passing arbitrary tool JSON schemas to
the Anthropic API directly. A prose-description approach was tried first and
reached only ~40% single-shot tool-call reliability.

The fix: register tools as a real MCP server via `--mcp-config` (a documented
`-p`-mode capability), with each tool marked
`_meta: {"anthropic/alwaysLoad": true}` to skip Claude Code's deferred
ToolSearch indirection and get the full schema into the initial prompt turn.
The shim kills the `claude` subprocess the instant it sees a `tool_use` block,
before `tools/call` ever reaches the MCP server -- dispatch and execution stay
entirely on the OpenAI-client side. Reliability: 5/5 repeatable through the
actual shim.

Tool prompt-cache stays intact for a stable tool set across resumed sessions;
changing the tool set mid-session correctly busts the cache, same as any
system-prompt change. This requires the manifest file backing `--mcp-config`
to itself stay byte-identical for an unchanged tool set: `build_mcp_tool_config`
caches it by tool-set content hash (LRU, bounded, evicted files unlinked)
rather than writing a fresh temp file -- and therefore a changed
`--mcp-config` -- on every single call regardless of whether the tool set
actually changed, which is what the original implementation did.

Tool names that can't satisfy MCP's `^[a-zA-Z0-9_-]{1,64}$` constraint are
silently dropped (there is no prose-description fallback path).

A turn that declares tools but gets no `tool_use` back is, with MCP handling
dispatch, now far more likely to mean "no tool was needed" than "dispatch
failed" -- so `CLAUDE_OPENAI_TOOL_CALL_MAX_RETRIES` defaults to `0` (no
retry). Set it above 0 to re-enable the bounded retry (2s base / 15s cap
backoff) as a safety net. When re-enabled, a retry on a *resumed*
conversation forks a new session from the original checkpoint via
`--fork-session` (billed as a cache hit of the existing context, not a
full-history resend, and never mutates the original session's own
transcript) instead of starting over with the full history, as it did
before `--fork-session` was adopted for this.

### `--resume` and the redundant-tool-call bias

A resumed turn that delivers a tool result normally sends only that one new
message (see "Session/conversation caching" above) -- this shim kills the
`claude` subprocess the instant it sees a `tool_use` block, so Claude Code
itself never records a matching `tool_result` for it, and the session it
persists to disk ends on an unresolved tool call. A real controlled
experiment (a real Hermes-scale payload -- 40 declared tools, a
32,601-character system prompt -- held byte-identical) found that resuming
from that state measurably increases the model's tendency to re-issue the
tool call it just made, or a close variant of it, before answering: 0/4
trials did this when the full conversation was resent fresh each turn
(what this shim did before session caching worked correctly, and what the
direct Anthropic API's inherent statelessness also produces), vs. 3/4
trials under `--resume` + delta-only continuation.

Full-history resend is not an available fix (real conversations run to
thousands of messages; resending that on every tool round is the same cost
profile that made the pre-fix behavior expensive). The fix that verified
clean across a real multi-round agentic loop (3 trials x 4 consecutive
resumed rounds, 40 tools, 32,601-char prompt, 0/12 redundant calls):
`resolve_session` widens a tool-result-continuation delta to also resend
the immediately-preceding assistant `tool_calls` message, so the model is
never depending on Claude Code's own persisted copy of it. This is exactly
one extra message, not accumulated history. Verified end-to-end against
the real running shim (not just direct-CLI approximations): 0/12 repeats
across 3 trials of 4 consecutive tool calls each, and negligible cache
cost (`cache_read_input_tokens` stays flat turn over turn; `cache_creation_input_tokens`
grows only by the ordinary few hundred tokens of genuinely new content per
round). The warm pool's live-process transport was separately confirmed
*worse* than cold `--resume` for this same shape, both with the bare
tool-result delta (3/4 redundant calls) and with the widened one (4/4,
even worse) -- tool-result-continuation resumes are always served cold,
never from the warm pool, regardless of fingerprint match; this is a
verified, permanent exclusion, not a conservative placeholder.

Two other mitigations were tried and ruled out empirically: forking a new
session id (`--fork-session`) for the tool-continuation step tested *worse*
than plain resume (4/4 redundant calls); appending a short system-prompt
note via `--append-system-prompt` composes fine with this shim's
`--system-prompt` override but decisively busts Anthropic's prompt cache
the first time it's used (`cache_creation_input_tokens` jumping from
low-hundreds to a near-total re-cache), so it was dropped outright.

### Session-cache tolerance for client-side history rewrites

Some clients mutate already-sent message content between turns in ways that
are invisible to the user but would otherwise look like history divergence
and force an expensive, continuity-losing fresh session. Verified against
Hermes Agent's real source and live traffic captures (not guessed):

- **Mid-turn "steer" messages**: Hermes appends an
  `[OUT-OF-BAND USER MESSAGE ...]...[/OUT-OF-BAND USER MESSAGE]` block to the
  content of the last `role: "tool"` message when a user sends a message
  while a turn is still running. The marker is permanent, immutable history
  on Hermes's side once injected (confirmed by Hermes's own test suite) --
  the shim strips it before comparing tool-result content, but only for
  `role: "tool"` messages, since the identical text also appears, by design,
  in Hermes's system-prompt boilerplate explaining the marker to the model
  (stripping it there would be an accidental match, not a meaningful one).
- **Context-compaction demotion**: Hermes's context compressor can later
  replace an aging tail tool-result with a one-line
  `"[<tool> output demoted at compaction -- N chars preserved in session
  history...]"` stub once it ages out of its kept window. Unlike the OOB
  marker, this is deliberately treated as a real divergence -> fresh session,
  not tolerated as a match: a resumed Claude session never gets history
  re-sent (only the new delta), so tolerating a full-content/stub pair as
  equal would mean Claude's own session keeps the full-size content forever
  and never inherits Hermes's compaction. A fresh session's baseline is
  exactly what Hermes now sends (stub included), so going fresh is what
  actually shrinks this shim's Claude session in step with Hermes's own --
  the DEBUG log flags this case distinctly from a genuine divergence so it
  doesn't read as a bug.
- **Client-asserted session identity**: if a system message contains a
  `Session ID: <id>` line (emitted by Hermes's `--pass-session-id`, CLI/TUI
  only as of this writing -- not wired into Hermes's webui gateway backend),
  the shim keys the conversation by that id directly instead of hashing the
  system+first message, and additionally stops treating system-prompt
  content differences (a live timestamp/model/provider line, typically) as
  divergence -- the client has already asserted "same session", so that's
  volatile metadata, not a real content change. Any client can opt into this
  by emitting the same line; it isn't Hermes-specific.
- **Narration text alongside a tool call**: a real live divergence (Hermes
  dogfooding this repo through the shim) showed a client not reliably
  replaying the assistant's narration text that preceded a tool call --
  `content: ""` came back where the original reply had real text, while the
  tool call's `id` and arguments matched exactly. Since `tool_calls[].id` is
  a Claude-generated, effectively unique identifier per call, a match there
  (plus matching arguments) is already decisive evidence it's the same turn
  regardless of what happened to any accompanying narration -- so content is
  now ignored entirely whenever a message carries `tool_calls`, the same way
  `reasoning`/`reasoning_details` already are.

### Real streaming: PTY trick

`claude`'s stdout is fully buffered (not line-buffered) on a plain pipe, causing
`--include-partial-messages` events to arrive in 2-3 giant bursts regardless of
actual generation time. Spawning `claude` with a real PTY (`pty.openpty()`) as
its stdout forces line buffering, producing real per-token deltas spread across
generation time.

Active only for the safe streaming path (single choice, no tools, no
`json_schema`). All other combinations use buffered-then-emit SSE
(protocol-correct, just not real-time).

### Isolation

Every `claude` subprocess runs with:

- A fresh empty temp directory as its `cwd` (prevents file exposure).
- `--setting-sources ""` -- excludes `~/.claude/settings.json` and its hooks. Auth reads from the separate `~/.claude/.credentials.json` and is unaffected.
- `--tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}'` on no-tools requests -- prevents locally-configured MCP servers from leaking into responses.
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` -- disables the autoupdater, telemetry, error reporting, and feedback prompt. Side effect: `claude` won't self-update while running under this shim; run `claude update` manually on your own schedule.

## License

This package is licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
and can understand more at http://choosealicense.com/licenses/apache/ on the
sidebar notes.

Apache License v2.0 is a MIT-like license. This means, in plain English:
- It's truly open source
- You can use it as you wish, for money or not
- You can sublicense it (change the license!!)
- This way, you can even use it on your closed-source project

As long as:
- You cannot use the authors names, logos, etc, to endorse a project
- You keep the authors copyright notices where this code got used, even on your closed-source project
(come on, even Microsoft kept BSD notices on Windows about its TCP/IP stack :P)
