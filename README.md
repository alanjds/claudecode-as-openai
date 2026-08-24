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

```
pip install claudecode-as-openai
# or from source:
pip install .
```

## Quickstart

```
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

```
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
| Tool calling (`tools`/`tool_calls`) | ✅ | Registered as real MCP tool schemas -- 100% turn-1 dispatch reliability. Falls back to a prose-description path with bounded retry only for tool names that violate MCP naming rules. |
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
| Authentication | ❌ | None. Fine as `127.0.0.1` binding is hardcoded. |

## Tests

```
# Offline unit tests (no live claude needed):
python3 -m unittest tests.test_shim_unit -v

# Live integration tests (need a running shim and authenticated claude):
claudecode-as-openai &
python3 tests/test_plain_chat.py
python3 tests/test_tool_call_reliability.py --attempts 15
```

---

## Internals lore

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
system-prompt change.

Falls back to a prose-description path with bounded retry (4 retries, 2s base /
15s cap) for any tool name that can't satisfy MCP's `^[a-zA-Z0-9_-]{1,64}$`
constraint.

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
