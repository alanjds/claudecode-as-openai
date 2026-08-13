# claudecode-as-openai

OpenAI-chat-completions-compatible HTTP shim over the local `claude` CLI
(Claude Code), so Hermes (or anything else speaking the OpenAI chat API)
can drive a Claude subscription instead of metered API billing.

## Why this exists

Hermes's native `copilot-acp` provider can be pointed at `claude-agent-acp`
(see the `hermes-agent`/`claude-code-subscription-shim` skills), but that
path has no session caching (fresh subprocess every turn) and no real
model-selection parameter. This shim exists to get a `--model` flag per
request working reliably. (Session caching via `--session-id`/`--resume`
was the original goal too, but is NOT currently implemented -- see
"Capability audit" below.)

## Capability audit (2026-08-13)

Verified empirically against a live running shim, not just by reading the
code. Legend: ✅ works as expected, ⚠️ partial/misleading, ❌ not
implemented / silently ignored.

| Capability | Status | Notes |
|---|---|---|
| Model selection (`model` field) | ✅ | `--model <value>` passed straight to `claude -p` per request. Verified `sonnet` -> `claude-sonnet-4-6`, `opus` -> `claude-opus-4-6` both correctly selected, confirmed by asking Claude to self-report its version. |
| Session/conversation caching | ❌ | **Not implemented.** No `--session-id`/`--resume` anywhere in the code -- every request spawns a brand-new `claude -p` subprocess with zero session continuity at the CLI level. Multi-turn context "works" only because OpenAI clients (correctly) resend the full message history every call, which the shim replays into a fresh Claude Code process each time. Confirmed by measuring `cache_creation_input_tokens`: it grows linearly with conversation length (422 -> 477 -> 1719 across 3 turns) rather than staying flat, meaning the actual conversation content is NOT cached and is reprocessed from scratch on every call. The ~22-23K `cache_read_input_tokens` seen on every request is Anthropic's automatic caching of Claude Code's own static system harness/tool-definitions overhead, not this shim's doing, and not conversation-specific. |
| Streaming (`stream: true`) | ⚠️ | SSE protocol shape is correct (proper `data:`/`[DONE]` framing, real event-stream headers) but it is **not real token streaming**. `call_claude_with_tool_retry` blocks until the entire Claude Code response is ready, then emits it as a single `content` delta chunk. Verified: a 100-word response arrived as one chunk after a flat 12.3s wait, not incrementally. Fine for correctness, but a real-time "typing" UI gets nothing until the full answer is done. |
| Tool calling (`tools`/`tool_calls`) | ⚠️ | Works via real Anthropic `tool_use` blocks, but only ~40% reliably per call before the bounded retry (see above); with retry, effectively reliable in practice. |
| `tool_choice: "required"` / `"none"` / specific-function forcing | ❌ | Silently ignored. Verified: sending `tool_choice: "required"` with an unrelated prompt still returned plain text, no tool call -- the parameter has zero effect. |
| `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `logprobs`, `presence_penalty`/`frequency_penalty` | ❌ | None of these are read from the request payload at all (confirmed by inspecting `do_POST`: only `messages`, `tools`, `model`, `stream` are extracted). Silently accepted and silently ignored -- no error, no warning. |
| `n` (multiple choices) | ❌ | Always returns exactly 1 choice regardless of `n`. Verified `n: 3` still returns `len(choices) == 1`. |
| `response_format: {"type": "json_object"}` | ❌ | Not enforced. Verified: asking for JSON output with this flag set still returned the JSON wrapped in a ` ```json ` markdown fence, not raw parseable JSON. |
| Vision / image content blocks | ❌ | Silently dropped. `_flatten_content()` only extracts `type: "text"` blocks from multi-part content; `image_url` blocks are discarded with no error, no warning, no image ever reaching Claude. |
| Local MCP tool leakage | ⚠️ **notable gap** | `--disallowedTools` only blocks Claude Code's own named built-ins (Bash, Read, Write, etc.). It does **nothing** against MCP servers configured in the invoking user's local `~/.claude` settings -- those remain live and can be dispatched unpredictably. Verified live: asking to "list files using any tool you have" got Claude's own reply "The tools I have are MCP-based," and a separate request returned a real `tool_calls` response for `mcp__cachebro__read_files`, a locally-configured MCP tool never declared in the request's `tools` array. This means the shim's actual tool surface is NOT fully controlled by the API caller -- it depends on whatever MCP servers happen to be configured on the machine running the shim. |
| Error responses | ❌ | Always HTTP 200, even on failure. An invalid `model` value returns `200 OK` with Claude's plain-text complaint stuffed into `message.content` (not a proper OpenAI-style `4xx` with `{"error": {...}}`). A genuine internal exception returns HTTP 500 with an `{"error": {...}}` body, so that path exists, but expected-but-invalid input (bad model name) doesn't reach it. |
| Authentication | ❌ | None. Any `Authorization` header is accepted or ignored; anyone who can reach the port can use it. Fine for `127.0.0.1`-only binding (the default), a real gap if ever bound to `0.0.0.0`. |
| `/v1/models` | ⚠️ | Returns a single hardcoded entry (`DEFAULT_MODEL = "sonnet"`), not the actual list of models the underlying `claude` CLI/subscription supports. |

**Bottom line:** model selection is the one advertised capability that
fully works as designed. Streaming is protocol-correct but not real
streaming. Session caching -- the other half of this project's original
motivation -- was never actually built. Sampling/formatting parameters
and multi-choice are uniformly no-ops. The MCP tool leakage is worth
fixing before relying on this for anything where the exact tool surface
matters (security-sensitive contexts especially).

## Current approach: native `tool_use` + bounded retry

`claudecode_as_openai/shim.py` calls `claude -p --output-format
stream-json` and parses the real Anthropic `tool_use` content blocks from
the NDJSON stream (not a hand-rolled text convention). Custom tools are
described in the system prompt; Claude's own built-in tools (Bash, Read,
Edit, etc.) are blocked via `--disallowedTools <enumerated list>` so they
don't compete for dispatch.

**Known limitation:** this gets real `tool_use` blocks, but only ~40%
reliably per single call (Claude sometimes narrates "let me check..."
instead of dispatching — matches the failure mode in
[cline/cline#10336](https://github.com/cline/cline/issues/10336), a
community workaround for an unrelated bug in Cline's *own* agentic XML
tool vocabulary colliding with Claude Code's native tools; it does not
reflect Cline's actual Claude Code provider). Mitigation is a bounded
retry with exponential backoff (`call_claude_with_tool_retry`, up to 4
retries / 5 total attempts, 2s base / 15s cap) when tools were requested
but no `tool_use` came back. Verified 15/15 in manual testing with retries
vs 6/15 without -- but this is *probabilistic*, not deterministic:
expected success asymptotically approaches but never reaches 100%.

## Investigated and ruled out: a deterministic fix (2026-08-12)

Went looking for an architecturally deterministic alternative (no retry
needed) across three reference implementations. None of them has one --
this genuinely appears to be an inherent limitation of the Claude Code
CLI/agent harness, not something this shim is missing:

1. **Old Cline (`cline/cline` commits `9dea336c`/`8a6441fd`,
   `src/core/api/providers/claude-code.ts`)**: real, but this file is
   **deleted in current `main`** -- Cline has since migrated off its own
   spawn-and-parse code entirely. Its historical fix was
   `@withRetry({maxRetries: 4, baseDelay: 2000, maxDelay: 15000})` around
   the whole call -- a retry decorator, not a deterministic mechanism.
   This shim's current retry parameters were tuned to match those
   numbers, on the theory that Cline's own engineers had already tuned
   them empirically.
2. **Current Cline**: delegates entirely to the third-party
   `ai-sdk-provider-claude-code` npm package
   (`sdk/packages/llms/src/providers/vendors/community.ts`, dynamic
   import). Cline itself has no bespoke retry/parsing logic for this path
   anymore.
3. **`ai-sdk-provider-claude-code`** (the delegate package, unofficial/
   community-maintained, wraps `@anthropic-ai/claude-agent-sdk`): no
   built-in retry-with-backoff for this failure mode. Its own
   troubleshooting docs show a generic example retry loop for users to
   write themselves.
4. **Official `@anthropic-ai/claude-agent-sdk`**: has a first-class retry
   message (`SDKAPIRetryMessage`), but only for transport/API-level
   errors (rate limits, connection failures) -- not for "the model chose
   to narrate instead of dispatching a tool." No `tool_choice: "required"`
   or equivalent forcing parameter is exposed through the CLI/SDK harness
   (that's a raw Anthropic Messages API parameter the agent harness
   doesn't surface to callers).

**Conclusion:** every implementation that touches this problem treats it
as inherently probabilistic and handles it with a retry loop, because the
Claude Code harness doesn't expose the underlying `tool_choice` forcing
knob. The retry wrapper in this shim is the standard approach here, not a
stopgap. See the (intentionally kept, unmerged) `explore/text-tool-parser`
branch for the investigation trail and the ruled-out text-parser
alternative.

## Tests

Two tiers:

```
# Fast, offline, CI-safe: mocks the `claude` CLI subprocess with canned
# NDJSON fixtures matching real observed output shapes. No live claude
# binary or subscription needed.
python3 -m unittest tests.test_shim_unit -v

# Live integration tests: need a real `claude` CLI on PATH, authenticated
# to an actual Claude subscription, and a running shim instance. NOT run
# in CI. Run these locally before tagging a release.
python3 -m claudecode_as_openai.shim 8977 &
python3 tests/test_plain_chat.py
python3 tests/test_tool_call_reliability.py --attempts 15
```

## Installation

```
pip install claudecode-as-openai   # once published
# or, from source:
pip install .
```

## Running

```
python3 -m claudecode_as_openai [port]        # shorthand, defaults to 8977
python3 -m claudecode_as_openai.shim [port]   # equivalent, explicit form
claudecode-as-openai [port]                    # equivalent, via entry point
```
