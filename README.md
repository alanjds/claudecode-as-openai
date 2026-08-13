# claudecode-as-openai

OpenAI-chat-completions-compatible HTTP shim over the local `claude` CLI
(Claude Code), so Hermes (or anything else speaking the OpenAI chat API)
can drive a Claude subscription instead of metered API billing.

## Why this exists

Hermes's native `copilot-acp` provider can be pointed at `claude-agent-acp`
(see the `hermes-agent`/`claude-code-subscription-shim` skills), but that
path has no session caching (fresh subprocess every turn) and no real
model-selection parameter. This shim exists to get `--model`,
`--session-id`/`--resume`, and several other OpenAI-shaped API surfaces
working reliably against a local Claude Code CLI/subscription instead of
metered API billing.

## Capability audit (2026-08-13)

Verified empirically against a live running shim, not just by reading the
code. Legend: ✅ works as expected, ⚠️ partial/misleading, ❌ not
implemented / silently ignored.

| Capability | Status | Notes |
|---|---|---|
| Model selection (`model` field) | ✅ | `--model <value>` passed straight to `claude -p` per request. Verified `sonnet` -> `claude-sonnet-4-6`, `opus` -> `claude-opus-4-6` both correctly selected, confirmed by asking Claude to self-report its version. |
| Session/conversation caching | ✅ | Implemented via conversation fingerprinting (`resolve_session`/`record_session`): when a client echoes back the exact prior assistant reply (as real OpenAI clients do), the shim resumes the matching Claude Code session with `--resume` and sends only the new delta message, instead of replaying full history into a fresh subprocess. Falls back to a fresh session on any divergence (edited/regenerated history). Verified live: `cache_creation_input_tokens` dropped from ~3500 (fresh turn) to ~30-40 tokens (resumed turn) across a real 3-turn conversation, with correct recall across turns. |
| Streaming (`stream: true`) | ⚠️ | SSE protocol shape is correct (proper `data:`/`[DONE]` framing, real event-stream headers) but it is **not real token streaming** for a normal request. `call_claude_with_tool_retry` blocks until the entire Claude Code response is ready, then emits it as a single `content` delta chunk. Verified: a 100-word response arrived as one chunk after a flat 12.3s wait, not incrementally. (Claude Code *does* expose real token-level deltas via `--include-partial-messages` -- this shim currently only turns that on internally for early `stop`-sequence detection, not for the client-facing SSE stream; wiring real token streaming through to the client is a natural next step, not attempted yet.) |
| Tool calling (`tools`/`tool_calls`) | ⚠️ | Works via real Anthropic `tool_use` blocks, but only ~40% reliably per call before the bounded retry (see below); with retry, effectively reliable in practice (10/10 in the latest live regression run). |
| `tool_choice: "none"` | ✅ | Routed through the same full lockdown path as "no tools requested" (built-ins + MCP servers both blocked, see below) -- no tool description is even added to the system prompt. |
| `tool_choice: "required"` / forcing a specific function | ❌ | Accepted but not enforced -- no equivalent mechanism exists in the Claude Code harness (no raw Anthropic `tool_choice` parameter is exposed through the CLI). |
| `max_tokens` / `max_completion_tokens` | ✅ | Mapped to the `CLAUDE_CODE_MAX_OUTPUT_TOKENS` env var (real, documented Claude Code setting), scoped per-subprocess-call. Verified: a low cap correctly triggers `finish_reason: "length"`. |
| `stop` (stop sequences) | ✅ | Claude Code has no native stop-sequence flag, so this is implemented as **real early termination**, not just post-hoc truncation: with `--include-partial-messages` (only enabled when `stop` is actually set), the shim reads real token-level `content_block_delta` events and kills the `claude` subprocess (`proc.terminate()`) the instant a stop sequence appears in the accumulated text -- before the rest of the response is generated or billed. Verified live: a 500-word-story prompt took 22.3s with no stop sequence vs 4.9s with one that matched early in the output (~4.5x faster, plus the unbilled remainder of the response). A message-level check remains as a safety net for edge cases (e.g. the overall timeout firing first). |
| `n` (multiple choices) | ⚠️ | Implemented up to a bounded limit (`MAX_N_CHOICES`, currently 5) -- each choice is a fully independent fresh Claude Code subprocess call (no session caching across choices, by design: parallel-choice semantics don't fit single-session continuity). Requests above the limit get a proper `400 invalid_request_error`, not silent truncation. |
| `response_format: {"type": "json_object"}` / `{"type": "json_schema", ...}` | ✅ | Mapped to `--json-schema` (a real Claude Code flag). `json_object` maps to an open `{"type": "object"}` schema; `json_schema` passes the caller's schema through directly. Requires `--max-turns 3` internally (Claude Code's own corrective retry mechanism via an internal `StructuredOutput` tool) -- verified empirically that `max_turns 1` leaves it hanging at `error_max_turns` with the schema never actually produced. Verified live: returns genuinely valid, parseable JSON, not markdown-fenced. |
| `temperature`, `top_p`, `seed`, `logprobs`, `presence_penalty`/`frequency_penalty` | ❌ | Genuine CLI limitation, not a shim gap -- no equivalent flags exist anywhere in Claude Code (checked `claude --help` and the official env-vars docs). Silently accepted, silently ignored. The closest available knob is `--effort <low|medium|high|max>` (reasoning effort, not sampling temperature -- a different axis, not currently wired up). |
| Vision / image content blocks | ❌ | Silently dropped. `_flatten_content()` only extracts `type: "text"` blocks from multi-part content; `image_url` blocks are discarded with no error, no warning, no image ever reaching Claude. |
| Local MCP tool leakage | ✅ **fixed** | `--disallowedTools` alone only blocks Claude Code's own named built-ins (Bash, Read, Write, etc.) -- it does nothing against MCP servers configured in the invoking user's local `~/.claude` settings, which could otherwise be dispatched unpredictably even on a request that declared no tools. Fixed for the no-tools-requested path via `--tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}'` together (verified: `--tools ""` alone still leaked real MCP tool calls; the combination gets `tools available: []`, fully closed). Note: when tools ARE requested, native `tool_use` dispatch is still used (see "Current approach" below) and this full lockdown does not apply there. |
| CWD / local file exposure | ✅ **fixed** | Every spawned `claude` subprocess previously inherited the shim's own working directory, and Claude Code is aware of and can reference real files there (confirmed live: a "list files using any tool you have" request surfaced real project file references). Fixed by spawning every `claude` call with `cwd=` pointed at a dedicated, empty, per-run sandbox temp directory instead. |
| Error responses | ✅ | Claude Code's real structured error signal (`chunk["error"]`, e.g. `"invalid_request"`, or `result` chunks with `is_error: true`) is now classified and translated into a proper OpenAI-shaped `{"error": {"message", "type", "code"}}` body with a matching HTTP status (400/404/429/500/503 as appropriate) via `ClaudeCliError`/`_classify_error_text`. Verified live: an invalid model now returns HTTP 404 with `code: "model_not_found"`, not a silent `200 OK` with the complaint text stuffed into `content`. |
| Authentication | ❌ | None. Any `Authorization` header is accepted or ignored; anyone who can reach the port can use it. Fine for `127.0.0.1`-only binding (the default), a real gap if ever bound to `0.0.0.0`. |
| `/v1/models` | ⚠️ | Returns a single hardcoded entry (`DEFAULT_MODEL = "sonnet"`), not the actual list of models the underlying `claude` CLI/subscription supports. |

**Bottom line:** model selection, session caching, error translation,
`max_tokens`, `response_format`, `tool_choice: "none"`, bounded `n`, the
MCP tool leakage, and the CWD file-exposure issue are all now implemented
and live-verified. Real token-level streaming to the CLIENT (as opposed to
the internal use of `--include-partial-messages` for early `stop`
detection) and raw sampling parameters (`temperature`/`top_p`/etc, which
Claude Code doesn't expose at all) remain the two honest gaps.

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
