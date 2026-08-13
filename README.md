# claudecode-as-openai

OpenAI-chat-completions-compatible HTTP shim over the local `claude` CLI
(Claude Code), so Hermes (or anything else speaking the OpenAI chat API)
can drive a Claude subscription instead of metered API billing.

## Why this exists

Hermes's native `copilot-acp` provider can be pointed at `claude-agent-acp`
(see the `hermes-agent`/`claude-code-subscription-shim` skills), but that
path has no session caching (fresh subprocess every turn) and no real
model-selection parameter. This shim exists to get both: `claude -p
--session-id/--resume` caching and a `--model` flag per request.

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
