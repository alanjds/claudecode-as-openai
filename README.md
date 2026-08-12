# openai-claudecli-bridge

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

`openai_claudecli_bridge/shim.py` calls `claude -p --output-format
stream-json` and parses the real Anthropic `tool_use` content blocks from
the NDJSON stream (not a hand-rolled text convention). Custom tools are
described in the system prompt; Claude's own built-in tools (Bash, Read,
Edit, etc.) are blocked via `--disallowedTools <enumerated list>` so they
don't compete for dispatch.

**Known limitation:** this gets real `tool_use` blocks, but only ~40%
reliably per single call (Claude sometimes narrates "let me check..."
instead of dispatching — matches the failure mode in
[cline/cline#10336](https://github.com/cline/cline/issues/10336)). Current
mitigation is a bounded retry (`call_claude_with_tool_retry`, up to 3
attempts) when tools were requested but no `tool_use` came back. Verified
15/15 in manual testing with retries vs 6/15 without — but this is
*probabilistic*, not deterministic: expected worst-case success is roughly
`1 - (1 - p)^3` for per-call success rate `p`, not 100%.

## Exploring: Cline's actual approach (deterministic)

Cline's real production fix for a related bug uses `--tools ""
--strict-mcp-config --mcp-config '{"mcpServers":{}}'` to remove ALL native
tool dispatch. With zero native tools, Claude reliably falls back to a
plain-text tool-call convention (`<function_calls>/<invoke>` XML or a raw
`<tool_call>{"name":...}` JSON blob) instead of `tool_use` blocks — Cline
ships a text/XML parser for exactly this. This is architecturally
deterministic (no coin-flip on whether Claude dispatches), at the cost of
writing and maintaining that parser. See the `explore/text-tool-parser`
branch for this work.

## Tests

```
python3 -m openai_claudecli_bridge.shim 8977 &
python3 tests/test_plain_chat.py
python3 tests/test_tool_call_reliability.py --attempts 15
```
