# Changelog

## 0.1.0
_Unreleased_

* Initial release: OpenAI-chat-completions-compatible HTTP shim over the
  local `claude` CLI (Claude Code), so any OpenAI-client-compatible tool
  (e.g. Hermes) can drive a Claude subscription instead of metered API
  billing.
* Native Anthropic `tool_use` content-block parsing (not a hand-rolled
  JSON-envelope or text convention) via `--output-format stream-json` and
  `--disallowedTools <built-in list>`.
* Bounded retry with exponential backoff (4 retries, 2s base / 15s cap)
  for the documented tool-dispatch flakiness -- see README.md for the
  full investigation trail and why no deterministic fix exists in this
  ecosystem today.
* Renamed from the original working name `openai-claudecli-bridge`.
* Session/conversation caching: `resolve_session`/`record_session`
  fingerprint the conversation and resume the matching Claude Code
  session via `--resume` (sending only the new delta message) instead of
  replaying full history into a fresh subprocess every turn. Falls back
  to a fresh session on any history divergence.
* OpenAI-style error translation: Claude Code's real structured error
  signal is classified (`ClaudeCliError`/`_classify_error_text`) into
  proper `{"error": {"message", "type", "code"}}` bodies with matching
  HTTP status codes (400/404/429/500/503), instead of always returning
  HTTP 200.
* `max_tokens`/`max_completion_tokens` via the `CLAUDE_CODE_MAX_OUTPUT_TOKENS`
  env var, scoped per subprocess call.
* `response_format: json_object` / `json_schema` via `--json-schema`
  (with the required `--max-turns 3` for Claude Code's internal
  corrective retry mechanism).
* `tool_choice: "none"` routed through the full lockdown path.
* Bounded `n` (multiple choices) support, each choice a fully independent
  subprocess call; requests above the limit get a proper 400 instead of
  silent truncation.
* Fixed MCP tool leakage: `--tools "" --strict-mcp-config --mcp-config
  '{"mcpServers":{}}'` together fully close the tool surface when no
  tools are requested (previously, locally-configured MCP servers could
  still be dispatched unpredictably).
* Fixed CWD/local-file exposure: every spawned `claude` subprocess now
  runs with `cwd` pointed at a dedicated empty sandbox temp directory,
  instead of inheriting the shim's own working directory (which Claude
  Code could reference/read from).
* `stop` sequences implemented as real early termination, not just
  post-hoc truncation: with `--include-partial-messages` enabled only
  when `stop` is set, the shim reads token-level `content_block_delta`
  events and kills the subprocess the instant a stop sequence appears,
  cutting both latency and cost on early stops (verified ~4.5x faster on
  a long-generation test case).
