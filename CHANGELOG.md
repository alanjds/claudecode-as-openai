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
