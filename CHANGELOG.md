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
* Real token-level streaming for `stream: true`: fixed by spawning
  `claude` with a real PTY (`pty.openpty()`) instead of a plain pipe --
  Claude Code's own stdout is fully buffered (not line-buffered) without
  a terminal attached, which previously meant `--include-partial-messages`
  events arrived in 2-3 giant bursts regardless of actual generation time.
  Verified live: a 300-word story now streams as 21 separate chunks over
  ~19s instead of one chunk at the end. Only active for the safe case
  (single choice, no tools requested, no `response_format: json_schema`);
  every other combination keeps the previous buffered-then-emit behavior.
* One-time-per-parameter-name stderr warning for sampling parameters that
  have no Claude Code equivalent (`temperature`, `top_p`, `seed`,
  `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`,
  `logit_bias`) -- accepted, not enforced, never hard-errored (since real
  clients routinely send explicit defaults that carry no signal of intent).
* `/v1/models` now queries the real, current Anthropic model list from
  `https://api.anthropic.com/v1/models` (a free metadata call) instead of
  returning a hardcoded 3-entry guess. Prefers Claude Code's own OAuth
  access token (`~/.claude/.credentials.json`) over `ANTHROPIC_API_KEY`
  over a hardcoded fallback list, in that order, so the real list is
  available without requiring a separate API key. Verified live: returns
  the real current 10-model list via OAuth alone; falls back correctly to
  the hardcoded list when no credentials are available at all. Results
  cached in-process for 5 minutes.
* Fixed local hook/settings leakage: every spawned `claude` call now
  passes `--setting-sources ""`, excluding the invoking user's real
  `~/.claude/settings.json` (user/project/local scopes) entirely.
  Verified live: a real `SessionStart` hook previously fired and injected
  a marker string into model context on a plain call; with this flag, no
  hook message reaches the model at all. `--bare` mode was considered as
  a stronger lockdown but ruled out -- it strictly requires
  `ANTHROPIC_API_KEY` and never reads OAuth/keychain (verified live:
  fails with "Not logged in" under OAuth-only auth), which would break
  this shim's entire subscription-based premise.
* **Tool-call reliability fix: native MCP tool registration.** Custom
  tools declared in a request's `tools` array are now registered as a
  real MCP tool server (`claudecode_as_openai/mcp_tool_server.py`, a
  stdio-only child process, never a TCP listener) via `--mcp-config`,
  with each tool marked `_meta: {"anthropic/alwaysLoad": true}` to skip
  Claude Code's ToolSearch deferred-loading and get the full JSON
  schema into the initial prompt turn. This replaces the old
  prose-description-in-the-system-prompt approach for any tool whose
  name satisfies MCP's `^[a-zA-Z0-9_-]{1,64}$` naming pattern (the
  prose path remains as an automatic fallback for names that don't).
  Verified live through the actual running shim: 5/5 repeatable turn-1
  tool dispatches with correct OpenAI-shaped `tool_calls` output
  (previously ~40% single-shot reliability with the prose approach).
  Session/prompt caching verified to stay intact for a stable tool set
  across `--resume`'d turns (`cache_creation_input_tokens` flat at
  ~110-140 tokens after the first turn, vs ~15,800 on the first);
  changing the tool set mid-session correctly busts the cache, same as
  any other system-context change would. Also investigated and
  explicitly rejected: forging direct
  `https://api.anthropic.com/v1/messages` calls with the CLI's own
  OAuth token (100% reliable but bypasses `-p` mode entirely, ruled
  out as out-of-bounds for this project). See README.md's "Native MCP
  tool registration" section for the full writeup.
* OpenRouter (https://openrouter.ai) model-name compatibility:
  `normalize_model_name()` translates OpenRouter's Anthropic model slug
  convention (e.g. `anthropic/claude-sonnet-4.5`,
  `~anthropic/claude-sonnet-latest`) into Claude Code's own `--model`
  convention before every invocation, so clients already configured for
  OpenRouter-style model names work unchanged against this shim. Handles
  the `~`/`anthropic/` prefix stripping, `-latest` bare-alias mapping
  (`claude-sonnet-latest` -> `sonnet`), dot-to-dash version conversion
  (`4.5` -> `4-5`, since Claude Code's `--model` flag rejects the dotted
  form outright), and OpenRouter's `-fast` Fast-mode suffix (stripped
  with a one-time stderr warning, since `-p` mode has no reachable
  fast-mode equivalent). Already-native Claude Code model strings and
  anything unrecognized pass through unchanged. Verified live through
  the actual running shim: 7/7 OpenRouter-style model names resolved
  correctly via `curl` against `/v1/chat/completions`, each confirmed by
  asking the model to self-report its exact version string. Added 6 new
  unit tests. Full suite now 73 tests (was 67), all passing.
