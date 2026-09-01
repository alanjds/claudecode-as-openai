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
* OpenRouter (https://openrouter.ai) reasoning-tokens compatibility:
  `resolve_reasoning_effort()` maps OpenRouter's `reasoning` request
  parameter (`{"effort": "high"}`, `{"max_tokens": N}`, or
  `{"enabled": true}`) onto Claude Code's own `--effort
  <low|medium|high|max>` flag, clamping OpenRouter's wider effort
  vocabulary (`none`/`minimal`/`xhigh`) to the nearest accepted value
  since `--effort` only accepts exactly those four levels (verified
  live: anything else is rejected outright). Verified live via a
  stream-json capture that `--effort` genuinely produces real
  `thinking` content blocks (with a cryptographic `signature` field)
  ahead of the final `text` block, not a cosmetic no-op -- these are
  now captured (previously silently discarded) and surfaced back as
  both of OpenRouter's supported response shapes:
  `message.reasoning` (plaintext) and `message.reasoning_details`
  (structured array, `type: "reasoning.text"`,
  `format: "anthropic-claude-v1"`, signature preserved). Verified live
  end-to-end through the actual running shim: a real request with
  `reasoning: {"effort": "high"}` returned genuine thinking content in
  both fields with a real signature attached; a plain request with no
  `reasoning` key carried neither field. Live token-level SSE streaming
  is skipped whenever `reasoning` is set (falls back to the existing
  buffered path) since the streaming callback only hooks `text_delta`
  events and correctly ordering `thinking` before `text` needs a second
  callback path not currently wired up.
* OpenAI-shaped nested usage details: `_build_openai_usage()` adds
  `usage.prompt_tokens_details.cached_tokens` (OpenAI's actual
  documented shape) alongside the existing flat
  `cache_read_input_tokens`/`cache_creation_input_tokens` custom keys
  (kept for backward compatibility), so clients/dashboards that
  specifically parse OpenAI's standard nested usage structure get real
  numbers instead of ignoring a field they don't recognize.
  `completion_tokens_details.reasoning_tokens` is intentionally NOT
  added -- Claude Code's usage payload has no separate reasoning-token
  count (verified live: `output_tokens` already includes any
  thinking-block tokens, undifferentiated), so there's no honest number
  to report there.
* Added 14 new unit tests covering `resolve_reasoning_effort`,
  `build_reasoning_details`, and `_build_openai_usage`. Full suite now
  87 tests (was 73), all passing.
* Cleanup pass: extracted 4 shared helpers to remove duplicated logic
  (`_scoped_env_overrides`, `_build_sse_chunk`, `_gen_tool_id`,
  `_parse_ndjson_line`) and trimmed narrated-investigation comment/
  docstring bloat throughout `shim.py` (1877 -> 1701 lines) down to the
  load-bearing "what + why" behind each non-obvious choice. No behavior
  change; all 87 tests pass and the running shim was live-verified after
  each step.
* Startup-latency: every spawned `claude` subprocess now gets
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` (see `_BASE_ENV_OVERRIDES`
  in shim.py), bundling `DISABLE_AUTOUPDATER`/`DISABLE_TELEMETRY`/
  `DISABLE_ERROR_REPORTING`/`DISABLE_FEEDBACK_COMMAND`. Since this shim
  spawns a fresh `claude -p` process per request, skipping the
  autoupdater's version-check network call and telemetry/error-reporting
  init on every single call is a real per-request latency win. Trade-off:
  `claude` will never self-update while running under this shim -- the
  operator is expected to run `claude update` manually on their own
  schedule. See README.md "Startup-latency" section.
* Observability: new `claudecode_as_openai/tracking.py` module with a
  pluggable `UsageTracker` seam (`register_tracker`/`emit_usage`), a
  `JsonlFileTracker` sink (`CLAUDE_OPENAI_USAGE_LOG`), and an optional
  `LiteLLMCostTracker` attaching `cost_usd` via LiteLLM's pricing table
  (`CLAUDE_OPENAI_TRACK_COST=1`). All best-effort -- a tracker exception
  is swallowed and never affects the response, matching this shim's
  existing warm-pool-parking failure contract. New `tracking` extra
  (`pip install "claudecode-as-openai[tracking]"`) for `litellm`/`logfire`;
  the shim runs identically with neither installed. See README.md
  "Observability" section.
* Fixed usage under-reporting across the tool-call retry loop:
  `call_claude_with_tool_retry` previously kept only the winning
  attempt's `usage`, silently discarding every earlier attempt's real,
  billed tokens. Usage is now summed across every attempt before being
  returned, so both the response's `usage` field and tracker events
  reflect true billed usage.
* DEBUG-level logging (`CLAUDE_OPENAI_LOG_LEVEL=DEBUG`) of the exact
  `claude` command line at every subprocess spawn (cold and warm-pool
  init), plus which path (warm vs. cold) served each turn. The value
  following `--system-prompt`/`--json-schema` is always redacted to
  `<len=N>` -- prompt/schema content is never written to logs;
  user message content was already never in argv (sent via stdin).
* Optional OTEL/Logfire span tracing (`CLAUDE_OPENAI_TRACING=1`,
  `LOGFIRE_TOKEN`): nested spans per request, per `n`-fan-out choice, per
  tool-retry attempt, and per `claude` subprocess spawn (carrying the
  redacted command and `gen_ai.usage.*` attributes) -- surfaces the
  retry-loop/fan-out/warm-vs-cold amplification that's invisible from a
  single request otherwise. Degrades to a true no-op with `logfire`
  uninstalled or tracing disabled.
* `resolve_session` now logs, at DEBUG, exactly why it picked `fresh` vs.
  `resume` -- no prior entry, an existing entry whose incoming message
  count isn't a strict superset of what's synced, a diverged prefix (with
  the exact message index that differs), or a match. Verified live: this
  is what let a real "every session looks cold" report get diagnosed from
  a single log capture -- the session-cache matching logic itself turned
  out to be correct the whole time.
* `TOOL_CALL_MAX_RETRIES` now defaults to `0` (override via
  `CLAUDE_OPENAI_TOOL_CALL_MAX_RETRIES`) instead of `4`. Root cause of the
  "every session looks cold" investigation above: native MCP tool
  registration already reaches ~100% turn-1 dispatch when a tool call is
  actually warranted (see "Tool-call reliability fix" above), so a turn
  that declares tools but doesn't call one is now overwhelmingly a
  correct "no tool needed" response, not a dispatch failure -- retrying
  it was discarding session/warm-pool continuity for no benefit on every
  such turn, which is common for clients (agentic web UIs especially)
  that declare tools on every message regardless of whether one is
  relevant. When retries ARE re-enabled and the original call was a
  resumed conversation, a retry now forks a new session from that same
  original checkpoint via `--fork-session` instead of starting over with
  the full history -- verified live against the real CLI: a fork is
  billed as a cache hit of the source's entire context (not a resend)
  and leaves the source session's own transcript untouched, so every
  retry stays an equally clean re-ask at a fraction of the cost. (A
  first-ever-turn retry, with no pre-existing session to fork from,
  keeps the old full-history-resend fallback -- there's nothing cheaper
  available for that case.) Verified live end-to-end with retries
  re-enabled (warm attempt fails -> first retry plain `--resume` -> all
  further retries `--resume ... --fork-session` off the same original
  checkpoint, never off each other) and with the new default (exactly
  one subprocess spawn, no retry, ~9x faster than the exhausted-retries
  case for the same non-tool-needing turn).
* Refined the Hermes out-of-band steer-message tolerance added for the
  session-cache fix above: `_strip_oob` now only applies to `role: "tool"`
  message content, not every string-content message -- it was previously
  also matching the identical marker text that appears, by design, in
  Hermes's own system-prompt boilerplate explaining the marker to the
  model (harmless today since that text is static per-conversation, but
  an accidental match, not a designed one). Verified against Hermes's
  real source (`agent/prompt_builder.py`, `agent_runtime_helpers.py`)
  and a live traffic capture that the marker format and injection point
  are exactly as assumed, and that the marker is permanent, immutable
  conversation history on Hermes's side (never disappears on its own).
* Identified, but deliberately did NOT tolerate, a second, distinct
  Hermes-side content-shrink source found during that same investigation:
  Hermes's context compressor can later replace an aging tail tool-result
  with a one-line `"[<tool> output demoted at compaction ...]"` stub once
  it ages out of its kept window. A resumed Claude session never gets
  history re-sent (only the new delta), so treating a full-content/stub
  pair as a match would mean Claude's own session keeps the full-size
  content forever, growing unbounded regardless of how much Hermes
  compacts on its own side -- Hermes's compaction signal would never
  reach Claude's context at all. `resolve_session` correctly falls
  through to its existing `fresh` behavior here: a fresh session's
  baseline IS `full_openai_messages` exactly as Hermes now sends it (stub
  included), so it's what actually lets this shim's Claude session
  inherit Hermes's compaction and stay bounded over long conversations.
  The DEBUG divergence log now flags this specific case
  ("compaction demotion, expected fallout, not a bug") so it doesn't
  read as a regression when it fires.
* Added client-asserted session identity support: when a system message
  contains a `Session ID: <id>` line (emitted by Hermes's
  `--pass-session-id`, currently CLI/TUI only -- confirmed not wired into
  Hermes's webui gateway backend), `resolve_session` keys the
  conversation by that id directly instead of hashing the system +
  first message, and stops treating system-prompt content differences
  as divergence for that conversation (the client already asserted "same
  session", so a live timestamp/model/provider line differing is
  volatile metadata, not a real change). Not Hermes-specific -- any
  client can opt in by emitting the same line. DEBUG logs now also state
  which of the two keying strategies was used for every `resolve_session`
  call.
* Fixed a new real live divergence (2026-09-01, Hermes dogfooding this
  repo through the shim): the assistant's narration text preceding a tool
  call wasn't reliably replayed back by the client (`content: ""` came
  back where the original reply had real text), even though the tool
  call's `id` and arguments matched exactly. `tool_calls[].id` is a
  Claude-generated, effectively unique identifier per call -- a match
  there (plus matching arguments) is already decisive evidence of the
  same turn, so `_normalize_message` now ignores `content` entirely
  whenever `tool_calls` is present, the same way `reasoning`/
  `reasoning_details` already are. 129 tests pass (2 new since the
  previous entry).
