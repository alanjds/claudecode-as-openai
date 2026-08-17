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
| Model selection (`model` field) | ✅ | `--model <value>` passed straight to `claude -p` per request, after translating OpenRouter-style Anthropic model slugs (`anthropic/claude-sonnet-4.5`, `~anthropic/claude-sonnet-latest`, etc -- see "OpenRouter model-name compatibility" below) into Claude Code's own naming convention. Verified `sonnet` -> `claude-sonnet-4-6`, `opus` -> `claude-opus-4-6` both correctly selected, confirmed by asking Claude to self-report its version. |
| Session/conversation caching | ✅ | Implemented via conversation fingerprinting (`resolve_session`/`record_session`): when a client echoes back the exact prior assistant reply (as real OpenAI clients do), the shim resumes the matching Claude Code session with `--resume` and sends only the new delta message, instead of replaying full history into a fresh subprocess. Falls back to a fresh session on any divergence (edited/regenerated history). Verified live: `cache_creation_input_tokens` dropped from ~3500 (fresh turn) to ~30-40 tokens (resumed turn) across a real 3-turn conversation, with correct recall across turns. |
| Streaming (`stream: true`) | ✅ | Real token-level streaming for the common case (single choice, no tools requested, no `response_format: json_schema`). Claude Code's own stdout is **fully buffered** (not line-buffered) when piped without a TTY -- verified: with `--include-partial-messages`, real per-token deltas arrived in 2-3 giant bursts within ~20ms regardless of actual generation time. Fixed by spawning `claude` with a real PTY (`pty.openpty()`) as its stdout instead of a plain pipe, which forces line buffering like an interactive terminal. Verified live: a 300-word story streamed as 21 separate content chunks over ~19s (avg gap 0.56s between chunks), matching real generation pacing, not one chunk at the end. Falls back to the previous buffered-then-emit behavior (SSE framing still correct, just not real-time) whenever tools are requested, `n > 1`, or `response_format: json_schema` is set -- see "Real streaming" section below for why those combinations aren't safe to stream live. |
| Tool calling (`tools`/`tool_calls`) | ✅ | Registered as real MCP tool schemas via `--mcp-config` (see "Native MCP tool registration" below) -- verified live to reach **100% turn-1 dispatch reliability** (5/5 through the actual shim, repeatable), vs ~40% for the earlier prose-description approach. Falls back to the prose path (bounded retry, ~40% single-shot) only for tool names that can't satisfy MCP's naming constraint. |
| `tool_choice: "none"` | ✅ | Routed through the same full lockdown path as "no tools requested" (built-ins + MCP servers both blocked, see below) -- no tool description is even added to the system prompt. |
| `tool_choice: "required"` / forcing a specific function | ❌ | Accepted but not enforced -- no equivalent mechanism exists in the Claude Code harness (no raw Anthropic `tool_choice` parameter is exposed through the CLI, and forcing it via a raw `tool_choice: {"type":"any"}` call was investigated and found not to compose safely with a possibly-empty tool set -- see "Native MCP tool registration" below). |
| `max_tokens` / `max_completion_tokens` | ✅ | Mapped to the `CLAUDE_CODE_MAX_OUTPUT_TOKENS` env var (real, documented Claude Code setting), scoped per-subprocess-call. Verified: a low cap correctly triggers `finish_reason: "length"`. |
| `stop` (stop sequences) | ✅ | Claude Code has no native stop-sequence flag, so this is implemented as **real early termination**, not just post-hoc truncation: with `--include-partial-messages` (enabled whenever `stop` is set and/or real streaming is active), the shim reads real token-level `content_block_delta` events and kills the `claude` subprocess (`proc.terminate()`) the instant a stop sequence appears in the accumulated text -- before the rest of the response is generated or billed. Verified live: a 500-word-story prompt took 22.3s with no stop sequence vs 4.9s with one that matched early in the output (~4.5x faster, plus the unbilled remainder of the response), and 5.2s when combined with real streaming. A message-level check remains as a safety net for edge cases (e.g. the overall timeout firing first). One accepted tradeoff: when the match spans across two token-level deltas, the client may have already received a few extra characters in the delta that completes the match, before the shim could detect it -- the FINAL recorded/session-cached text is still correctly truncated either way, only the live SSE stream itself briefly shows a few extra characters in that edge case. |
| `n` (multiple choices) | ⚠️ | Implemented up to a bounded limit (`MAX_N_CHOICES`, currently 5) -- each choice is a fully independent fresh Claude Code subprocess call (no session caching across choices, by design: parallel-choice semantics don't fit single-session continuity). Requests above the limit get a proper `400 invalid_request_error`, not silent truncation. |
| `response_format: {"type": "json_object"}` / `{"type": "json_schema", ...}` | ✅ | Mapped to `--json-schema` (a real Claude Code flag). `json_object` maps to an open `{"type": "object"}` schema; `json_schema` passes the caller's schema through directly. Requires `--max-turns 3` internally (Claude Code's own corrective retry mechanism via an internal `StructuredOutput` tool) -- verified empirically that `max_turns 1` leaves it hanging at `error_max_turns` with the schema never actually produced. Verified live: returns genuinely valid, parseable JSON, not markdown-fenced. |
| `temperature`, `top_p`, `seed`, `logprobs`, `presence_penalty`/`frequency_penalty`, `logit_bias` | ⚠️ | Genuine CLI limitation, not a shim gap -- no equivalent flags exist anywhere in Claude Code (checked `claude --help` and the official env-vars docs). These are silently **accepted** (never hard-errored -- real OpenAI clients routinely send explicit defaults like `temperature: 1.0` on every request, which carries no signal the caller actually wants non-default behavior, so erroring on presence would break compatibility for no benefit) but a one-time-per-parameter-name warning is written to the shim's own stderr (`claudecode-as-openai: warning: '<param>' was requested but Claude Code has no equivalent...`) so an operator running the shim can tell a request asked for something it can't honor. |
| `reasoning` (OpenRouter reasoning-tokens) | ✅ | Mapped to Claude Code's own `--effort <low\|medium\|high\|max>` flag -- see "OpenRouter reasoning-tokens compatibility" below. Verified live: setting `reasoning.effort` genuinely produces real extended-thinking content, surfaced back to the client as `message.reasoning` (plaintext) and `message.reasoning_details` (OpenRouter's structured array shape, signature preserved). |
| `usage.prompt_tokens_details.cached_tokens` | ✅ | The shim already tracks real prompt-cache-hit counts internally (`cache_read_input_tokens`); now also emitted in OpenAI's actual nested shape (`usage.prompt_tokens_details.cached_tokens`) alongside the existing flat custom field, so clients that specifically parse the standard nested structure (several cost-tracking dashboards/proxies do) get real numbers instead of ignoring a field they don't recognize. `completion_tokens_details.reasoning_tokens` is intentionally NOT fabricated -- Claude Code's usage payload has no separate reasoning-token count (verified live: `output_tokens` already includes any thinking-block tokens, undifferentiated), so no honest number exists to report there. |
| Vision / image content blocks | ❌ | Silently dropped. `_flatten_content()` only extracts `type: "text"` blocks from multi-part content; `image_url` blocks are discarded with no error, no warning, no image ever reaching Claude. |
| Local MCP tool leakage | ✅ **fixed** | `--disallowedTools` alone only blocks Claude Code's own named built-ins (Bash, Read, Write, etc.) -- it does nothing against MCP servers configured in the invoking user's local `~/.claude` settings, which could otherwise be dispatched unpredictably even on a request that declared no tools. Fixed for the no-tools-requested path via `--tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}'` together (verified: `--tools ""` alone still leaked real MCP tool calls; the combination gets `tools available: []`, fully closed). Note: when tools ARE requested, native `tool_use` dispatch is still used (see "Current approach" below) and this full lockdown does not apply there. |
| CWD / local file exposure | ✅ **fixed** | Every spawned `claude` subprocess previously inherited the shim's own working directory, and Claude Code is aware of and can reference real files there (confirmed live: a "list files using any tool you have" request surfaced real project file references). Fixed by spawning every `claude` call with `cwd=` pointed at a dedicated, empty, per-run sandbox temp directory instead. |
| Local hook/settings leakage | ✅ **fixed** | Every spawned `claude` call previously read the invoking user's real `~/.claude/settings.json` (user/project/local scopes), meaning a locally-configured hook (`SessionStart`, etc) could silently execute and inject arbitrary extra text into every completion this shim serves. Verified live (2026-08-14): a real `SessionStart` hook fired and injected a marker string into model context on a plain call. Fixed via `--setting-sources ""` on every invocation, which excludes user/project/local settings.json entirely (enterprise-managed `policySettings`/`flagSettings` remain unaffected either way, by design). Verified live: with this flag, the model explicitly confirmed no hook message was present, while OAuth auth, tool_use dispatch, and session resume all continued to work correctly (auth reads from a separate credentials file, not settings.json). |
| Error responses | ✅ | Claude Code's real structured error signal (`chunk["error"]`, e.g. `"invalid_request"`, or `result` chunks with `is_error: true`) is now classified and translated into a proper OpenAI-shaped `{"error": {"message", "type", "code"}}` body with a matching HTTP status (400/404/429/500/503 as appropriate) via `ClaudeCliError`/`_classify_error_text`. Verified live: an invalid model now returns HTTP 404 with `code: "model_not_found"`, not a silent `200 OK` with the complaint text stuffed into `content`. Note: for the real-streaming path, an error occurring mid-stream (after the SSE `200` headers are already committed) is surfaced as a final `content` delta plus `finish_reason: "stop"`, not a fresh HTTP error status -- the status line can't be changed once streaming has started. |
| Authentication | ❌ | None. Any `Authorization` header is accepted or ignored; anyone who can reach the port can use it. Fine for `127.0.0.1`-only binding (the default), a real gap if ever bound to `0.0.0.0`. |
| `/v1/models` | ✅ **fixed** | Now queries the real, current Anthropic model list from `https://api.anthropic.com/v1/models` (a free metadata call, not a billed completion) instead of returning a hardcoded 3-entry guess. Auth preference order: (1) Claude Code's own OAuth access token from `~/.claude/.credentials.json` -- the same subscription credentials `claude` itself uses day-to-day, so this works without requiring a separate `ANTHROPIC_API_KEY` and keeps the shim's whole "avoid metered billing" premise intact; (2) `ANTHROPIC_API_KEY` from the environment, if OAuth is absent/expired; (3) a hardcoded fallback list (extracted from strings in the compiled `claude` binary) if neither auth method works or the request fails for any reason. Verified live: returns the real current 10-model list (`claude-opus-5`, `claude-sonnet-5`, `claude-sonnet-4-6`, etc.) via OAuth with no `ANTHROPIC_API_KEY` set at all; verified the fallback chain with no credentials present at all correctly returns the hardcoded 3-entry list. Results are cached in-process for 5 minutes to avoid a network round-trip on every poll. |

**Bottom line:** model selection, session caching, error translation,
`max_tokens`, `response_format`, `tool_choice: "none"`, bounded `n`, the
MCP tool leakage, the CWD file-exposure issue, real token-level
streaming, a real `/v1/models` list, and now reliable (100% verified
turn-1) tool-call dispatch via native MCP registration are all
implemented and live-verified. The one remaining genuine gap is raw
sampling parameters (`temperature`/`top_p`/etc), which Claude Code
doesn't expose anywhere in its CLI -- these are accepted with a
one-time stderr warning rather than hard-erroring or silently doing
nothing without telling anyone.

## `/v1/models`: querying the real list

Claude Code has no `models list` subcommand, and the invalid-model error
text doesn't enumerate valid options either -- but the Anthropic SDK
bundled inside the compiled `claude` binary does have a real
`GET /v1/models` endpoint (confirmed by inspecting strings/logic in the
binary itself). This endpoint is a free metadata call, not a billed
completion.

Two ways to authenticate it, tried in this preference order:

1. **OAuth** (`~/.claude/.credentials.json`, `claudeAiOauth.accessToken`)
   -- Claude Code's own subscription credentials. Verified live: a Bearer
   token read from this file authenticates successfully against the real
   endpoint, expiry is checked against the same `expiresAt` field Claude
   Code itself uses so an expired token isn't used for a doomed request.
   Preferred because it requires no extra configuration and keeps the
   shim's "avoid metered API billing" premise intact even for this
   metadata call.
2. **`ANTHROPIC_API_KEY`** from the environment, as a fallback if OAuth
   is absent, expired, or unreadable.

If neither works (offline, revoked token, network failure, whatever),
falls back to a hardcoded list extracted from strings embedded in the
compiled `claude` binary -- stale by definition, but never worse than
before this feature existed.

## OpenRouter model-name compatibility

OpenRouter (https://openrouter.ai) is a widely-used OpenAI-compatible
proxy in front of many providers, and a lot of existing OpenAI-shaped
tooling already sends its Anthropic model naming convention in the
`model` field -- e.g. `anthropic/claude-sonnet-4.5`,
`~anthropic/claude-sonnet-latest` (see
https://openrouter.ai/~anthropic/claude-sonnet-latest for the reference
naming scheme). Claude Code's own `--model` flag uses a different
convention (dashed version numbers, `claude-sonnet-4-5`, plus bare
`sonnet`/`opus`/`haiku` aliases that resolve to whatever is currently
"latest").

`normalize_model_name()` translates the former into the latter before
every `claude -p --model <value>` invocation, so a client already
configured for OpenRouter-style model names works against this shim
without any changes on the caller's side. Each transform below was
verified live against the real `claude` CLI (`claude --model <x> -p`
with a "state your exact model version" probe prompt, 2026-08-14):

| Input | Normalized to | Result |
|---|---|---|
| `~anthropic/claude-sonnet-latest` | `sonnet` | resolved live to `claude-sonnet-4-6` |
| `anthropic/claude-opus-latest` | `opus` | resolved live to `claude-opus-4-6` |
| `anthropic/claude-haiku-latest` | `haiku` | (bare alias, resolves to Claude Code's current "latest" haiku) |
| `anthropic/claude-sonnet-4.5` | `claude-sonnet-4-5` | dot-to-dash; the dotted form is rejected outright by `--model` ("may not exist or you may not have access to it"), confirmed live |
| `anthropic/claude-opus-4.8-fast` | `claude-opus-4-8` | OpenRouter's Fast-mode suffix is stripped (no `-p`-mode equivalent exists to route to), with a one-time stderr warning |
| `claude-sonnet-4-5-20250929` (already-native) | unchanged | passed straight through |

The `~` prefix (OpenRouter's "auto-routed to latest" marker) and the
`anthropic/` provider prefix are both stripped unconditionally before
matching, so `~anthropic/claude-sonnet-latest`,
`anthropic/claude-sonnet-latest`, and `claude-sonnet-latest` are all
equivalent inputs. Anything that doesn't match a recognized OpenRouter
pattern (bare Claude Code aliases, full dated model IDs, or any other
model string entirely, e.g. an OpenAI or other-provider `model` value a
caller might send by mistake) passes through completely unchanged --
this is a compatibility translation layer, not a validator, and errors
on an unrecognized model are left to Claude Code's own real error
response (see "Error responses" above).

## OpenRouter reasoning-tokens compatibility

OpenRouter's `reasoning` request parameter
(https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) is
how OpenAI-shaped clients ask a model to think step-by-step and get
that thinking back in the response. Claude Code's own equivalent is
`--effort <low|medium|high|max>` -- verified live (via a stream-json
capture with a math prompt) that setting `--effort` genuinely produces
real `thinking` content blocks (each with a cryptographic `signature`
field) ahead of the final `text` block, not a cosmetic no-op.

`resolve_reasoning_effort()` extracts an effort level from either of
OpenRouter's real request shapes:

- `{"reasoning": {"effort": "high"}}` -- mapped directly. `--effort`
  only accepts exactly `low`/`medium`/`high`/`max` (verified live:
  `none`, `minimal`, and `xhigh` are all rejected outright with
  `"argument ... is invalid. It must be one of: low, medium, high,
  max"`), so OpenRouter's wider vocabulary is clamped to the nearest
  accepted value (`minimal` -> `low`, `xhigh` -> `max`) rather than
  passed through and erroring the whole request. `effort: "none"`
  disables reasoning entirely (no `--effort` flag at all).
- `{"reasoning": {"max_tokens": N}}` -- Anthropic-style token budget,
  approximated to an effort level via banding, since Claude Code's `-p`
  mode has no raw token-budget flag to pass this through to directly.
- `{"reasoning": {"enabled": true}}` alone -> `medium`, matching
  OpenRouter's own documented default.

The captured `thinking` text (plus its signature) is surfaced back to
the client as both of OpenRouter's supported response shapes:
`message.reasoning` (a plain string, for simple consumers) and
`message.reasoning_details` (the structured array shape, type
`reasoning.text`, `format: "anthropic-claude-v1"` -- the exact format
OpenRouter itself uses to tag real Anthropic thinking blocks, signature
preserved for consumers that round-trip reasoning back into a
follow-up request). Neither field is present at all when no reasoning
was requested or none was produced, rather than emitting an empty
placeholder.

`reasoning.exclude: true` (OpenRouter's "think but don't show me")
is accepted but not enforced -- Claude Code has no mechanism to reason
internally while withholding the thinking block from the transcript,
so this shim's reasoning is always returned when requested.

Verified live end-to-end through the actual running shim: a real
`curl` request with `reasoning: {"effort": "high"}` against
`/v1/chat/completions` returned a genuine `thinking` block's content in
both `message.reasoning` and `message.reasoning_details[0].text`, with
a real signature attached; a plain request with no `reasoning` key
carried neither field at all.

One consequence: live token-level SSE streaming (see "Real streaming"
below) is skipped whenever `reasoning` is set, falling back to the
existing buffered-then-emit behavior. The streaming path's
`stream_callback` only hooks `text_delta` events, and correctly
capturing the `thinking` block requires seeing it arrive BEFORE the
`text` block starts -- adding a second callback path for
`thinking_delta` was judged not worth the complexity for what is, in
practice, a fairly rare combination (reasoning + real-time streaming
UI) versus every other already-working combination.

## Usage reporting: OpenAI's nested `usage.prompt_tokens_details` shape

This shim has tracked real prompt-cache-hit/-creation counts internally
since the session-caching work (`cache_read_input_tokens`,
`cache_creation_input_tokens`), but previously only exposed them as
flat custom keys on the `usage` object -- a shape most OpenAI-facing
tooling doesn't recognize. `_build_openai_usage()` now ALSO nests the
cache-read count into OpenAI's actual documented shape,
`usage.prompt_tokens_details.cached_tokens`
(https://platform.openai.com/docs/api-reference/chat/object), so
clients/dashboards that specifically parse that nested structure get
real numbers instead of silently ignoring a field they don't
recognize. The flat custom keys are kept unchanged for backward
compatibility with anything already reading them directly.

`usage.completion_tokens_details.reasoning_tokens` is intentionally NOT
fabricated: Claude Code's own `usage` payload has no separate
reasoning-token count (verified live -- `output_tokens` already
includes any thinking-block tokens, undifferentiated), so there is no
honest number to report there. Making one up would be exactly the kind
of unmeasured-estimate-presented-as-fact this project avoids.

## Hook/settings isolation: `--setting-sources ""`, not `--bare`

Without any lockdown, every spawned `claude` call previously read the
invoking user's real `~/.claude/settings.json` (user/project/local
scopes). This matters because settings can configure hooks (e.g.
`SessionStart`) that execute arbitrary local commands and inject their
output as extra context into every completion -- verified live
(2026-08-14): a real `SessionStart` hook fired and injected a marker
string into model context on a plain call, invisibly to the API caller.

`--bare` mode looked like the obvious fix (it also skips CLAUDE.md
discovery, plugin sync, LSP, and more), but it was ruled out: `--bare`
strictly requires `ANTHROPIC_API_KEY` and never reads OAuth or the
system keychain (confirmed both in the CLI's own `--help` text and by
inspecting `isAnthropicAuthEnabled()`/`getApiKey()` in a reconstructed
source reference for an early public version of Claude Code -- see the
note below on how that reference was used). Verified live: `--bare`
fails with "Not logged in · Please run /login" under this shim's normal
OAuth-only setup, and only succeeds once a real `ANTHROPIC_API_KEY` is
exported. Since this shim's entire premise is riding the user's Claude
subscription instead of metered API billing, adopting `--bare` would
silently break every request for exactly the audience this project
serves.

The actual fix: `--setting-sources ""` on every invocation, which
excludes user/project/local settings.json entirely (enterprise-managed
`policySettings`/`flagSettings` remain unaffected by this flag either
way -- `getEnabledSettingSources()` always includes those regardless).
Verified live: with this flag, an injected test hook no longer fires at
all (the model explicitly confirmed no hook message was present in its
context), while OAuth auth, tool_use dispatch, and `--session-id`/
`--resume` continuity all continued to work correctly -- auth reads from
a separate credentials file (`~/.claude/.credentials.json`), not from
settings.json, so it's untouched by this flag.

Separately: `tool_choice: "required"` / forcing a specific tool has no
reachable path from `-p`/`--print` mode at all -- not a flag we're
missing, but a case the CLI's own request-building code hardcodes
`toolChoice: undefined` for on every regular turn (confirmed via the
same source reference). This is a genuine, unresolvable CLI limitation
from the shim's side.

**A note on how these findings were reached:** both were cross-checked
against a third-party, explicitly "unofficial, research purposes only"
reconstruction of an early public Claude Code release (TypeScript
recovered from a published npm package's source map, not Anthropic's
actual source repository). It was used strictly read-only, to confirm
or falsify conclusions already reached by black-box CLI probing -- no
code from that reconstruction was copied into this shim.

## Native MCP tool registration: the real fix for tool-call reliability

The single biggest reliability gap this shim had was tool-call
dispatch: `-p`/`--print` mode has no flag to pass arbitrary JSON-schema
tool definitions directly to the underlying Anthropic API, so the
original approach described custom tools in prose inside the system
prompt (see `render_tools_into_system_prompt`, still kept as a
fallback). That got real `tool_use` blocks, but only ~40% reliably per
call -- the model would frequently narrate a plan instead of actually
dispatching the tool, requiring a bounded retry loop to paper over it.

**The fix, verified live (2026-08-14):** Claude Code's MCP integration
*is* a fully supported, documented `-p`-mode capability (`--mcp-config`,
`--strict-mcp-config`, `--allowedTools`), and MCP tool schemas
registered this way ARE passed through to the real Anthropic API as
genuine, JSON-schema-validated `tools` entries -- exactly the same
mechanism a first-class built-in tool uses. By default, MCP tools go
through Claude Code's internal "ToolSearch" deferred-loading (the model
has to search for the tool by name in one turn, then dispatch it in a
second), but setting `_meta: {"anthropic/alwaysLoad": true}` on a tool
definition skips that indirection entirely and puts the full schema in
the initial prompt turn, just like a built-in.

This shim now builds a small MCP tool manifest from the OpenAI-shaped
`tools` array on every request (`build_mcp_tool_config` in
`shim.py`), each tool marked `alwaysLoad: true`, and points
`--mcp-config` at a bundled stdio server (`mcp_tool_server.py`) that
serves that manifest and executes nothing itself -- `tools/call`
requests never actually reach it in practice, because the shim
terminates the `claude` subprocess the instant it observes the
`tool_use` content block on the wire (the same "kill fast" pattern
already used for stop-sequence early termination), so the real
dispatch/execution stays entirely on the OpenAI-client side, matching
how `tool_calls` is supposed to work in the OpenAI API shape. Verified
live: the manifest server's own `tools/call` handler never fires even
once across repeated trials, confirmed by instrumenting it to log
every call it receives.

**Transport is stdio, not HTTP/SSE, deliberately** -- this must never
bind a TCP port (no port-clash risk, nothing exposed on any network
interface, ever). The MCP server is a plain child process of `claude`,
wired via stdio pipes only, same as Claude Code's other local MCP
integrations.

**Reliability, verified live:** 8/8 turn-1 dispatch in an initial raw
CLI probe, 5/5 in a repeat, then 5/5 again through the actual running
shim end-to-end (`curl` against `/v1/chat/completions`), each returning
a correct OpenAI-shaped `tool_calls` entry with the right function name
and arguments. Correctly does NOT force a tool call on an unrelated
question, and correctly picks the right tool among multiple registered
tools (verified with a two-tool manifest: "what's the weather" vs
"what's the time" each dispatched the correct one).

**Prompt caching stays intact for a stable tool set.** The manifest is
written to a per-request temp file, but the *MCP server command itself*
(the script path + interpreter) stays byte-identical across calls --
only the manifest file path differs if the tool set changes. Anthropic
prices tool schemas as part of the cacheable system context, and this
was verified directly: a 5-turn `--resume`'d conversation with a stable
tool set showed `cache_creation_input_tokens` flat at ~110-140 tokens
per turn after the first (vs ~15,800 on the first turn), with
`cache_read_input_tokens` climbing normally -- the existing session
caching mechanism (`--session-id`/`--resume`, see above) already
handles this correctly with no extra work needed. Conversely, changing
the tool set mid-session (an OpenAI client declaring different tools
between turns while somehow still trying to resume the same session)
was verified to bust the cache from that point forward exactly like
Anthropic's own docs describe for any system-prompt change --
`cache_creation_input_tokens` jumped straight back to ~16,000, a full
fresh write. This is expected, correct behavior, not a shim bug: a
changed tool schema genuinely is a changed system context from
Anthropic's point of view, regardless of what produced the change.

**Compatible with callers who have their own real MCP servers.** The
OpenAI chat-completions `tools` array has no concept of "MCP server"
at all -- whatever produced a caller's declared tools (their own MCP
servers, hardcoded definitions, anything else) is already fully
resolved into flat OpenAI function schemas by the time a request
reaches this shim. This shim's own MCP server is a pure internal
implementation detail on the Claude-Code-facing side of that boundary;
callers never see "MCP" anywhere in the round trip, exactly as with the
earlier prose-based approach.

**Falls back to the prose-description path automatically** for any
tool whose name can't satisfy MCP's `^[a-zA-Z0-9_-]{1,64}$` naming
constraint (checked in `build_mcp_tool_manifest`) -- this should be
rare in practice since OpenAI's own `function.name` field already
requires the same pattern, but is handled defensively rather than
silently dropping a tool a caller asked for.

**Not pursued: raw direct API calls with the CLI's OAuth token.**
Before landing on MCP registration, forging a direct
`https://api.anthropic.com/v1/messages` request using Claude Code's own
OAuth credentials (readable from `~/.claude/.credentials.json`) plus
the `oauth-2025-04-20` beta header was investigated and verified to
also reach 100% reliable tool dispatch (10/10) with full raw
`tool_choice` support -- but this was explicitly rejected: it means
building HTTP requests that impersonate the `claude` binary's own
internal auth flow to bypass `-p` mode entirely, which is a
categorically different (and clearly out-of-bounds) approach compared
to registering a real, documented MCP server and letting Claude Code's
own request-building logic do the work.

## Real streaming: the PTY trick

`claude`'s own stdout is fully buffered (not line-buffered) whenever its
stdout is a plain pipe with no terminal attached -- this is standard
glibc stdio behavior (`stdio` only flushes on a full buffer or process
exit when `isatty()` is false), not a Claude Code bug, but it means naive
`subprocess.Popen(stdout=subprocess.PIPE)` gets `--include-partial-messages`
events in 2-3 giant bursts instead of spread across the actual generation
time. There's no CLI flag to force unbuffered/line-buffered output.

The fix: spawn `claude` with a real PTY (`pty.openpty()`) as its stdout
file descriptor. A PTY makes the child think it's talking to an
interactive terminal, which is exactly the condition under which glibc's
stdio switches to line buffering. Verified live: the same
`--include-partial-messages` request that produced 2-3 bursts over a
plain pipe produced 21 individually-timed deltas spread realistically
across ~22s of generation through a PTY.

This only activates for the safe streaming path (single choice, no tools
requested, no `response_format: json_schema` -- see `can_stream_live` in
`shim.py`): with tool retry in play, a failed attempt's narration text
must not reach the client before the shim knows to discard it and retry,
and `--json-schema`'s multi-turn corrective mechanism doesn't map cleanly
onto a single token stream either. Every other combination keeps using
the previous buffered-then-emit SSE behavior (protocol-correct, just not
real-time).

## Current approach: native MCP tool registration + bounded retry fallback

`claudecode_as_openai/shim.py` calls `claude -p --output-format
stream-json` and parses the real Anthropic `tool_use` content blocks from
the NDJSON stream (not a hand-rolled text convention). Custom tools are
registered as a real MCP tool server via `--mcp-config` (see "Native MCP
tool registration" above) -- verified live to reach 100% turn-1 dispatch
reliability. Claude's own built-in tools (Bash, Read, Edit, etc.) are
blocked via `--disallowedTools <enumerated list>` so they don't compete
for dispatch, whichever tool-registration path is in use.

**Legacy fallback path (prose description in the system prompt):** used
only when a tool name can't satisfy MCP's naming constraint (see
`build_mcp_tool_manifest`). This path only gets real `tool_use` blocks
~40% reliably per single call (Claude sometimes narrates "let me
check..." instead of dispatching — matches the failure mode in
[cline/cline#10336](https://github.com/cline/cline/issues/10336), a
community workaround for an unrelated bug in Cline's *own* agentic XML
tool vocabulary colliding with Claude Code's native tools; it does not
reflect Cline's actual Claude Code provider). Mitigation is a bounded
retry with exponential backoff (`call_claude_with_tool_retry`, up to 4
retries / 5 total attempts, 2s base / 15s cap) when tools were requested
but no `tool_use` came back. Verified 15/15 in manual testing with retries
vs 6/15 without -- but this is *probabilistic*, not deterministic:
expected success asymptotically approaches but never reaches 100%. The
same retry wrapper still wraps the MCP path too, as a safety net, but in
practice shouldn't need more than its first attempt.

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

**Update (2026-08-14): superseded for the common case by native MCP
registration** (see "Native MCP tool registration" above). The
conclusion above still holds for `tool_choice: "required"`-style forcing
(genuinely not exposed anywhere in the CLI/SDK harness), but the
separate "narrates instead of dispatching" problem this retry loop was
built for turned out to have a real fix after all -- not via a
`tool_choice` knob, but by using MCP tool registration (a different,
fully supported Claude Code capability) to get the model to treat
custom tools as first-class, schema-backed tools instead of prose it
has to be talked into trusting. The retry loop remains in place as a
safety net for the legacy prose-fallback path and for any future
regression, but in practice shouldn't be needed once a tool set is
representable as a valid MCP manifest.

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
