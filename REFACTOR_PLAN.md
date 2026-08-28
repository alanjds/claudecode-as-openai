# Shim Refactor Plan: Split into Modules

**Branch:** `refactor/split-shim-modules`  
**Current:** 2,616 lines in single `shim.py`  
**Target:** Modular structure with clean separation of concerns

## Proposed Modules

### 1. `errors.py` (60 lines)
- `ClaudeCliError` exception class
- `_classify_error_text()` error categorization
- **Remove:** Comments about prose fallback reliability

### 2. `models.py` (200 lines)
- `normalize_model_name()` - model name normalization
- `resolve_reasoning_effort()` - reasoning parameter handling
- `_anthropic_id_to_openrouter()` - model ID translation
- `_supports_reasoning_params()` - feature detection
- `_model_entry()` - model metadata building
- `fetch_model_list()` - fetch available models
- **Keep:** OAuth token fetching (internal)
- **Remove:** Dead code paths for deprecated models

### 3. `tools.py` (150 lines)
- `build_mcp_tool_manifest()` - MCP tool schemas
- `build_mcp_tool_config()` - MCP configuration
- `strip_mcp_tool_prefix()` - MCP name prefix handling
- `_gen_tool_id()` - tool ID generation
- **Remove:** Comments about prose fallback path (~10 lines)
- **Remove:** References to ~40% reliability (~5 lines)

### 4. `messages.py` (100 lines)
- `build_claude_messages()` - OpenAI→Claude message translation
- `system_prompt_from_messages()` - system prompt extraction
- `_flatten_content()` - content flattening

### 5. `sessions.py` (120 lines)
- `_conversation_key()` - session fingerprinting
- `_messages_equal()` - message comparison
- `resolve_session()` - session cache lookup/creation
- `record_session()` - session persistence
- Global: `_SESSIONS_DB`, `_SESSIONS_DB_LOCK`
- **Remove:** Outdated cache implementation notes

### 6. `warm_pool.py` (250 lines)
- `WarmPool` class - persistent process pooling
- `WarmProcess` class - warm process lifecycle
- `_parking_fingerprint()` - session fingerprinting for pooling
- Constants: `_WARM_POOL`, `CLAUDE_TIMEOUT_S`, `_CLAUDE_CWD`
- **Remove:** Comments about prose fallback causing issues
- **Update:** Comments about max_turns=1 safety

### 7. `quota.py` (150 lines)
- `_log_quota_snapshot()` - quota logging
- Global: `_rate_limit_cache`, `_rate_limit_cache_lock`
- `_build_key_response()` - /v1/key endpoint
- `_build_credits_response()` - /v1/credits endpoint

### 8. `parsing.py` (100 lines)
- `_parse_ndjson_line()` - NDJSON parsing
- `_iter_ndjson_lines()` - line iteration (stdio)
- `_iter_ndjson_lines_pty()` - line iteration (PTY)

### 9. `streaming.py` (400 lines)
- `call_claude_streaming()` - main streaming entry point
- `_consume_claude_response()` - response parsing
- `_run_one_completion()` - single turn execution
- `_build_claude_cmd()` - subprocess command building
- Helper: `_normalize_stop_sequences()`, `_backoff_delay_s()`, `call_claude_with_tool_retry()`
- Reasoning: `build_reasoning_details()`, `_build_openai_usage()`
- **Remove:** Comments about prose fallback needing retry (~5 lines)

### 10. `http.py` (200 lines)
- `Handler` class - HTTP request handler
- `ThreadingHTTPServer` - HTTP server
- Helper: `_warn_unsupported_sampling_params()`, `_apply_stop_sequences()`, `_build_sse_chunk()`, `_scoped_env_overrides()`

## Cleanup Opportunities

### Comments to Remove/Update
1. Line ~1690: "MCP tool dispatch (and prose fallback, now removed)" → "MCP tool dispatch"
2. Line ~1850: "~40% single-shot" → remove percentage reference
3. Line ~2197: "(MCP is the only supported path now; prose fallback removed)" → "(MCP is the only supported path)"
4. Line ~889: Update warm_pool comment about max_turns safety

### Code to Remove
1. Dead model aliases (deprecated OpenRouter names)
2. Old error handling for ~40% tool call failure
3. Retry loop documentation that referenced prose fallback

## Dependency Graph
```
errors.py (no deps)
  ↓
models.py → errors.py
tools.py → errors.py
messages.py
sessions.py → messages.py
parsing.py
quota.py
warm_pool.py → sessions.py, quota.py, parsing.py
streaming.py → errors.py, tools.py, messages.py, parsing.py, warm_pool.py, quota.py
http.py → streaming.py, models.py, errors.py
shim.py (main) → all modules
```

## Execution Order
1. Extract `errors.py` (no dependencies)
2. Extract `models.py` (depends on errors)
3. Extract `tools.py` (depends on errors)
4. Extract `messages.py` (no deps)
5. Extract `sessions.py` (depends on messages)
6. Extract `parsing.py` (no deps)
7. Extract `quota.py` (no deps)
8. Extract `warm_pool.py` (depends on sessions, quota, parsing)
9. Extract `streaming.py` (depends on errors, tools, messages, parsing, warm_pool, quota)
10. Extract `http.py` (depends on streaming, models, errors)
11. Update `shim.py` to import and delegate to modules
12. Clean up obsolete comments throughout

## Success Criteria
- ✓ All 10 modules load without import errors
- ✓ Circular dependencies eliminated
- ✓ Single responsibility per module
- ✓ Obsolete comments removed
- ✓ All tests pass with modular structure
- ✓ `shim.py` remains as thin orchestration layer
