#!/usr/bin/env python3
"""OpenAI-chat-completions-compatible shim over the local Claude Code CLI.

This module is now a thin compatibility/entry-point layer: the actual
implementation lives in the split modules below (errors, models, tools,
messages, sessions, parsing, quota, warm_pool, streaming, server). Kept
around, and re-exporting their public + test-facing names, so:

  - `python3 -m claudecode_as_openai.shim [port]` / `claudecode-as-openai`
    (see pyproject.toml's [project.scripts]) keep working unchanged.
  - tests/test_shim_unit.py's white-box `shim.<name>` access (including
    `unittest.mock.patch.object(shim.subprocess, "Popen", ...)`-style
    patching of shared stdlib modules) keeps working -- patching an
    attribute on a stdlib module object patches it everywhere that module
    is imported, regardless of which of the modules below actually calls
    it, as long as they access it as `<module>.<attr>` rather than
    `from <module> import <attr>`.

See README.md "Capability audit" for the full, empirically-verified list
of what works, what's approximated, and what's a genuine CLI limitation,
and CHANGELOG.md for the investigation trail behind each design choice.

Run:
    python3 -m claudecode_as_openai.shim [port]   # default port 8977

Point Hermes at it:
    hermes config set model.provider custom
    hermes config set model.base_url http://127.0.0.1:8977/v1
    hermes config set model.api_key not-needed
    hermes config set model.default sonnet
"""
import json
import os
import pty
import subprocess
import sys
import time
import urllib.error
import urllib.request

from claudecode_as_openai.constants import TOOL_CALL_MAX_RETRIES
from claudecode_as_openai.state import _SESSION_STORE
from claudecode_as_openai.errors import ClaudeCliError, _classify_error_text
from claudecode_as_openai.models import (
    KNOWN_MODEL_ALIASES, _model_list_cache, _read_claude_oauth_token,
    _warned_fast_models, fetch_model_list, normalize_model_name,
    resolve_reasoning_effort,
)
from claudecode_as_openai.messages import build_claude_messages, system_prompt_from_messages
from claudecode_as_openai.tools import build_mcp_tool_config, build_mcp_tool_manifest, strip_mcp_tool_prefix
from claudecode_as_openai.sessions import record_session, resolve_session
from claudecode_as_openai.quota import _build_credits_response, _build_key_response
from claudecode_as_openai.streaming import (
    _apply_stop_sequences, _backoff_delay_s, _build_claude_cmd, _build_openai_usage,
    build_reasoning_details, call_claude_streaming, call_claude_with_tool_retry,
)
from claudecode_as_openai.server import (
    DEFAULT_PORT, Handler, _warn_unsupported_sampling_params, _warned_sampling_params,
)
from claudecode_as_openai.server import main as _http_main


def main():
    _http_main()


if __name__ == "__main__":
    main()
