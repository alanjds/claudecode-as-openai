"""Shared constants across claudecode-as-openai modules."""

import os
import re

# MCP tool registration
_MCP_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MCP_SERVER_NAME = "shim_tools"
_MCP_TOOL_SERVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mcp_tool_server.py"
)

# Claude subprocess defaults
_CLAUDE_BIN = "claude"
CLAUDE_TIMEOUT_S = 300

# Applied to every spawned `claude` subprocess (see _scoped_env_overrides in
# server.py). CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC bundles
# DISABLE_AUTOUPDATER, DISABLE_TELEMETRY, DISABLE_ERROR_REPORTING, and
# DISABLE_FEEDBACK_COMMAND into one flag. Since this shim spawns a fresh
# `claude -p` process per request, the startup work those four disable
# (autoupdater version check, telemetry, error reporting, feedback prompts)
# is otherwise paid on every single call -- disabling it is a real
# per-request latency win. This also means `claude` will never self-update
# here; the user updates it manually on their own schedule instead.
_BASE_ENV_OVERRIDES = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}

# Claude Code's built-in tool names, PLUS the MCP-adjacent helper tools
# that also leak custom-tool dispatch surface: RemoteTrigger (a generic
# dispatcher that can invoke ANY declared custom tool by name), and
# ListMcpResourcesTool/ReadMcpResourceTool (MCP resource browsers, not
# server-specific so --strict-mcp-config doesn't touch them). This does
# NOT fully close the leak when tools ARE requested -- see README "MCP
# tool leakage". When NO tools are requested, the no-tools path
# (--tools "" --strict-mcp-config) closes this completely instead.
EXTENDED_DISALLOWED_TOOLS = ",".join(
    [
        "Task", "TaskOutput", "Bash", "Glob", "Grep", "Read", "Edit", "Write",
        "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch", "TaskStop",
        "AskUserQuestion", "Skill", "EnterPlanMode", "ExitPlanMode",
        "EnterWorktree", "ExitWorktree", "CronCreate", "CronDelete",
        "CronList", "ToolSearch", "RemoteTrigger", "ListMcpResourcesTool",
        "ReadMcpResourceTool",
    ]
)

# Model defaults
DEFAULT_MODEL = "sonnet"
MAX_N_CHOICES = 5
# Default 0: native MCP tool registration already reaches ~100% turn-1
# tool-dispatch reliability (see CHANGELOG "Tool-call reliability fix"),
# so a turn that declares tools but doesn't call one is now overwhelmingly
# more likely to be a correct "no tool needed" response than a genuine
# dispatch failure -- retrying it just discards session/warm-pool
# continuity for no benefit. Kept as a safety net (see
# call_claude_with_tool_retry) for any future regression -- override via
# CLAUDE_OPENAI_TOOL_CALL_MAX_RETRIES to re-enable without a code change.
# When origin session_mode is "resume", a retry forks a new session from
# the original checkpoint via --fork-session (cheap: cache reuse, not a
# full-history resend) instead of starting over fresh.
try:
    TOOL_CALL_MAX_RETRIES = int(os.environ.get("CLAUDE_OPENAI_TOOL_CALL_MAX_RETRIES", "0"))
except ValueError:
    TOOL_CALL_MAX_RETRIES = 0
_OPENROUTER_ALIASES = ("sonnet", "opus", "haiku")

# Diagnostic/operational escape hatch: forces every turn onto the cold path
# (a fresh `claude` subprocess per call) even when a parked WarmProcess would
# otherwise match, skipping the warm pool entirely -- no process is parked
# either, so nothing is ever spawned-and-discarded for a pool that will never
# be consulted. Added while investigating a real live regression where
# `--resume` measurably increases the model's tendency toward a redundant
# exploratory tool call (see CHANGELOG): the confirmed 3/4-of-4-trials repro
# used the cold `--resume` path exclusively, and the warm pool's live-process
# stream-json frame delivery was separately shown to mishandle at least one
# non-standard input shape -- this flag lets that be isolated empirically
# (does disabling the warm pool change the real, observed rate of redundant
# tool calls?) without a code change, and doubles as a plain operational
# toggle for anyone who wants to rule out the warm pool while debugging
# something else entirely.
WARM_POOL_DISABLED = os.environ.get("CLAUDE_OPENAI_DISABLE_WARM_POOL", "") == "1"
