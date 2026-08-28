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
TOOL_CALL_MAX_RETRIES = 4
_OPENROUTER_ALIASES = ("sonnet", "opus", "haiku")
