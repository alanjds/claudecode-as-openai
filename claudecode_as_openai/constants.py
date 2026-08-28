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
_EXTENDED_DISALLOWED_TOOLS = [
    "Task", "TaskOutput", "Bash", "Glob", "Grep", "Read", "Edit", "Write",
    "NotebookEdit", "WebFetch", "TodoWrite", "WebSearch", "TaskStop",
    "AskUserQuestion", "Skill", "EnterPlanMode", "ExitPlanMode",
    "EnterWorktree", "ExitWorktree", "CronCreate", "CronDelete", "CronList",
    "ToolSearch", "RemoteTrigger", "ListMcpResourcesTool", "ReadMcpResourceTool",
]

# Model defaults
DEFAULT_MODEL = "sonnet"
MAX_N_CHOICES = 1
TOOL_CALL_MAX_RETRIES = 3
_OPENROUTER_ALIASES = ("sonnet", "opus", "haiku")
