#!/usr/bin/env python3
"""MCP tool manifest/config building and tool-name mangling for native
Claude Code tool-calling."""

import json
import os
import sys
import tempfile
import uuid

from claudecode_as_openai.constants import (
    _MCP_TOOL_NAME_RE,
    _MCP_SERVER_NAME,
    _MCP_TOOL_SERVER_PATH,
)


def _gen_tool_id():
    return f"toolu_{uuid.uuid4().hex[:20]}"


def build_mcp_tool_manifest(tools):
    """Translate an OpenAI `tools` array into an MCP tool manifest (list of
    {"name", "description", "inputSchema"}). Returns None if any tool name
    fails MCP's naming constraint -- callers drop the tools entirely in
    that case rather than registering a partial/incorrect manifest."""
    if not tools:
        return None
    manifest = []
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name")
        if not name or not _MCP_TOOL_NAME_RE.match(name):
            return None
        manifest.append({
            "name": name,
            "description": fn.get("description", "") or "",
            "inputSchema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return manifest


def build_mcp_tool_config(tools):
    """Returns a dict {"mcp_config", "manifest_path", "allowed_tools"} for
    the given OpenAI tools array, or None if tools couldn't be represented
    as an MCP manifest (see build_mcp_tool_manifest). allowed_tools holds
    the "mcp__<server>__<name>" forms Claude Code reports tool_use blocks
    under -- callers must strip this prefix again before handing the name
    back to an OpenAI client (see strip_mcp_tool_prefix). The manifest is
    written to a per-call temp file (not inlined into the mcp_config JSON)
    so the *server command* stays byte-identical across calls -- only the
    file path changes if the tool set itself changes, which keeps prompt
    caching intact whenever the actual tool set is stable across a
    resumed session (see README \"MCP native tool registration\" for the
    caching mechanics verified live)."""
    manifest = build_mcp_tool_manifest(tools)
    if manifest is None:
        return None
    fd, manifest_path = tempfile.mkstemp(prefix="claudecode-as-openai-mcp-manifest-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f, sort_keys=True)
    mcp_config = {
        "mcpServers": {
            _MCP_SERVER_NAME: {
                "command": sys.executable,
                "args": [_MCP_TOOL_SERVER_PATH, manifest_path],
            }
        }
    }
    allowed = [f"mcp__{_MCP_SERVER_NAME}__{t['name']}" for t in manifest]
    return {"mcp_config": mcp_config, "manifest_path": manifest_path, "allowed_tools": allowed}


_MCP_TOOL_PREFIX = f"mcp__{_MCP_SERVER_NAME}__"


def strip_mcp_tool_prefix(name):
    """Reverses the "mcp__<server>__" mangling Claude Code applies to MCP
    tool names in tool_use blocks, so the name reported back to an OpenAI
    client matches exactly what the caller originally declared."""
    if name and name.startswith(_MCP_TOOL_PREFIX):
        return name[len(_MCP_TOOL_PREFIX):]
    return name
