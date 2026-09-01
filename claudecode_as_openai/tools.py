#!/usr/bin/env python3
"""MCP tool manifest/config building and tool-name mangling for native
Claude Code tool-calling."""

import hashlib
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
from claudecode_as_openai.state import (
    _MCP_MANIFEST_CACHE,
    _MCP_MANIFEST_CACHE_MAX,
    _MCP_MANIFEST_LOCK,
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


def _manifest_path_for(manifest_json):
    """Return a temp file path holding manifest_json, reusing a cached one
    for the exact same content instead of writing a fresh file every call.

    Every earlier version of this function wrote a brand-new temp file
    (and therefore a changed --mcp-config, since the manifest path is part
    of the MCP server's launch args) on every single call, even when the
    tool set was byte-identical to the previous call -- defeating the very
    cache-stability this module's docstring already claimed. A real live
    divergence (2026-09-01: Claude repeatedly re-issuing the exact same
    tool call across an otherwise-correctly-resumed session, never seen
    with either a full-history resend or the direct Anthropic API for the
    same task) is consistent with Claude Code's own --resume continuity
    treating an always-changing --mcp-config as "the tool config changed"
    on every resumed turn. Caching by content hash makes a stable tool set
    produce a byte-identical --mcp-config across an entire resumed
    session, matching the documented (but previously unimplemented) intent
    -- worth keeping regardless of whether it turns out to be the fix for
    that specific symptom.

    Cache is owned entirely here: entries are evicted LRU-style (oldest
    first) once _MCP_MANIFEST_CACHE_MAX distinct tool sets are cached,
    unlinking the evicted file -- callers must NOT unlink manifest_path
    themselves any more (see streaming.py/warm_pool.py teardown, which no
    longer do)."""
    content_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    with _MCP_MANIFEST_LOCK:
        cached_path = _MCP_MANIFEST_CACHE.get(content_hash)
        if cached_path is not None and os.path.exists(cached_path):
            _MCP_MANIFEST_CACHE.move_to_end(content_hash)
            return cached_path
        fd, manifest_path = tempfile.mkstemp(prefix="claudecode-as-openai-mcp-manifest-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write(manifest_json)
        _MCP_MANIFEST_CACHE[content_hash] = manifest_path
        _MCP_MANIFEST_CACHE.move_to_end(content_hash)
        while len(_MCP_MANIFEST_CACHE) > _MCP_MANIFEST_CACHE_MAX:
            _, evicted_path = _MCP_MANIFEST_CACHE.popitem(last=False)
            try:
                os.unlink(evicted_path)
            except OSError:
                pass
        return manifest_path


def build_mcp_tool_config(tools):
    """Returns a dict {"mcp_config", "manifest_path", "allowed_tools"} for
    the given OpenAI tools array, or None if tools couldn't be represented
    as an MCP manifest (see build_mcp_tool_manifest). allowed_tools holds
    the "mcp__<server>__<name>" forms Claude Code reports tool_use blocks
    under -- callers must strip this prefix again before handing the name
    back to an OpenAI client (see strip_mcp_tool_prefix). The manifest is
    written to a temp file (not inlined into the mcp_config JSON) so the
    *server command* stays byte-identical across calls; _manifest_path_for
    additionally keeps the file path itself stable whenever the tool set's
    content is stable, so the whole --mcp-config stays byte-identical too
    -- not just the command shape -- across a resumed session with an
    unchanged tool set (see README \"MCP native tool registration\")."""
    manifest = build_mcp_tool_manifest(tools)
    if manifest is None:
        return None
    manifest_json = json.dumps(manifest, sort_keys=True)
    manifest_path = _manifest_path_for(manifest_json)
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
