#!/usr/bin/env python3
"""Minimal MCP stdio server used by shim.py's native-tool-schema path.

Why this exists: Claude Code's `-p`/`--print` mode has no flag to accept
arbitrary JSON-schema tool definitions directly (no `--tools-schema`
equivalent), so tools requested by an OpenAI client were previously only
ever described as prose in the system prompt -- Claude would emit a
`tool_use` block anyway because it was convinced by the framing, but this
had no real JSON-schema validation and only ~40% single-shot reliability
(see TOOL_CALL_MAX_RETRIES in shim.py).

MCP registration is a real, fully documented, first-class Claude Code
extension mechanism (`--mcp-config`), and MCP tool schemas ARE passed to
the underlying Anthropic API as genuine, ajv-validated `tools` entries.
Verified live (2026-08-14): a tool declared via MCP with
`_meta["anthropic/alwaysLoad"] = True` skips Claude Code's internal
"ToolSearch" deferred-loading indirection and is dispatched on turn 1 with
100% reliability across repeated trials (8/8, then 5/5 in a follow-up
run), including correct selection among multiple declared tools and
correctly NOT firing on unrelated prompts.

Transport: stdio only, deliberately -- MCP also supports an HTTP/SSE
transport, but stdio needs no port binding at all (the server is a child
process of `claude`, wired via pipes), so there is no possibility of a
port clash, no port-scanning surface, and nothing to expose on any
network interface. This matches the user's explicit ask: sockets or
stdio, never a TCP port.

Execution semantics: this server NEVER actually executes a tool call.
shim.py kills the `claude` subprocess the instant it observes a `tool_use`
content block in the stream-json output -- verified live (2026-08-14) that
this reliably happens BEFORE the corresponding `tools/call` JSON-RPC
request reaches this server (checked via a logging build of this same
server: 3/3 kills landed with only `initialize`/`tools/list` ever
received, never `tools/call`). This server's own `tools/call` handler
exists purely as a defensive fallback (a stub response) in case that race
is ever lost for some future Claude Code version -- shim.py, not this
server, is the single source of truth for what happens when a tool is
"called": it translates the `tool_use` block straight into an OpenAI
`tool_calls` response and hands control back to the caller.

Protocol: JSON-RPC 2.0 over newline-delimited stdio, matching Claude
Code's stdio MCP client (no Content-Length framing needed).

Usage: `python3 -m claudecode_as_openai.mcp_tool_server <manifest_path>`
The manifest is a JSON file: a list of {"name", "description",
"inputSchema"} objects (already-sanitized MCP tool names -- see
build_mcp_manifest() in shim.py for the OpenAI-tools -> MCP-manifest
translation and naming convention).
"""
import json
import sys


def _send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _load_tools(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        tools = json.load(f)
    for tool in tools:
        # Forces turn-1 dispatch instead of Claude Code's deferred
        # "ToolSearch" indirection -- see module docstring.
        tool.setdefault("_meta", {})["anthropic/alwaysLoad"] = True
    return tools


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: mcp_tool_server.py <manifest_path>\n")
        sys.exit(1)
    manifest_path = sys.argv[1]

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = req.get("method")
        req_id = req.get("id")

        if method == "initialize":
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "claudecode-as-openai-tools", "version": "0.1.0"},
                },
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            try:
                tools = _load_tools(manifest_path)
            except (OSError, json.JSONDecodeError) as exc:
                _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(exc)}})
                continue
            _send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})
        elif method == "tools/call":
            # Defensive fallback only -- see module docstring. shim.py
            # kills the `claude` subprocess before this is normally ever
            # reached; a real reply here is never expected to be seen by
            # the end user.
            _send({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{
                        "type": "text",
                        "text": "error: this tool is dispatched by the calling OpenAI client, "
                                "not executed locally. If you see this message, the shim's "
                                "intercept-before-execution race was lost -- please report this.",
                    }],
                    "isError": True,
                },
            })
        elif req_id is not None:
            _send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "method not found"}})


if __name__ == "__main__":
    main()
