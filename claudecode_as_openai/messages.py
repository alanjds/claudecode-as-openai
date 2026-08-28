#!/usr/bin/env python3
"""OpenAI -> Claude message translation."""

import json

from claudecode_as_openai.tools import _gen_tool_id


def _flatten_content(content):
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return content or ""


def build_claude_messages(openai_messages):
    """Translate a list of OpenAI chat messages into Claude Code's native
    message array shape: [{"role": "user"|"assistant", "content": <str or
    blocks>}]. System messages are skipped here (handled separately by the
    caller via system_prompt_from_messages) since Claude Code takes the
    system prompt as a separate CLI flag, not as an array element."""
    claude_messages = []

    for m in openai_messages:
        role = m.get("role", "user")
        if role == "system":
            continue

        if role == "tool":
            claude_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.get("tool_call_id", ""),
                            "content": _flatten_content(m.get("content")),
                        }
                    ],
                }
            )
            continue

        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            content_blocks = []
            text = _flatten_content(m.get("content"))
            if text:
                content_blocks.append({"type": "text", "text": text})
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", _gen_tool_id()),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            claude_messages.append(
                {"role": "assistant", "content": content_blocks or text}
            )
            continue

        # user (or anything else) -> plain user message
        claude_messages.append({"role": "user", "content": _flatten_content(m.get("content"))})

    return claude_messages


def system_prompt_from_messages(openai_messages):
    parts = [_flatten_content(m.get("content")) for m in openai_messages if m.get("role") == "system"]
    return "\n\n".join(p for p in parts if p)


