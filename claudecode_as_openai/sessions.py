#!/usr/bin/env python3
"""Session caching: fingerprint a conversation and track how much of it has
already been synced to a Claude Code session, so a continuing conversation
can send only the new trailing messages via --resume instead of the full
history every time. Verified live: a resumed turn re-processes only the
delta (a few hundred tokens) instead of the whole conversation."""

import hashlib
import json
import uuid

from claudecode_as_openai.state import _SESSION_STORE, _SESSION_STORE_MAX, _SESSION_LOCK


def _conversation_key(openai_messages):
    """Stable key for the conversation this message list belongs to: hash
    of the system messages + the first non-system message. Stays stable
    across the whole conversation even as more turns are appended."""
    sys_msgs = [m for m in openai_messages if m.get("role") == "system"]
    first_non_system = next((m for m in openai_messages if m.get("role") != "system"), None)
    basis = json.dumps([sys_msgs, first_non_system], sort_keys=True, default=str)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _messages_equal(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def resolve_session(openai_messages):
    """Returns (mode, claude_session_id, delta_openai_messages, conv_key).
    mode is "resume" (continuing a known conversation -- only send the new
    tail messages) or "fresh" (new conversation, or the history diverged
    from what we last synced -- send everything)."""
    key = _conversation_key(openai_messages)
    with _SESSION_LOCK:
        entry = _SESSION_STORE.get(key)
        if entry:
            synced = entry["synced_messages"]
            match = len(openai_messages) > len(synced) and _messages_equal(openai_messages[: len(synced)], synced)
            if match:
                delta = openai_messages[len(synced) :]
                return "resume", entry["claude_session_id"], delta, key
    return "fresh", str(uuid.uuid4()), openai_messages, key


def record_session(conv_key, claude_session_id, full_openai_messages, assistant_reply_message):
    with _SESSION_LOCK:
        synced = list(full_openai_messages) + [assistant_reply_message]
        _SESSION_STORE[conv_key] = {"claude_session_id": claude_session_id, "synced_messages": synced}
        while len(_SESSION_STORE) > _SESSION_STORE_MAX:
            oldest = next(iter(_SESSION_STORE))
            if oldest == conv_key:
                break
            _SESSION_STORE.pop(oldest, None)

