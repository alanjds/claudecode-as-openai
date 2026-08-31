#!/usr/bin/env python3
"""Session caching: fingerprint a conversation and track how much of it has
already been synced to a Claude Code session, so a continuing conversation
can send only the new trailing messages via --resume instead of the full
history every time. Verified live: a resumed turn re-processes only the
delta (a few hundred tokens) instead of the whole conversation."""

import hashlib
import json
import os
import re
import tempfile
import uuid

from claudecode_as_openai.state import _SESSION_STORE, _SESSION_STORE_MAX, _SESSION_LOCK
from claudecode_as_openai.tracking import logger


def _conversation_key(openai_messages):
    """Stable key for the conversation this message list belongs to: hash
    of the system messages + the first non-system message. Stays stable
    across the whole conversation even as more turns are appended."""
    sys_msgs = [m for m in openai_messages if m.get("role") == "system"]
    first_non_system = next((m for m in openai_messages if m.get("role") != "system"), None)
    basis = json.dumps([sys_msgs, first_non_system], sort_keys=True, default=str)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


_OOB_RE = re.compile(
    r"\[OUT-OF-BAND USER MESSAGE.*?\[/OUT-OF-BAND USER MESSAGE\]",
    re.DOTALL,
)


def _strip_oob(text):
    """Strip Hermes out-of-band user message blocks from a tool-result string.

    Hermes appends [OUT-OF-BAND USER MESSAGE ...][/OUT-OF-BAND USER MESSAGE]
    to tool-result content when a user sends a mid-turn message. The shim
    stores the message WITH this block, but on the next turn Hermes sends
    the clean tool result without it, causing a spurious divergence.
    """
    if not isinstance(text, str):
        return text
    return _OOB_RE.sub("", text).rstrip()


def _normalize_message(msg):
    """Return a copy of msg with fields normalized for comparison.

    Accepts either a single message dict or a list of message dicts
    (resolve_session compares a prefix slice as a list). Strips
    model-internal fields (reasoning, reasoning_details) that the client
    never sees and cannot round-trip back to us.  Normalizes content
    (null == "") and tool-call argument JSON formatting so minor
    serialization differences don't cause false divergence. Strips
    Hermes out-of-band user message blocks from tool-result content
    (injected mid-turn; absent on replay).
    """
    import copy
    if isinstance(msg, list):
        return [_normalize_message(m) for m in msg]
    m = copy.deepcopy(msg)
    m.pop("reasoning", None)
    m.pop("reasoning_details", None)
    if m.get("content") in (None, ""):
        m["content"] = None
    elif isinstance(m.get("content"), str):
        stripped = _strip_oob(m["content"])
        m["content"] = stripped if stripped else None
    for tc in m.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.dumps(json.loads(args), sort_keys=True, separators=(",", ":"))
            except (json.JSONDecodeError, TypeError):
                pass
    return m


def _messages_equal(a, b):
    return json.dumps(_normalize_message(a), sort_keys=True, default=str) == json.dumps(_normalize_message(b), sort_keys=True, default=str)


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
            length_ok = len(openai_messages) > len(synced)
            match = length_ok and _messages_equal(openai_messages[: len(synced)], synced)
            if match:
                logger.debug(
                    "resolve_session: conv_key=%s -> resume (n_incoming=%d n_synced=%d)",
                    key[:12], len(openai_messages), len(synced),
                )
                delta = openai_messages[len(synced) :]
                return "resume", entry["claude_session_id"], delta, key
            if not length_ok:
                logger.debug(
                    "resolve_session: conv_key=%s -> fresh (entry found, but incoming has %d "
                    "messages <= the %d already synced -- not a superset, treating as diverged)",
                    key[:12], len(openai_messages), len(synced),
                )
            else:
                first_diff = next(
                    (i for i in range(len(synced))
                     if not _messages_equal(openai_messages[i], synced[i])),
                    None,
                )
                logger.debug(
                    "resolve_session: conv_key=%s -> fresh (entry found, but prefix diverged "
                    "at message index %s of %d synced)",
                    key[:12], first_diff, len(synced),
                )
                if first_diff is not None and logger.isEnabledFor(10):  # 10 == logging.DEBUG
                    _dump_path = os.path.join(tempfile.gettempdir(), "claudecode_diverge_dump.json")
                    with open(_dump_path, "w") as _f:
                        json.dump(
                            {
                                "conv_key": key[:12],
                                "first_diff_index": first_diff,
                                "n_synced": len(synced),
                                "incoming": openai_messages[first_diff],
                                "synced": synced[first_diff],
                            },
                            _f, indent=2, default=str,
                        )
                    logger.debug("resolve_session: diverged message dump -> %s", _dump_path)
        else:
            logger.debug(
                "resolve_session: conv_key=%s -> fresh (no prior entry for this key; "
                "n_incoming=%d, %d known conv_keys)",
                key[:12], len(openai_messages), len(_SESSION_STORE),
            )
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

