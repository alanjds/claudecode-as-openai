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


# Hermes's --pass-session-id (CLI/TUI only today; not wired into the webui
# gateway backend -- verified against Hermes source) embeds this exact line
# in the system prompt (agent/system_prompt.py: `f"\nSession ID: {agent.session_id}"`,
# only when the flag is on). When present, it's a far more reliable
# conversation identity than hashing system+first message: it's immune to
# any incidental system-prompt content drift across turns (dates, live
# context, etc.) that would otherwise silently split one conversation into
# multiple fingerprints. Any client can adopt this by emitting the same
# line -- it's not Hermes-specific despite the origin.
_CLIENT_SESSION_ID_RE = re.compile(r"^Session ID: (\S+)$", re.MULTILINE)


def _extract_client_session_id(openai_messages):
    for m in openai_messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            match = _CLIENT_SESSION_ID_RE.search(m["content"])
            if match:
                return match.group(1)
    return None


def _conversation_key(openai_messages):
    """Stable key for the conversation this message list belongs to.

    Prefers a client-supplied "Session ID: <id>" line in the system prompt
    when present (see _extract_client_session_id) -- an explicit, stable
    identity token straight from the client. Falls back to a hash of the
    system messages + the first non-system message, which stays stable
    across the whole conversation even as more turns are appended, but can
    in principle be defeated by system-prompt content that drifts between
    turns of the same conversation."""
    client_sid = _extract_client_session_id(openai_messages)
    if client_sid:
        return f"client-sid:{client_sid}"
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

    Hermes's steer() appends this marker to the content of the LAST role
    "tool" message when a user sends a mid-turn message (verified against
    Hermes source: agent/prompt_builder.py STEER_MARKER_OPEN/CLOSE,
    agent_runtime_helpers.py apply_pending_steer_to_tool_results()). Once
    injected it is permanent, immutable conversation history on Hermes's
    side (Hermes's own tests assert this) -- the divergence this patches is
    a one-time timing gap: the shim cached synced_messages from before
    Hermes's in-process steer injection landed on that same message.

    Only meaningful on role=="tool" content -- see caller in
    _normalize_message. The identical marker text also appears, verbatim,
    in Hermes's system prompt (STEER_CHANNEL_NOTE, telling the model what
    the marker means) but that's static per-conversation and never a
    source of divergence, so it's intentionally left untouched here.
    """
    if not isinstance(text, str):
        return text
    return _OOB_RE.sub("", text).rstrip()


# Hermes's context compressor demotes an aging tail tool-result to a one-line
# stub once it falls out of the kept-tail window (agent/context_compressor.py
# _lean_recovery_stub()/_LEAN_TAIL_KEEP_TOOL_ROUNDS): the full content is
# still recoverable Hermes-side via session_search, but is permanently gone
# from what Hermes itself will ever resend.
#
# Unlike the OOB marker, this is NOT tolerated as a match. A resumed Claude
# session never gets Hermes's compacted history re-sent to it -- only the
# new delta -- so treating a full/stub pair as "equal" would mean Claude's
# own session keeps the full-size content forever and grows unbounded
# regardless of how much Hermes compacts on its own side; Hermes's
# compaction signal would never reach Claude's context at all. Falling
# through to the existing "fresh" behavior is what actually lets this
# shim's Claude session inherit Hermes's compaction: a fresh session's
# baseline IS full_openai_messages exactly as Hermes now sends it (stub
# included), so it starts back down at Hermes's own, smaller size instead
# of the old session's ever-growing one. That's a real cost (one full
# resend, no cache hit that turn) traded for bounded context growth in
# lockstep with Hermes -- the entire point of compaction existing.
#
# _is_lean_recovery_stub is kept only to make the DEBUG divergence log (see
# resolve_session) say WHICH kind of fresh this is -- expected compaction
# fallout vs. a genuine unexpected divergence worth investigating.
_LEAN_STUB_RE = re.compile(
    r"^\[.+ output demoted at compaction — [\d,]+ chars preserved in "
    r"session history\.(?: Recover with session_search\(query=\.\.\., "
    r"session_id='[^']*'\))?\]$"
)


def _is_lean_recovery_stub(content):
    return isinstance(content, str) and bool(_LEAN_STUB_RE.match(content.strip()))


def _normalize_message(msg, ignore_system_content=False):
    """Return a copy of msg with fields normalized for comparison.

    Accepts either a single message dict or a list of message dicts
    (resolve_session compares a prefix slice as a list). Strips
    model-internal fields (reasoning, reasoning_details) that the client
    never sees and cannot round-trip back to us.  Normalizes content
    (null == "") and tool-call argument JSON formatting so minor
    serialization differences don't cause false divergence. Strips
    Hermes out-of-band user message blocks from tool-result content
    (injected mid-turn; absent on replay).

    ignore_system_content: when True, system-role content is blanked out
    entirely before comparison. Only ever passed True when the conversation
    was keyed by an explicit client-supplied session id (see
    _conversation_key) -- that's a trust boundary the client itself drew:
    it told us "this is the same session", so a differing system prompt
    (a live timestamp/model/provider line, typically) is volatile metadata,
    not evidence of divergence. Never applied otherwise.
    """
    import copy
    if isinstance(msg, list):
        return [_normalize_message(m, ignore_system_content) for m in msg]
    m = copy.deepcopy(msg)
    m.pop("reasoning", None)
    m.pop("reasoning_details", None)
    if ignore_system_content and m.get("role") == "system":
        m["content"] = None
    elif m.get("content") in (None, ""):
        m["content"] = None
    elif isinstance(m.get("content"), str):
        content = m["content"]
        if m.get("role") == "tool":
            content = _strip_oob(content)
        m["content"] = content if content else None
    for tc in m.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                fn["arguments"] = json.dumps(json.loads(args), sort_keys=True, separators=(",", ":"))
            except (json.JSONDecodeError, TypeError):
                pass
    return m


def _messages_equal(a, b, ignore_system_content=False):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _messages_equal(x, y, ignore_system_content) for x, y in zip(a, b)
        )
    na, nb = _normalize_message(a, ignore_system_content), _normalize_message(b, ignore_system_content)
    return json.dumps(na, sort_keys=True, default=str) == json.dumps(nb, sort_keys=True, default=str)


def resolve_session(openai_messages):
    """Returns (mode, claude_session_id, delta_openai_messages, conv_key).
    mode is "resume" (continuing a known conversation -- only send the new
    tail messages) or "fresh" (new conversation, or the history diverged
    from what we last synced -- send everything)."""
    key = _conversation_key(openai_messages)
    trust_identity = key.startswith("client-sid:")
    logger.debug(
        "resolve_session: conv_key=%s keyed by %s",
        key[:12],
        "client-supplied Session ID line (system-prompt drift ignored)"
        if trust_identity else "hash of system+first message",
    )
    with _SESSION_LOCK:
        entry = _SESSION_STORE.get(key)
        if entry:
            synced = entry["synced_messages"]
            length_ok = len(openai_messages) > len(synced)
            match = length_ok and _messages_equal(
                openai_messages[: len(synced)], synced, ignore_system_content=trust_identity
            )
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
                     if not _messages_equal(openai_messages[i], synced[i], ignore_system_content=trust_identity)),
                    None,
                )
                compaction_triggered = first_diff is not None and (
                    _is_lean_recovery_stub(openai_messages[first_diff].get("content"))
                    or _is_lean_recovery_stub(synced[first_diff].get("content"))
                )
                logger.debug(
                    "resolve_session: conv_key=%s -> fresh (entry found, but prefix diverged "
                    "at message index %s of %d synced%s)",
                    key[:12], first_diff, len(synced),
                    " -- looks like Hermes context-compaction demotion, expected fallout, not a bug"
                    if compaction_triggered else "",
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

