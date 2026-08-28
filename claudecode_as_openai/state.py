"""Shared mutable state across claudecode-as-openai modules.

This module holds all global state that multiple modules need to access.
Separation of state from logic prevents circular imports.
"""

import threading
import tempfile

# Session persistence cache
_SESSION_STORE = {}  # conv_key -> {"claude_session_id": str, "synced_messages": list}
_SESSION_STORE_MAX = 200
_SESSION_LOCK = threading.Lock()

# Rate limit tracking
_rate_limit_cache = None
_rate_limit_cache_lock = threading.Lock()

# Claude subprocess environment
_CLAUDE_CWD = None
_CLAUDE_CWD_LOCK = threading.Lock()

# Warm process pool singleton
_WARM_POOL = None
_WARM_POOL_LOCK = threading.Lock()

def initialize():
    """Initialize global state (call once at startup, before any request is
    served). Idempotent -- a second call is a no-op so importing http.py
    more than once in the same process (e.g. across test modules) doesn't
    spawn a second warm pool or sandbox directory."""
    global _CLAUDE_CWD, _WARM_POOL
    if _CLAUDE_CWD is None:
        # Every spawned `claude` subprocess inherits whatever directory the
        # shim process happens to be running from, and Claude Code can read
        # real files there. Spawning every `claude` call from a dedicated,
        # empty, per-run temp directory closes this off.
        _CLAUDE_CWD = tempfile.mkdtemp(prefix="claudecode-as-openai-sandbox-")
    if _WARM_POOL is None:
        # Deferred import: warm_pool.py imports several names from this
        # module at its own top level, so importing it back at state.py's
        # module level would be circular. By the time initialize() is
        # actually called (from http.py, after the whole module graph has
        # finished loading), this resolves without issue.
        from claudecode_as_openai.warm_pool import WarmPool
        _WARM_POOL = WarmPool()
