"""Shared mutable state across claudecode-as-openai modules.

This module holds all global state that multiple modules need to access.
Separation of state from logic prevents circular imports.
"""

import threading
import tempfile

# Session persistence cache
_SESSIONS_DB = {}
_SESSIONS_DB_LOCK = threading.Lock()

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
    """Initialize global state (call once at startup)."""
    global _CLAUDE_CWD
    _CLAUDE_CWD = tempfile.mkdtemp(prefix="claude_")
