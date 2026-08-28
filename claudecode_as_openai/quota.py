#!/usr/bin/env python3
"""Quota/rate-limit state: snapshot logging plus the /v1/key and
/v1/credits response builders backed by the last rate_limit_event captured
from a claude subprocess."""

import sys

from claudecode_as_openai import state


def _log_quota_snapshot():
    """Log current quota state to stderr if available. Called after every
    completion to give operators visibility into quota burn rate and
    remaining headroom."""
    info = state._rate_limit_cache
    if not info:
        return
    status = info.get("status", "unknown")
    five_h = info.get("unifiedWindows", {}).get("five_hour", {})
    util = five_h.get("utilization", 0.0)
    resets_at = five_h.get("resetsAt")
    # Only log at WARNING/CRITICAL levels to reduce noise on normal operations
    if util >= 0.90:
        severity = "CRITICAL" if util >= 0.95 else "WARNING"
        sys.stderr.write(
            "claudecode-as-openai: quota_snapshot status=%s util_5h=%.1f%% "
            "severity=%s resets_at=%s\n" % (status, util * 100, severity, resets_at)
        )


def _build_key_response():
    """Return an OpenRouter-compatible /v1/key response backed by the last
    rate_limit_event captured from the claude subprocess.

    Uses the 7-day window as the primary quota because it is the limit
    most likely to constrain a day's work.  The 5-hour window is included
    in the `rate_limits` extension field so callers can surface it.

    Returns null values before the first turn completes (no data yet)."""
    info = state._rate_limit_cache
    if not info:
        return {
            "data": {
                "label": "claude-code-subscription",
                "limit": None,
                "limit_remaining": None,
                "is_free_tier": False,
            }
        }
    windows = info.get("unifiedWindows", {})
    five_h = windows.get("five_hour", {})
    seven_d = windows.get("seven_day", {})

    # Use the 5-hour window as the primary quota: it is the current/active
    # limit users hit first. The 7-day window is included in the extension
    # field for informational display.
    five_h_used = five_h.get("utilization", 0.0)
    limit = 100
    usage = round(five_h_used * limit, 2)
    limit_remaining = round(limit - usage, 2)

    return {
        "data": {
            "label": "claude-code-subscription",
            "limit": limit,
            "usage": usage,
            "limit_remaining": limit_remaining,
            "is_free_tier": False,
            "rate_limit_status": info.get("status"),
            "rate_limits": {
                "5h": {
                    "percent_used": round(five_h.get("utilization", 0.0) * 100),
                    "resets_at": five_h.get("resetsAt"),
                },
                "7d": {
                    "percent_used": round(seven_d.get("utilization", 0.0) * 100),
                    "resets_at": seven_d.get("resetsAt"),
                },
            },
        }
    }


def _build_credits_response():
    """Return an OpenRouter-compatible /v1/credits response.

    Maps the 5-hour (current/active) usage window to a 0-100 credit scale,
    consistent with /v1/key so that clients reading total_credits/total_usage
    get a coherent view."""
    info = state._rate_limit_cache
    if not info:
        return {"data": {"total_credits": None, "total_usage": None}}
    five_h = info.get("unifiedWindows", {}).get("five_hour", {})
    total_credits = 100
    total_usage = round(five_h.get("utilization", 0.0) * total_credits, 2)
    return {"data": {"total_credits": total_credits, "total_usage": total_usage}}
