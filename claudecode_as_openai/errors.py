#!/usr/bin/env python3
"""Error types and classification for the Claude Code OpenAI shim."""

class ClaudeCliError(Exception):
    """Carries enough info to build an OpenAI-shaped error response.
    http_status: int: e.g. 400, 401, 404, 429, 500, 503.
    error_type: OpenAI error taxonomy string, e.g. "invalid_request_error",
        "authentication_error", "rate_limit_error", "api_error".
    code: short machine-readable code, e.g. "model_not_found", or None.
    """

    def __init__(self, http_status, error_type, message, code=None, param=None):
        super().__init__(message)
        self.http_status = http_status
        self.error_type = error_type
        self.message = message
        self.code = code
        self.param = param

    def to_openai_body(self):
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


# Heuristic classification of Claude Code's own plain-text error messages
# into OpenAI's error taxonomy. This is necessarily fragile -- it string-
# matches phrases from https://code.claude.com/docs/en/errors (checked
# 2026-08-13) -- and will need updating if Anthropic changes their error
# wording. Order matters: more specific patterns should come first.
_ERROR_PATTERNS = [
    (
        (
            "not logged in", "please run /login", "login expired",
            "oauth token", "invalid api key", "could not resolve authentication",
            "invalid auth token", "authentication credentials",
            "organization has disabled api key authentication",
            "organization has disabled claude subscription access",
        ),
        401, "authentication_error", "authentication_failed",
    ),
    (
        (
            "session limit", "weekly limit", "credit balance is too low",
            "spend limit", "request rejected (429)",
            "server is temporarily limiting requests", "rate limit",
        ),
        429, "rate_limit_error", "rate_limit_exceeded",
    ),
    (
        (
            "issue with the selected model", "not a recognized model id",
            "restricted by your organization", "not available with the claude",
        ),
        404, "invalid_request_error", "model_not_found",
    ),
    (
        (
            "prompt is too long", "context exceeds", "request too large",
            "conversation too long", "extra inputs are not permitted",
        ),
        400, "invalid_request_error", "context_length_exceeded",
    ),
    (
        ("overloaded", "internal server error", "500 internal"),
        503, "api_error", "overloaded",
    ),
]


def _classify_error_text(text):
    lower = (text or "").lower()
    for phrases, status, etype, code in _ERROR_PATTERNS:
        if any(p in lower for p in phrases):
            return status, etype, code
    return 500, "api_error", None

