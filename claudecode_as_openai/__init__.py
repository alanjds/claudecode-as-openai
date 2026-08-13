"""claudecode-as-openai: OpenAI-chat-completions-compatible HTTP shim over
the local Claude Code CLI.

See shim.py for the implementation and README.md for usage, the documented
tool-calling reliability tradeoffs, and the investigation trail for why a
bounded retry (not a deterministic mechanism) is the current approach.
"""
__version__ = '0.1.0'
