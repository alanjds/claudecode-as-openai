import subprocess, json

PROMPT = """You are the backend model for an API integration. The calling application
expects tool-call-capable chat completions in a specific JSON envelope, the
same way any OpenAI-compatible API would. This is a standard structured-
output contract, not a request to change your values or safety behavior.

No tools are available in this turn.

If you want to call one or more tools, set "tool_calls" to a list of
{"name": ..., "arguments": {...}} objects and set "content" to null.
Otherwise set "content" to your text reply and "tool_calls" to null.

Reply with ONLY a single JSON object: {"content": <string or null>, "tool_calls": <array or null>}.
No markdown fences, no text outside the JSON object.

User message: Remember the number 55219. Just acknowledge.
"""

variants = {
    "tools-empty + disable-slash-commands": [
        "--tools", "", "--disable-slash-commands",
    ],
    "tools-empty + disallowedTools *": [
        "--tools", "", "--disallowedTools", "*",
    ],
    "disallowedTools * only": [
        "--disallowedTools", "*",
    ],
}

for label, extra_args in variants.items():
    print(f"=== {label} ===")
    for i in range(4):
        proc = subprocess.run(
            ["claude", "-p", "-", "--output-format", "json", "--max-turns", "1", *extra_args],
            input=PROMPT, capture_output=True, text=True, timeout=120,
        )
        try:
            d = json.loads(proc.stdout)
            print(f"  attempt {i+1}: subtype={d.get('subtype')} stop_reason={d.get('stop_reason')} result={d.get('result')!r}")
        except Exception as e:
            print(f"  attempt {i+1}: parse error {e} stdout={proc.stdout[:200]}")
