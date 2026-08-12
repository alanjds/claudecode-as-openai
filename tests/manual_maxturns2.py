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

for i in range(5):
    proc = subprocess.run(
        ["claude", "-p", "-", "--output-format", "json", "--tools", "", "--max-turns", "2"],
        input=PROMPT, capture_output=True, text=True, timeout=120,
    )
    d = json.loads(proc.stdout)
    print(f"attempt {i+1}: subtype={d.get('subtype')} stop_reason={d.get('stop_reason')} is_error={d.get('is_error')} num_turns={d.get('num_turns')} result={d.get('result')!r}")
