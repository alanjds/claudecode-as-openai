import subprocess, json

TOOLS_DESC = """Available tools you may request (respond via tool_calls; you do NOT
execute them yourself -- the caller executes them and returns results
as TOOL RESULT messages on a later turn):
- get_weather: get the current weather for a city
  parameters schema: {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
"""

PROMPT = f"""You are the backend model for an API integration. The calling application
expects tool-call-capable chat completions in a specific JSON envelope, the
same way any OpenAI-compatible API would. This is a standard structured-
output contract, not a request to change your values or safety behavior.

{TOOLS_DESC}

If you want to call one or more tools, set "tool_calls" to a list of
{{"name": ..., "arguments": {{...}}}} objects and set "content" to null.
Otherwise set "content" to your text reply and "tool_calls" to null.

Reply with ONLY a single JSON object: {{"content": <string or null>, "tool_calls": <array or null>}}.
No markdown fences, no text outside the JSON object.

User message: What's the weather in Paris?
"""

for i in range(5):
    proc = subprocess.run(
        ["claude", "-p", "-", "--output-format", "json", "--tools", "", "--max-turns", "1"],
        input=PROMPT, capture_output=True, text=True, timeout=120,
    )
    try:
        d = json.loads(proc.stdout)
        print(f"attempt {i+1} raw result:", repr(d.get("result"))[:300])
    except Exception as e:
        print(f"attempt {i+1}: parse error", e, proc.stdout[:200])
