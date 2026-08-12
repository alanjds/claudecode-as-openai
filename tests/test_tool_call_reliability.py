"""Reliability test for tool-call dispatch via the shim's HTTP endpoint.

Run this against a live shim instance (`python3 -m openai_claudecli_bridge.shim
8977`) to verify the bounded-retry wrapper actually improves real-world
tool_use success rate. See README.md / SKILL notes for the documented
baseline: ~40% single-shot native tool_use success without retries
(cline/cline#10336-style flakiness), ~100% observed with 3x bounded retry
in manual testing (small sample; rerun periodically to confirm it holds).

Usage:
    python3 tests/test_tool_call_reliability.py [--attempts N] [--port PORT]
"""
import argparse
import json
import sys
import time
import urllib.request

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def call(url, messages, tools=None, model="sonnet", timeout=90):
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run(port, attempts):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    successes = 0
    for i in range(1, attempts + 1):
        t0 = time.time()
        r = call(
            url,
            [{"role": "user", "content": "What's the weather in Paris? Use the get_weather tool."}],
            tools=[WEATHER_TOOL],
        )
        dt = time.time() - t0
        msg = r["choices"][0]["message"]
        has_tool_call = bool(msg.get("tool_calls"))
        successes += int(has_tool_call)
        print(
            f"attempt {i}/{attempts} [{dt:.1f}s] "
            f"tool_calls={msg.get('tool_calls')!r} "
            f"content={str(msg.get('content'))[:60]!r}"
        )
    print(f"\n{successes}/{attempts} produced a real tool_call "
          f"({100 * successes / attempts:.0f}%)")
    return successes, attempts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attempts", type=int, default=15,
                    help="Use >=15 to avoid small-sample noise (see SKILL notes)")
    p.add_argument("--port", type=int, default=8977)
    args = p.parse_args()
    successes, attempts = run(args.port, args.attempts)
    sys.exit(0 if successes == attempts else 1)


if __name__ == "__main__":
    main()
