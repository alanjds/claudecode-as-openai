"""Smoke test for plain chat (no tools) via the shim's HTTP endpoint.
Confirms the non-tool path stays fast and doesn't trigger retry logic."""
import json
import time
import urllib.request

def call(url, messages, timeout=60):
    payload = {"model": "sonnet", "messages": messages, "stream": False}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main(port=8977):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    t0 = time.time()
    r = call(url, [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Remember the number 55219. Just acknowledge."},
    ])
    dt = time.time() - t0
    msg = r["choices"][0]["message"]
    print(f"[{dt:.1f}s] content={msg.get('content')!r}")
    assert msg.get("tool_calls") is None, "plain chat should never produce tool_calls"
    assert "55219" in (msg.get("content") or ""), "response should echo back the number"
    print("OK")


if __name__ == "__main__":
    main()
