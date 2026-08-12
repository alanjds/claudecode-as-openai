import os
os.environ["HERMES_COPILOT_ACP_COMMAND"] = "claude-agent-acp"
os.environ["HERMES_COPILOT_ACP_ARGS"] = ""

import sys
sys.path.insert(0, "/var/home/alanjds/.hermes/hermes-agent")

from hermes_cli.runtime_provider import resolve_runtime_provider

# Simulate exactly what cli.py does: provider="copilot-acp", no explicit api_key/base_url
try:
    result = resolve_runtime_provider(
        requested="copilot-acp",
        explicit_api_key=None,
        explicit_base_url=None,
    )
    print("SUCCESS:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Exception type:", type(e))
