import sys
sys.path.insert(0, "/var/home/alanjds/.hermes/hermes-agent")
from hermes_cli.runtime_provider import resolve_runtime_provider

try:
    result = resolve_runtime_provider(requested="copilot-acp")
    print("SUCCESS:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
