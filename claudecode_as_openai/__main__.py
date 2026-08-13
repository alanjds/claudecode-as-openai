"""Allows `python3 -m claudecode_as_openai` as a shorthand for
`python3 -m claudecode_as_openai.shim`."""
from .shim import main

if __name__ == "__main__":
    main()
