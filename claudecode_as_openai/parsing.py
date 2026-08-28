#!/usr/bin/env python3
"""NDJSON streaming line parsers for Claude Code subprocess output."""

import json
import os
import select
import time


def _parse_ndjson_line(line):
    """Shared tail logic for both NDJSON readers below: strip, skip blank,
    parse JSON, skip silently on a decode error (a malformed/partial line
    is not fatal -- just not a usable chunk). Returns the parsed dict, or
    None if the line should be skipped."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _iter_ndjson_lines(proc):
    for raw_line in proc.stdout:
        parsed = _parse_ndjson_line(raw_line)
        if parsed is not None:
            yield parsed


def _iter_ndjson_lines_pty(master_fd, proc, timeout_s):
    """Same contract as _iter_ndjson_lines, but reads from a PTY master fd
    instead of a plain subprocess.PIPE.

    Why this exists: `claude`'s own stdout is FULLY BUFFERED (not just
    line-buffered) when its stdout is a plain pipe with no TTY attached --
    verified live: with --include-partial-messages, real per-token
    content_block_delta events all arrive in 2-3 giant bursts within the
    same ~20ms window regardless of the actual generation taking many
    seconds, because glibc's stdio only flushes on a full buffer or
    process exit when stdout isn't a terminal. Attaching a real PTY as
    `claude`'s stdout makes it use LINE buffering like an interactive
    terminal session -- verified live: the same request then delivers
    dozens of individual deltas spread realistically across the full
    generation time (e.g. 21 deltas over ~22s of a 300-word story,
    roughly one every 0.5-0.7s, matching genuine token-generation pacing).
    This is the only way to get real client-facing streaming out of this
    CLI; there is no flag to force unbuffered/line-buffered stdout.

    `timeout_s`, if given, is a single deadline for the WHOLE generator
    lifetime, not per-line -- correct for call_claude_streaming's
    one-shot use (one call, one bounded wait), but WRONG for a
    WarmProcess's background reader, which must survive indefinitely
    across an unbounded number of turns with idle gaps between them.
    Pass timeout_s=None for that case: the generator then only returns
    on EOF/process-exit, with no wall-clock cutoff at all (see
    WarmProcess._read_loop)."""
    deadline = None if timeout_s is None else time.time() + timeout_s
    buf = b""
    while True:
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            select_timeout = min(remaining, 1.0)
        else:
            select_timeout = 1.0
        try:
            ready, _, _ = select.select([master_fd], [], [], select_timeout)
        except (OSError, ValueError):
            return
        if not ready:
            if proc.poll() is not None:
                # Process exited and nothing left to read.
                try:
                    ready2, _, _ = select.select([master_fd], [], [], 0)
                except (OSError, ValueError):
                    ready2 = []
                if not ready2:
                    return
            continue
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            # EIO is the normal "slave side closed" signal on Linux PTYs.
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            raw_line, buf = buf.split(b"\n", 1)
            parsed = _parse_ndjson_line(raw_line.decode("utf-8", errors="replace"))
            if parsed is not None:
                yield parsed


