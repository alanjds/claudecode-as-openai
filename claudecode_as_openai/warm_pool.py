#!/usr/bin/env python3
"""Persistent warm-pool: keeps ONE already-spawned, already-bootstrapped
`claude -p --input-format stream-json` process parked per conversation,
ready to take the NEXT turn without paying the ~5s process-spawn/
bootstrap cost a fresh `-p --resume` invocation pays on every call
(measured live: cold `-p --resume` wall time vs the CLI's own
self-reported duration_ms showed a ~5.2s unaccounted gap -- pure
process-lifecycle overhead outside the API call itself -- that
collapses to ~5ms once a process is already resident and warm).

Design, per explicit user direction: at most ONE parked (idle, already
spawned) process at a time, never a process per concurrent conversation.
A "new" conversation here means "different from the immediately
preceding one" -- there is no attempt to keep N conversations warm
simultaneously.

  1. A turn for a brand-new conversation arrives -> served cold (no
     warm process can exist for a conversation that didn't exist yet).
     In parallel, once that reply's real Claude session_id is known,
     spawn a WarmProcess pre-resuming that exact session, parked
     waiting for turn 2.
  2. The next turn for the SAME conversation (matching fingerprint +
     session_id) arrives -> claim the parked WarmProcess, feed it the
     turn directly (no spawn), and spawn a fresh WarmProcess to park
     for turn 3 once this reply is known.
  3. A turn for a DIFFERENT conversation arrives (new fingerprint, or a
     continuation whose synced history no longer matches this parked
     process's session) -> the stale parked process is useless (its
     --resume target is for the wrong conversation) and is killed
     immediately; that turn is served cold, and a new WarmProcess is
     parked for whatever comes next.

Cross-compatible with the existing cold path by construction: a warm
process's Claude session_id is a completely normal Claude Code session
(created with plain --session-id / --resume, just kept alive across
turns via --input-format stream-json instead of exiting after one).
Verified live: a session created and advanced by a WarmProcess resumes
correctly via a totally separate one-shot `-p --resume` call after the
warm process is killed, and vice versa -- either side can pick up
where the other left off with no special handling needed."""

import json
import os
import pty
import queue
import subprocess
import sys
import tempfile
import threading
import time

from claudecode_as_openai.constants import CLAUDE_TIMEOUT_S
from claudecode_as_openai.errors import ClaudeCliError
from claudecode_as_openai import state


class WarmProcess:
    """One already-spawned `claude -p --input-format stream-json
    --output-format stream-json` subprocess, resumed onto a specific
    Claude session_id, parked waiting to serve exactly one more turn of
    that same conversation. Not reused after that turn is served --
    callers spawn a fresh WarmProcess for the turn after this one (see
    WarmPool)."""

    def __init__(self, fingerprint, session_id, model, system_prompt, tools_requested, tools,
                 mcp_tool_config, effort, use_pty, env_overrides=None):
        self.fingerprint = fingerprint
        self.session_id = session_id
        self.model = model
        self.system_prompt = system_prompt
        self.tools_requested = tools_requested
        self.tools = tools
        self.mcp_tool_config = mcp_tool_config
        self.effort = effort
        self.use_pty = use_pty
        self.env_overrides = env_overrides
        self.proc = None
        self.pty_master_fd = None
        self.system_prompt_file = None
        self._reader_thread = None
        self._line_queue = queue.Queue()
        self._lock = threading.Lock()
        self._claimed = False
        self.spawned_at = time.time()  # when this process was spawned
        self.parked_at = None          # set by WarmPool.park(); when it was parked idle
        self._start()

    def _start(self):
        # Deferred import: streaming.py imports WarmPool from this module
        # at its own top level, so importing it back here at module level
        # would be circular. By the time a WarmProcess is actually
        # constructed (well after both modules have finished loading),
        # this resolves without issue.
        from claudecode_as_openai.streaming import _build_claude_cmd

        # Warm pool always resumes a known session; json_schema=None here
        # (json_schema requires fresh spawns with full orchestration). With
        # prose fallback removed, max_turns=1 is always safe.
        max_turns = 1
        cmd = _build_claude_cmd(
            self.model, "resume", self.session_id, self.tools_requested, max_turns,
            json_schema=None, want_partial_messages=self.use_pty,
            mcp_tool_config=self.mcp_tool_config, effort=self.effort,
            input_format="stream-json",
        )
        if self.system_prompt:
            # Matches call_claude_streaming's own >4000-char threshold
            # for --system-prompt-file vs --system-prompt (see there for
            # why: argv length limits). A resumed session already has
            # its system prompt recorded server-side, so this is
            # belt-and-suspenders consistency with the cold path rather
            # than something a resume strictly needs.
            if len(self.system_prompt) > 4000:
                fd, self.system_prompt_file = tempfile.mkstemp(
                    prefix="claudecode-as-openai-sysprompt-", suffix=".txt",
                )
                with os.fdopen(fd, "w") as f:
                    f.write(self.system_prompt)
                cmd += ["--system-prompt-file", self.system_prompt_file]
            else:
                cmd += ["--system-prompt", self.system_prompt]
        # A warm process's env is fixed for its whole parked lifetime
        # (unlike the cold path's _scoped_env_overrides, which only
        # needs to patch subprocess.Popen for the duration of one
        # request's call stack) -- passed directly at spawn time here
        # instead. Without this, CLAUDE_CODE_MAX_OUTPUT_TOKENS and
        # CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC (see _BASE_ENV_OVERRIDES)
        # would silently NOT apply to any turn served by a warm process.
        spawn_env = dict(os.environ)
        if self.env_overrides:
            spawn_env.update(self.env_overrides)
        if self.use_pty:
            self.pty_master_fd, pty_slave_fd = pty.openpty()
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=pty_slave_fd,
                stderr=subprocess.PIPE, cwd=state._CLAUDE_CWD, env=spawn_env,
            )
            os.close(pty_slave_fd)
        else:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, cwd=state._CLAUDE_CWD, env=spawn_env,
            )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        # Runs for the whole process lifetime, pushing every parsed
        # NDJSON line onto a queue that send_turn drains per-turn. A
        # background reader (rather than reading synchronously inside
        # send_turn) is what lets the process sit parked indefinitely
        # between turns without a blocking read call pinning a thread
        # doing nothing useful -- this thread blocks on I/O, which is
        # cheap, not on CPU. timeout_s=None on the PTY path is
        # deliberate: _iter_ndjson_lines_pty's deadline is per-
        # generator-lifetime, not per-line, and this generator IS the
        # process's whole lifetime (spawn through however many turns get
        # served) -- a finite deadline here would silently kill an
        # idle-but-still-alive parked process out from under send_turn.
        # Per-turn bounding happens instead in send_turn's own
        # queue.get(timeout=...).
        from claudecode_as_openai.parsing import _iter_ndjson_lines, _iter_ndjson_lines_pty

        try:
            if self.use_pty:
                for parsed in _iter_ndjson_lines_pty(self.pty_master_fd, self.proc, timeout_s=None):
                    self._line_queue.put(parsed)
            else:
                for parsed in _iter_ndjson_lines(self.proc):
                    self._line_queue.put(parsed)
        except Exception:
            pass
        finally:
            self._line_queue.put(None)  # sentinel: stdout closed / process exited

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def claim(self):
        """Returns True exactly once -- the first caller to claim this
        process for a turn. Guards against a race between the pool
        handing out this same parked process to two concurrent
        requests."""
        with self._lock:
            if self._claimed or not self.is_alive():
                return False
            self._claimed = True
            return True

    def send_turn(self, claude_messages, stop=None, stream_callback=None):
        """Feed ONE turn (a list of new Claude-shaped messages -- the
        delta beyond what this session already has) to the already-
        running process and block for its result. Must only be called
        after claim() returns True. Uses the exact same response-parsing
        logic as the cold path (_consume_claude_response) so behavior is
        identical either way."""
        from claudecode_as_openai.streaming import _consume_claude_response, _normalize_stop_sequences

        stop_sequences = _normalize_stop_sequences(stop)
        # claude_messages is the delta list this shim would otherwise
        # have written whole to a fresh process's stdin; stream-json
        # input takes one JSON object per line instead, so each element
        # is sent as its own {"type": "user", "message": ...} frame.
        for msg in claude_messages:
            line = json.dumps({"type": msg.get("role", "user"), "message": msg})
            if self.use_pty:
                os.write(self.proc.stdin.fileno(), (line + "\n").encode())
            else:
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()

        def chunk_source():
            while True:
                try:
                    item = self._line_queue.get(timeout=CLAUDE_TIMEOUT_S)
                except queue.Empty:
                    # No output from the warm process for CLAUDE_TIMEOUT_S
                    # seconds -- it is stuck. Kill it so quota stops burning
                    # and surface a proper error to the caller.
                    elapsed = time.time() - self.spawned_at
                    sys.stderr.write(
                        f"claudecode-as-openai: warm-pool: subprocess"
                        f" {self.session_id} produced no output for"
                        f" {CLAUDE_TIMEOUT_S}s (total age {elapsed:.0f}s);"
                        f" killing\n"
                    )
                    self.kill()
                    raise ClaudeCliError(
                        500, "api_error",
                        f"Warm-pool subprocess timed out after"
                        f" {CLAUDE_TIMEOUT_S}s with no output.",
                        code="warm_process_timeout",
                    )
                if item is None:
                    return
                yield item

        deadline = time.time() + CLAUDE_TIMEOUT_S
        return _consume_claude_response(chunk_source(), deadline, stop_sequences, stream_callback)

    def kill(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        if self.use_pty and self.pty_master_fd is not None:
            try:
                os.close(self.pty_master_fd)
            except OSError:
                pass
        if self.system_prompt_file:
            try:
                os.unlink(self.system_prompt_file)
            except OSError:
                pass
        if self.mcp_tool_config is not None:
            try:
                os.unlink(self.mcp_tool_config["manifest_path"])
            except OSError:
                pass


class WarmPool:
    """Holds at most one parked WarmProcess at a time (see module-level
    comment above for the full protocol). Thread-safe: HTTP requests are
    served from a ThreadingHTTPServer, so claiming/replacing the parked
    process must be atomic against a concurrent request doing the same."""

    # A parked process that hasn't been claimed within this window is
    # evicted: any gap longer than this means the conversation probably
    # ended or moved on, so the next request will bring a different
    # fingerprint and kill it anyway. Killing proactively here prevents
    # the process from sitting alive indefinitely if the user never
    # returns -- which is exactly what caused the quota-burndown incident
    # (a parked --resume process kept retrying an invalid session for hours).
    _PARK_IDLE_TIMEOUT_S = 300   # 5 minutes idle (parked, unclaimed)
    # Hard upper bound on a process's entire lifetime regardless of idle
    # state: caps the worst-case quota burn from a stuck parked subprocess.
    _LIFETIME_TIMEOUT_S = 600    # 10 minutes total

    def __init__(self):
        self._lock = threading.Lock()
        self._parked = None  # WarmProcess | None
        self._reaper_thread = None  # Background thread that evicts stale processes

    def take_if_matching(self, fingerprint):
        """Returns the parked WarmProcess if it exists, is alive, not
        stale (idle or lifetime timeout -- see class constants), and
        matches `fingerprint` (same conversation), claiming it
        atomically so no other request can also take it. Returns None
        otherwise -- including when a parked process exists but is for
        a DIFFERENT conversation, in which case it's killed here (a
        stale parked process is useless once the conversation moves on,
        per the "at most one parked process, discarded on mismatch"
        design)."""
        with self._lock:
            candidate = self._parked
            if candidate is None:
                return None
            now = time.time()
            # Kill stale processes before even checking fingerprint.
            # Idle check: parked too long without being claimed.
            if (candidate.parked_at is not None
                    and now - candidate.parked_at > self._PARK_IDLE_TIMEOUT_S):
                self._parked = None
                idle_s = now - candidate.parked_at
                sys.stderr.write(
                    f"claudecode-as-openai: warm-pool: evicting"
                    f" {candidate.session_id} -- idle {idle_s:.0f}s"
                    f" > {self._PARK_IDLE_TIMEOUT_S}s limit\n"
                )
                candidate.kill()
                return None
            # Lifetime check: total age since spawn.
            if now - candidate.spawned_at > self._LIFETIME_TIMEOUT_S:
                self._parked = None
                age_s = now - candidate.spawned_at
                sys.stderr.write(
                    f"claudecode-as-openai: warm-pool: evicting"
                    f" {candidate.session_id} -- lifetime {age_s:.0f}s"
                    f" > {self._LIFETIME_TIMEOUT_S}s limit\n"
                )
                candidate.kill()
                return None
            if candidate.fingerprint != fingerprint:
                self._parked = None
                candidate.kill()
                return None
            self._parked = None
        if not candidate.claim():
            return None
        return candidate

    def park(self, warm_process):
        """Installs `warm_process` as the new parked process, killing
        whatever was parked before it (there is only ever one). Also
        checks if the process is already stale (shouldn't happen in normal
        flow, but defensive against race conditions) and starts the reaper
        thread if it's not already running."""
        with self._lock:
            stale = self._parked
            now = time.time()
            # Defensive check: if somehow the new process is already stale,
            # don't park it, just kill it immediately.
            if (now - warm_process.spawned_at > self._LIFETIME_TIMEOUT_S
                    or (warm_process.parked_at is not None
                        and now - warm_process.parked_at > self._PARK_IDLE_TIMEOUT_S)):
                sys.stderr.write(
                    f"claudecode-as-openai: warm-pool: not parking {warm_process.session_id}"
                    f" -- already stale on arrival\n"
                )
                warm_process.kill()
                return
            self._parked = warm_process
            warm_process.parked_at = now
            # Start the reaper thread if it's not already running
            if self._reaper_thread is None or not self._reaper_thread.is_alive():
                self._reaper_thread = threading.Thread(target=self._reap_loop, daemon=True)
                self._reaper_thread.start()
        if stale is not None:
            stale.kill()

    def discard(self, fingerprint):
        """Kills and clears the parked process if it matches
        `fingerprint`. Used when a conversation ends up not continuing
        the way the parked process assumed (e.g. the caller's next
        request diverged before ever reaching take_if_matching)."""
        with self._lock:
            candidate = self._parked
            if candidate is None or candidate.fingerprint != fingerprint:
                return
            self._parked = None
        candidate.kill()

    def _reap_loop(self):
        """Background thread that periodically evicts stale parked processes.
        Runs once the first process is parked, sleeps 60 seconds between checks.
        Exits cleanly if _parked becomes None."""
        while True:
            time.sleep(60)  # Check every 60 seconds for stale processes
            with self._lock:
                candidate = self._parked
                if candidate is None:
                    # No process parked; reaper can exit
                    return
                now = time.time()
                # Check if stale
                if now - candidate.spawned_at > self._LIFETIME_TIMEOUT_S:
                    age = now - candidate.spawned_at
                    sys.stderr.write(
                        f"claudecode-as-openai: warm-pool: reaper evicting"
                        f" {candidate.session_id} -- lifetime {age:.0f}s"
                        f" > {self._LIFETIME_TIMEOUT_S}s limit\n"
                    )
                    self._parked = None
                    candidate.kill()
                    return
                elif (candidate.parked_at is not None
                        and now - candidate.parked_at > self._PARK_IDLE_TIMEOUT_S):
                    idle = now - candidate.parked_at
                    sys.stderr.write(
                        f"claudecode-as-openai: warm-pool: reaper evicting"
                        f" {candidate.session_id} -- idle {idle:.0f}s"
                        f" > {self._PARK_IDLE_TIMEOUT_S}s limit\n"
                    )
                    self._parked = None
                    candidate.kill()
                    return


def _parking_fingerprint(conv_key, model, tools_requested, tools, effort, env_overrides=None):
    """What must stay IDENTICAL between the turn a WarmProcess was
    parked for and the turn that claims it. Deliberately includes
    everything _build_claude_cmd branches on for this call shape
    (model/tools/effort change the spawned command line) plus the
    session-cache conv_key (a different conv_key is, by definition, a
    different conversation) and env_overrides (max_tokens changes
    CLAUDE_CODE_MAX_OUTPUT_TOKENS, which is baked into a warm process's
    environment at spawn time -- see WarmProcess._start) -- any mismatch
    here means the parked process's command line, environment, or
    --resume target are wrong for the incoming turn, so per the user's
    explicit design it must be discarded and respawned, never reused
    with different parameters."""
    tools_key = None
    if tools:
        try:
            tools_key = json.dumps(tools, sort_keys=True, default=str)
        except Exception:
            tools_key = str(tools)
    env_key = json.dumps(env_overrides, sort_keys=True) if env_overrides else None
    return (conv_key, model, bool(tools_requested), tools_key, effort, env_key)
