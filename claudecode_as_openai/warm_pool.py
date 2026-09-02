#!/usr/bin/env python3
"""Persistent warm-pool: keeps ONE already-spawned `claude -p --resume`
process parked per conversation, its stdin pipe open but not yet written
to, ready to take the NEXT turn without paying the ~5s process-spawn/
bootstrap cost a fresh `-p --resume` invocation pays on every call
(measured live: cold `-p --resume` wall time vs the CLI's own
self-reported duration_ms showed a ~5.2s unaccounted gap -- pure
process-lifecycle overhead outside the API call itself).

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
     session_id) arrives -> claim the parked WarmProcess, write this
     turn's message array to its already-open stdin and close it (no
     spawn), and spawn a fresh WarmProcess to park for turn 3 once this
     reply is known.
  3. A turn for a DIFFERENT conversation arrives (new fingerprint, or a
     continuation whose synced history no longer matches this parked
     process's session) -> the stale parked process is useless (its
     --resume target is for the wrong conversation) and is killed
     immediately; that turn is served cold, and a new WarmProcess is
     parked for whatever comes next.

Each WarmProcess serves exactly one turn, ever, then is killed -- this
was already true even in the original --input-format stream-json design
(see git history), so it was never actually using stream-json's real
capability of feeding a SECOND turn into an already-running process. It
only ever needed "spawn now, deliver the one turn's input whenever it's
ready later", which a completely ordinary `-p --resume` process supports
just by holding its stdin pipe open, unwritten, until claim time -- no
special input mode required.

This replaced an earlier --input-format stream-json implementation after
a live investigation found that input mode itself (independent of
timing, independent of --replay-user-messages, independent of whether a
process was actually pre-parked or freshly spawned) reliably causes
Claude Code to re-issue a tool call it already made when resuming a
session that ended on that call -- see CHANGELOG for the full experiment
trail. Plain `-p --resume` with a single JSON-array stdin write (exactly
what the cold path already does, and now what this module does too, just
against a pre-spawned process) showed zero such regressions across every
trial run against it. A live timing experiment additionally confirmed
holding stdin open, unwritten, for ~2s before writing measurably hides
Claude Code's own session-reload/bootstrap latency (roughly halving the
time from write to first output) -- so this design keeps the warm pool's
whole reason to exist while dropping the one mechanism proven to cause
redundant tool calls.

Cross-compatible with the existing cold path by construction: a warm
process's Claude session_id is a completely normal Claude Code session
(created with plain --session-id / --resume). Verified live: a session
created and advanced by a WarmProcess resumes correctly via a totally
separate one-shot `-p --resume` call after the warm process is killed,
and vice versa -- either side can pick up where the other left off with
no special handling needed."""

import json
import os
import pty
import subprocess
import sys
import tempfile
import threading
import time

from claudecode_as_openai.constants import CLAUDE_TIMEOUT_S
from claudecode_as_openai import state
from claudecode_as_openai import tracking
from claudecode_as_openai.tracking import logger


class WarmProcess:
    """One already-spawned `claude -p --resume` subprocess for a specific
    Claude session_id, its stdin pipe open but not yet written to, parked
    waiting to serve exactly one turn of that same conversation. Not
    reused after that turn is served -- callers spawn a fresh WarmProcess
    for the turn after this one (see WarmPool).

    Deliberately NOT --input-format stream-json (see the module
    docstring for why): this is a completely ordinary one-shot `-p`
    process, identical to what the cold path spawns, except its stdin
    write is deferred from spawn time to claim time. Session-reload/
    bootstrap work Claude Code does independent of stdin content (e.g.
    reading --resume's target session) proceeds in the background while
    it's parked; only the response-generation work waits for the
    eventual write. Verified live: holding stdin open ~2s before writing
    roughly halves the time from write to first output, versus writing
    immediately after spawn."""

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
        self._debug_cmd = None
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
        # prose fallback removed, max_turns=1 is always safe. No
        # input_format passed -- see the module docstring for why this
        # must stay a plain one-shot process, never stream-json.
        max_turns = 1
        cmd = _build_claude_cmd(
            self.model, "resume", self.session_id, self.tools_requested, max_turns,
            json_schema=None, want_partial_messages=self.use_pty,
            mcp_tool_config=self.mcp_tool_config, effort=self.effort,
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
        # Captured once here (not rebuilt in send_turn, which never touches
        # argv again -- a warm process is spawned once and then only fed
        # its one turn over stdin) so send_turn's leaf span can still
        # report the command line this process was started with.
        self._debug_cmd = tracking.redact_cmd_for_log(cmd)
        logger.debug(
            "spawning claude (warm-pool init): %s | session_id=%s",
            self._debug_cmd, self.session_id,
        )
        # stdin is deliberately left unwritten here -- see send_turn and
        # the class docstring. Popen itself does not block on this.
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
        """Write the ONE turn this process was parked for (a list of
        Claude-shaped messages -- the delta beyond what this session
        already has) to its still-open stdin, close it, and block for
        the result. Must only be called after claim() returns True.

        This is deliberately identical to the cold path's single-shot
        stdin write (streaming.py's call_claude_streaming) -- the only
        difference is that the process was already spawned earlier (see
        the class docstring for why that's still a real speed win without
        needing stream-json). Reuses _consume_claude_response and the
        same NDJSON readers as the cold path so behavior is identical
        either way, not just similar."""
        from claudecode_as_openai.streaming import _consume_claude_response, _normalize_stop_sequences
        from claudecode_as_openai.parsing import _iter_ndjson_lines, _iter_ndjson_lines_pty

        stop_sequences = _normalize_stop_sequences(stop)
        with tracking.span("claude_spawn", cmd=self._debug_cmd, path="warm") as sp:
            payload = json.dumps(claude_messages)
            if self.use_pty:
                self.proc.stdin.write(payload.encode())
            else:
                self.proc.stdin.write(payload)
            self.proc.stdin.close()

            deadline = time.time() + CLAUDE_TIMEOUT_S
            chunk_source = (
                _iter_ndjson_lines_pty(self.pty_master_fd, self.proc, CLAUDE_TIMEOUT_S)
                if self.use_pty else _iter_ndjson_lines(self.proc)
            )
            result = _consume_claude_response(chunk_source, deadline, stop_sequences, stream_callback)
            if not result.get("session_id"):
                result["session_id"] = self.session_id
            usage = result.get("usage") or {}
            sp.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
            sp.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
            return result

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
        # self.mcp_tool_config["manifest_path"], if any, is NOT unlinked
        # here: see the matching comment in streaming.py's teardown -- it's
        # a cached, potentially-shared file owned by tools.py alone.


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
