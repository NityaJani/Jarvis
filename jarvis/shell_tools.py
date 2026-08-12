"""Sandboxed shell + file tools -- the closest equivalent to OpenClaw's
bash/read/write/edit tools, but scoped down hard, because Jarvis currently
has NO voice authentication (the README's "voice lock" step doesn't exist
in this codebase): anyone's voice near the mic can trigger a command. Three
layers of protection instead of trusting the wake word:

1. Filesystem sandbox -- every path is resolved relative to
   jarvis_workspace/ and refused outright if it would land outside it. Not
   a string prefix check: uses Path.resolve() + is_relative_to() so a
   "../.." can't escape it.
2. A denylist blocks known-destructive command patterns (sudo, rm -rf,
   disk-level writes, piping a download straight into a shell, etc.)
   before anything runs, regardless of confirmation.
3. Every actual execution or write is gated behind a SPOKEN confirmation,
   reusing the exact same mic/transcription pipeline as normal commands --
   Jarvis reads back exactly what it's about to do and requires a spoken
   "yes" before doing it. Silence or "no" cancels.

This is a mitigation, not equivalent to real speaker verification: whoever
speaks the confirmation gets to approve it, so it stops silent/accidental
execution (a TV, a passerby's wake-word trigger) but not a deliberate
impersonator standing right there. Actual voice-lock (per the README) is
the real fix for that half of the threat model -- build that first if this
tool is going to be relied on.
"""

import re
import subprocess
from pathlib import Path

from jarvis import config, transcribe, tts

WORKSPACE_DIR = config.ROOT_DIR / "jarvis_workspace"
COMMAND_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARS = 4000

_DENYLIST_PATTERNS = [
    r"\bsudo\b",
    r"\brm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\b",  # rm -rf / -fr in any flag order
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{.*\};\s*:",  # classic fork bomb
    r"curl[^|]*\|\s*(sh|bash|zsh)\b",
    r"wget[^|]*\|\s*(sh|bash|zsh)\b",
    r"\bchmod\s+-R\s+777\b",
    r">\s*/dev/(disk|sd|rdisk)",
]


def _ensure_workspace() -> Path:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_DIR


def _resolve_in_workspace(path: str) -> Path:
    workspace = _ensure_workspace().resolve()
    candidate = (workspace / path).resolve()
    if not candidate.is_relative_to(workspace):
        raise ValueError(f"'{path}' is outside the jarvis_workspace sandbox -- refused.")
    return candidate


def _denylisted(command: str) -> str | None:
    for pattern in _DENYLIST_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return pattern
    return None


def _confirm(prompt: str) -> bool:
    """Speaks `prompt`, listens for a spoken reply on the same mic pipeline
    used for normal commands, and returns whether it was affirmative.
    Silence (nothing heard in time) is treated as "no" -- fails closed."""
    tts.speak(prompt)
    reply = transcribe.listen_and_transcribe().strip().lower()
    return any(word in reply for word in ("yes", "yeah", "yep", "confirm", "go ahead", "do it", "sure"))


def run_shell(command: str) -> str:
    """Runs a shell command inside jarvis_workspace/, after a denylist
    check and a spoken confirmation."""
    hit = _denylisted(command)
    if hit:
        return f"refused: '{command}' matches a blocked pattern ({hit}) and was not run."
    if not _confirm(f"About to run: {command}. Say yes to confirm."):
        return "cancelled: did not get a spoken confirmation."
    try:
        result = subprocess.run(
            command, shell=True, cwd=_ensure_workspace(),
            capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {COMMAND_TIMEOUT_SECONDS}s"
    output = (result.stdout + result.stderr).strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[-MAX_OUTPUT_CHARS:]
    return f"exit {result.returncode}: {output or '(no output)'}"


def read_file(path: str) -> str:
    target = _resolve_in_workspace(path)
    if not target.exists():
        return f"error: '{path}' does not exist in jarvis_workspace."
    text = target.read_text(errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + "\n...(truncated)"
    return text


def write_file(path: str, content: str) -> str:
    target = _resolve_in_workspace(path)
    if not _confirm(f"About to write {len(content)} characters to {path}. Say yes to confirm."):
        return "cancelled: did not get a spoken confirmation."
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} characters to {path}."


def list_workspace() -> str:
    workspace = _ensure_workspace()
    entries = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*"))
    return "\n".join(entries) if entries else "(jarvis_workspace is empty)"


ACTIONS = {
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "list_workspace": list_workspace,
}
