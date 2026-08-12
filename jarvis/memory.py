"""Persistent memory: a plain-text workspace file the agent reads at the
start of every command and can update via voice, so facts/preferences
survive between commands instead of vanishing when run_command() returns.

Deliberately simple -- one markdown file, no database, no embeddings.
Same idea as OpenClaw's AGENTS.md or Claude Code's CLAUDE.md: a persistent,
human-readable, human-editable context file rather than a hidden store.
"""

from pathlib import Path

from jarvis import config

WORKSPACE_DIR = config.ROOT_DIR / "jarvis_workspace"
MEMORY_PATH = WORKSPACE_DIR / "MEMORY.md"

MAX_MEMORY_CHARS = 4000  # keep what's injected into every command bounded


def _ensure_file() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text("# Jarvis memory\n")


def _fact_lines() -> list[str]:
    _ensure_file()
    return [line for line in MEMORY_PATH.read_text().splitlines() if line.startswith("- ")]


def read_context() -> str:
    """Text injected into every command's system context. Truncates from
    the front (keeps the most recent facts) if memory has grown past
    MAX_MEMORY_CHARS, rather than silently including an unbounded prompt."""
    facts = "\n".join(_fact_lines())
    if not facts:
        return "(nothing remembered yet)"
    if len(facts) > MAX_MEMORY_CHARS:
        facts = "(earlier memory truncated)\n" + facts[-MAX_MEMORY_CHARS:]
    return facts


def remember_fact(fact: str) -> str:
    _ensure_file()
    with MEMORY_PATH.open("a") as f:
        f.write(f"- {fact.strip()}\n")
    return f"Remembered: {fact.strip()}"


def list_memory() -> str:
    return "\n".join(_fact_lines()) or "Nothing remembered yet."


def forget_fact(fact_substring: str) -> str:
    """Removes any remembered line containing `fact_substring` (case-
    insensitive). Not exact-match-only, since the user will describe what
    to forget conversationally, not quote the stored line verbatim."""
    needle = fact_substring.strip().lower()
    kept = [line for line in _fact_lines() if needle not in line.lower()]
    removed = len(_fact_lines()) - len(kept)
    MEMORY_PATH.write_text("# Jarvis memory\n" + "\n".join(kept) + ("\n" if kept else ""))
    return f"Forgot {removed} matching line(s)." if removed else f"Nothing matched '{fact_substring}'."


ACTIONS = {
    "remember_fact": remember_fact,
    "list_memory": list_memory,
    "forget_fact": forget_fact,
}
