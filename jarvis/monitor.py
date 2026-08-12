"""Background 'monitor' watches: notify the user (spoken + macOS banner) when
a condition becomes true, without blocking the wake-word command loop.

Each monitor runs as its own daemon thread polling every POLL_SECONDS, so
starting one returns immediately and normal voice commands keep working
while it watches in the background. Monitors are in-memory only (a registry
dict guarded by a lock) — they don't survive a restart, and there's no
persistence by design; this is meant for "watch this for the next while",
not durable scheduling.
"""

import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from jarvis import executor, tts

POLL_SECONDS = 2.0


@dataclass
class _Monitor:
    id: str
    description: str
    stop_event: threading.Event = field(default_factory=threading.Event)


_monitors: dict[str, _Monitor] = {}
_lock = threading.Lock()


def _notify(message: str) -> None:
    tts.speak(message)
    # Also drop a Notification Center banner — useful if you've stepped away
    # and won't hear the spoken alert.
    safe = message.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(
        ["osascript", "-e", f'display notification "{safe}" with title "Jarvis"'],
        check=False,
    )


def _start(description: str, body: Callable[[str, threading.Event], None]) -> str:
    """Registers and starts a monitor. `body` receives its own monitor id
    (so it can deregister itself when it finishes) and its stop_event (so it
    can be cancelled early via stop_monitor)."""
    monitor_id = uuid.uuid4().hex[:8]
    with _lock:
        _monitors[monitor_id] = _Monitor(id=monitor_id, description=description)
    thread = threading.Thread(
        target=body, args=(monitor_id, _monitors[monitor_id].stop_event), daemon=True
    )
    thread.start()
    return f"Started monitor {monitor_id}: {description}"


def _finish(monitor_id: str) -> None:
    with _lock:
        _monitors.pop(monitor_id, None)


def list_monitors() -> str:
    with _lock:
        if not _monitors:
            return "No active monitors."
        return "; ".join(f"[{m.id}] {m.description}" for m in _monitors.values())


def stop_monitor(monitor_id: str) -> str:
    with _lock:
        monitor = _monitors.get(monitor_id)
    if monitor is None:
        return f"No monitor with id '{monitor_id}'. Active: {list_monitors()}"
    monitor.stop_event.set()
    return f"Stopped monitor {monitor_id} ({monitor.description})."


def watch_app_text(app: str, keyword: str) -> str:
    """Notifies the first time `keyword` shows up in `app`'s UI text that
    wasn't already there when the watch started — e.g. a new message from
    someone appearing in a chat app's element list."""
    keyword_lower = keyword.lower()

    def body(monitor_id: str, stop_event: threading.Event) -> None:
        try:
            seen = set(executor.describe_ui(app).lower().splitlines())
        except Exception:
            seen = set()
        while not stop_event.is_set():
            try:
                lines = set(executor.describe_ui(app).lower().splitlines())
            except Exception:
                lines = set()
            hit = next((line for line in (lines - seen) if keyword_lower in line), None)
            if hit:
                _notify(f'{app}: new "{keyword}" — {hit}')
                break
            seen |= lines
            stop_event.wait(POLL_SECONDS)
        _finish(monitor_id)

    return _start(f'watching {app} for "{keyword}"', body)


def watch_clipboard(matching: str = "") -> str:
    """Notifies when the clipboard changes. If `matching` is given, only
    notifies (and stops) once the new content contains that text; otherwise
    notifies on every change and keeps running until stopped."""

    def body(monitor_id: str, stop_event: threading.Event) -> None:
        try:
            last = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
        except Exception:
            last = ""
        while not stop_event.is_set():
            try:
                current = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
            except Exception:
                current = last
            if current != last:
                if not matching or matching.lower() in current.lower():
                    _notify(f"Clipboard changed: {current[:120]}")
                    if matching:
                        break
                last = current
            stop_event.wait(POLL_SECONDS)
        _finish(monitor_id)

    description = f'watching clipboard for "{matching}"' if matching else "watching clipboard for any change"
    return _start(description, body)


def watch_app_launch(app: str) -> str:
    """Notifies once when `app` is launched/opened."""

    def body(monitor_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if executor.is_running(app):
                _notify(f"{app} is now open.")
                break
            stop_event.wait(POLL_SECONDS)
        _finish(monitor_id)

    return _start(f"watching for {app} to open", body)


def _parse_time(when: str) -> datetime:
    """Parses '8pm', '20:00', '8:30 am' etc. into the next datetime that
    matches — today if that time hasn't passed yet, else tomorrow."""
    cleaned = when.strip().lower().replace(" ", "")
    match = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", cleaned)
    if not match:
        raise ValueError(f"Could not understand time '{when}'. Use e.g. '8pm', '20:00', '8:30am'.")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def set_reminder(when: str, message: str) -> str:
    """Schedules a one-time spoken + banner reminder at a specific clock
    time today (or tomorrow if that time already passed)."""
    target = _parse_time(when)

    def body(monitor_id: str, stop_event: threading.Event) -> None:
        while not stop_event.is_set() and datetime.now() < target:
            stop_event.wait(min(POLL_SECONDS, 5.0))
        if not stop_event.is_set():
            _notify(message)
        _finish(monitor_id)

    return _start(f"reminder at {target.strftime('%I:%M %p')}: {message}", body)


ACTIONS = {
    "monitor_app_text": watch_app_text,
    "monitor_clipboard": watch_clipboard,
    "monitor_app_launch": watch_app_launch,
    "set_reminder": set_reminder,
    "list_monitors": list_monitors,
    "stop_monitor": stop_monitor,
}
