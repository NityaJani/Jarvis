"""Executes concrete UI actions on macOS via the Accessibility API rather
than guessing pixel coordinates from a screenshot — free/small vision models
proved unreliable at pixel-precise clicking, but every action here targets UI
elements by their actual name/role, which is deterministic.

UI element reading/clicking goes straight through PyObjC's AXUIElement
bindings rather than `osascript`/System Events UI scripting: the AppleScript
"entire contents" idiom we used originally could take 5-10+ seconds on
complex windows, since each element access is a slow AppleScript-interpreter
round trip. Direct Accessibility API calls avoid that overhead entirely."""

import subprocess
import time

import ApplicationServices as AX
import pyautogui

pyautogui.FAILSAFE = True  # move mouse to a screen corner to abort
pyautogui.PAUSE = 0.15

_INTERACTIVE_ROLES = {
    "AXButton",
    "AXMenuButton",
    "AXPopUpButton",
    "AXCheckBox",
    "AXRadioButton",
    "AXTextField",
    "AXTextArea",
    "AXStaticText",
    "AXMenuItem",
    "AXTab",
    "AXSlider",
    "AXRow",
    "AXCell",
}

_MAX_UI_ELEMENTS = 80
_MAX_UI_DEPTH = 30


def is_running(name: str) -> bool:
    return _pid_for_app(name) is not None


def _pid_for_app(name: str) -> int | None:
    # NSWorkspace.runningApplications() returns a cached snapshot that never
    # refreshes without an active CFRunLoop pumping notifications — useless
    # in a plain script/background thread. A one-off System Events query is
    # cheap (unlike the "entire contents" traversal) and always current.
    stdout, code, _ = _run_osascript(
        f'tell application "System Events" to get unix id of process "{_escape(name)}"'
    )
    if code == 0 and stdout.isdigit():
        return int(stdout)
    return None


def _ax_attr(element, attribute):
    err, value = AX.AXUIElementCopyAttributeValue(element, attribute, None)
    return value if err == 0 else None


def _element_label(element) -> str | None:
    for attr in (AX.kAXTitleAttribute, AX.kAXDescriptionAttribute, AX.kAXValueAttribute):
        value = _ax_attr(element, attr)
        if isinstance(value, str) and value:
            return value
    return None


def _frontmost_window(app_name: str, timeout: float = 2.0):
    # NSWorkspace's running-apps list can lag slightly right after an app is
    # launched/activated, so retry briefly instead of failing immediately.
    deadline = time.time() + timeout
    pid = None
    while time.time() < deadline:
        pid = _pid_for_app(app_name)
        if pid is not None:
            break
        time.sleep(0.1)
    if pid is None:
        raise RuntimeError(f"'{app_name}' is not running.")

    app_ref = AX.AXUIElementCreateApplication(pid)
    while time.time() < deadline:
        window = _ax_attr(app_ref, AX.kAXFocusedWindowAttribute)
        if window is not None:
            return window
        windows = _ax_attr(app_ref, AX.kAXWindowsAttribute)
        if windows:
            return windows[0]
        time.sleep(0.1)
    raise RuntimeError(f"'{app_name}' has no open window.")


def _walk_elements(element, out: list, depth: int = 0) -> None:
    if len(out) >= _MAX_UI_ELEMENTS or depth > _MAX_UI_DEPTH:
        return
    role = _ax_attr(element, AX.kAXRoleAttribute)
    label = _element_label(element)
    if role in _INTERACTIVE_ROLES and label:
        out.append((role, label, element))
    for child in _ax_attr(element, AX.kAXChildrenAttribute) or []:
        if len(out) >= _MAX_UI_ELEMENTS:
            return
        _walk_elements(child, out, depth + 1)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _run_osascript(script: str) -> tuple[str, int, str]:
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return result.stdout.strip(), result.returncode, result.stderr.strip()


def run_applescript(script: str) -> str:
    stdout, _, _ = _run_osascript(script)
    return stdout


def frontmost_app() -> str:
    stdout, _, _ = _run_osascript(
        'tell application "System Events" to get name of first process whose frontmost is true'
    )
    return stdout


def _bundle_id_for(name: str) -> str:
    """Resolves a Launch Services app name (e.g. "Microsoft Teams") or an
    already-running process's System Events name (e.g. "MSTeams") to its
    bundle identifier. Returns '' if neither resolves."""
    stdout, code, _ = _run_osascript(f'id of application "{_escape(name)}"')
    if code == 0 and stdout:
        return stdout
    stdout, code, _ = _run_osascript(
        f'tell application "System Events" to get bundle identifier of process "{_escape(name)}"'
    )
    return stdout if code == 0 else ""


def _frontmost_bundle_id() -> str:
    stdout, code, _ = _run_osascript(
        'tell application "System Events" to get bundle identifier of first process whose frontmost is true'
    )
    return stdout if code == 0 else ""


def _activate(name: str) -> None:
    result = subprocess.run(["open", "-a", name], capture_output=True, text=True)
    if result.returncode == 0:
        return
    # `name` is often a running process's internal name (e.g. "MSTeams",
    # what System Events/our own UI-reading exposes and what the model
    # sees), which doesn't always match the app's actual Launch Services
    # name ("Microsoft Teams") that `open -a` needs. If a process is
    # already running under that name, resolve its bundle identifier via
    # System Events and activate by that instead.
    bundle_id = _bundle_id_for(name)
    if bundle_id:
        subprocess.run(["open", "-b", bundle_id], check=False)


def _ensure_frontmost(name: str, timeout: float = 5.0) -> None:
    """Activates `name` and blocks until it's confirmed frontmost. Every
    action below calls this immediately before acting — focus can drift
    back to another app (e.g. the terminal/editor driving this process)
    between one action and the next, so activation can't just happen once
    at the start of a task.

    Confirmation compares bundle identifiers, not name strings: a process's
    System Events name ("MSTeams") and its Launch Services app name
    ("Microsoft Teams") can differ, and the model may use either — comparing
    raw strings made this falsely time out even when the app genuinely came
    to the front."""
    _activate(name)
    target_bundle_id = _bundle_id_for(name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if target_bundle_id and _frontmost_bundle_id() == target_bundle_id:
            return
        if frontmost_app().lower() == name.lower():
            return
        time.sleep(0.15)
    raise RuntimeError(f"Could not bring '{name}' to the front within {timeout}s.")


def open_app(name: str, timeout: float = 8.0) -> None:
    _activate(name)
    _ensure_frontmost(name, timeout=timeout)


def describe_ui(app_name: str) -> str:
    """Text summary of named interactive elements in app_name's frontmost
    window — used as the model's 'view' of the screen instead of a
    screenshot, so it can target elements by name rather than coordinates."""
    _ensure_frontmost(app_name)
    window = _frontmost_window(app_name)
    found: list = []
    _walk_elements(window, found)
    seen = set()
    lines = []
    for role, label, _ in found:
        key = (role, label)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f'{role}: "{label}"')
    return "\n".join(lines) if lines else "(no named interactive elements found)"


def click_element(app: str, name: str) -> None:
    _ensure_frontmost(app)
    window = _frontmost_window(app)
    found: list = []
    _walk_elements(window, found)
    for _, label, element in found:
        if label == name:
            err = AX.AXUIElementPerformAction(element, AX.kAXPressAction)
            if err != 0:
                raise RuntimeError(f"Could not click '{name}' in {app} (AXError {err})")
            return
    raise RuntimeError(f"UI element named '{name}' not found in {app}")


def click_menu_item(app: str, menu: str, item: str) -> None:
    _ensure_frontmost(app)
    script = f'''
    tell application "System Events"
        tell process "{_escape(app)}"
            tell menu bar 1
                tell menu bar item "{_escape(menu)}"
                    tell menu "{_escape(menu)}"
                        click menu item "{_escape(item)}"
                    end tell
                end tell
            end tell
        end tell
    end tell
    '''
    _, code, err = _run_osascript(script)
    if code != 0:
        raise RuntimeError(f"Could not click menu item '{menu}' > '{item}' in {app}: {err}")


def type_text(app: str, text: str) -> None:
    _ensure_frontmost(app)
    pyautogui.typewrite(text, interval=0.02)


def press_key(app: str, keys: str) -> None:
    """keys like 'return', 'cmd+c', 'cmd+shift+4'."""
    _ensure_frontmost(app)
    parts = [k.strip() for k in keys.split("+")]
    if len(parts) == 1:
        pyautogui.press(parts[0])
    else:
        pyautogui.hotkey(*parts)


def wait(seconds: float) -> None:
    time.sleep(seconds)


def take_screenshot() -> None:
    """Captures the whole screen to the clipboard (silently, no shutter
    sound) so it can be pasted elsewhere with press_key 'cmd+v' — no temp
    file to clean up, and pasting is how most chat/mail compose boxes accept
    an image anyway."""
    subprocess.run(["screencapture", "-c", "-x"], check=True)


ACTIONS = {
    "open_app": open_app,
    "click_element": click_element,
    "click_menu_item": click_menu_item,
    "type_text": type_text,
    "press_key": press_key,
    "wait": wait,
    "take_screenshot": take_screenshot,
}


def execute(action_name: str, params: dict) -> object:
    if action_name not in ACTIONS:
        raise ValueError(f"Unknown action: {action_name}")
    start = time.perf_counter()
    try:
        return ACTIONS[action_name](**params)
    finally:
        print(f"[executor] {action_name}({params}) took {time.perf_counter() - start:.2f}s")
