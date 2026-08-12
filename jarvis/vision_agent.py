"""Decides what to do about a spoken command and drives the executor step by
step until the task is done.

Perception comes from the macOS Accessibility API (executor.describe_ui) —
a text list of named UI elements in the frontmost app — rather than a
screenshot. Free/small vision-language models proved unreliable at guessing
pixel coordinates from an image; targeting elements by name is deterministic
and works with plain text-and-tool-calling models instead.

Uses a rotation of free LLM providers (Groq, OpenRouter, Gemini — see
config.VISION_PROVIDERS), all spoken to via the OpenAI-compatible
chat-completions format, so a single code path covers all of them. If a
provider errors or is rate-limited on a given step, the same step is retried
against the next provider in the rotation.
"""

import json
import time

from openai import OpenAI

from jarvis import browser, config, executor, gesture_crop, memory, monitor, shell_tools

MAX_STEPS = 15

SYSTEM_PROMPT = """You are Jarvis, a voice-controlled assistant that controls a \
macOS screen. You are given a spoken user command and, each turn, the name \
of the frontmost app plus a text list of its named interactive UI elements \
(buttons, menu items, text fields, etc.) — this is your only view of the \
screen, there is no image. Decide the single next action to take, or call \
`done` if the task is already complete, or `ask_user` if you need \
clarification. Only reference elements that actually appear in the current \
list — never invent a name. Take one action at a time; you will be shown a \
fresh element list after each action.

Important patterns:
- After `open_app`, many document-based apps (TextEdit, Pages, Word) show an
  "Open" file-picker dialog instead of a blank document. If you see a button
  named "New Document" in the element list, click it before typing anything.
- click_element/click_menu_item/type_text/press_key all take an `app`
  parameter — always pass the current frontmost app name shown to you.
- Never call `done` on faith. Before declaring success, check the CURRENT
  element list / app state for evidence the action actually happened. If you
  can't confirm it, keep working instead of reporting success.
- Microsoft Teams (process name "MSTeams") renders its UI in a way that
  exposes almost no named elements — its element list will usually be empty
  or near-empty. Don't try to find a search box or compose field by name
  there; use these known keyboard shortcuts with press_key instead:
  - cmd+e: open search
  - cmd+n: start a new chat
  - return: send the message currently typed in the compose box
  If a Teams task needs something beyond these, say so via `ask_user` rather
  than guessing at elements that won't exist.
- take_screenshot captures the whole screen to the clipboard. To send it
  somewhere, after capturing: open_app the target, click_element the
  message/compose field to focus it, then press_key 'cmd+v' to paste the
  image, then press_key 'return' to send it if that's what was asked.
- monitor_app_text/monitor_clipboard/monitor_app_launch/set_reminder all
  start a background watch and return immediately — they do NOT block. Call
  one of these, then call `done` right away with a short confirmation
  (mentioning the monitor id from the result if the user might want to stop
  it later). Never call `wait` after starting a monitor expecting it to
  finish.
- crop_screenshot_with_gesture is different: it BLOCKS until the user closes
  it (Esc, making a fist with either hand, or hands absent for a while).
  Pinching both hands (thumb to index) and releasing captures one crop and
  copies it to the clipboard, but
  the window keeps running afterward so they can capture more regions from
  the same shot — don't expect it to return after just one crop. Only call
  it when the command is specifically about cropping/selecting a region with
  hands/gesture (e.g. "crop it", "crop this screen"), not for a plain "take
  a screenshot". It already returns the final result (a summary of how many
  crops were made, or cancelled) — just
  relay that via `done` afterward, don't call `wait` first.
- You have persistent memory across commands: remembered facts are listed
  above under "Remembered context" (if any). Use remember_fact whenever the
  user tells you something worth keeping for later ("remember that...", a
  stated preference, a recurring detail) — don't wait to be asked twice.
  Use forget_fact if they say to forget/stop remembering something.
- run_shell/read_file/write_file/list_workspace are sandboxed to a
  jarvis_workspace/ folder and CANNOT touch anything outside it. run_shell
  and write_file will ask the user to confirm out loud before doing
  anything — this happens automatically, you don't need to ask separately;
  just call the tool and relay whatever result/cancellation comes back.
- browser_open/browser_click/browser_type/browser_press_key/browser_go_back
  control a real Chromium window for web tasks (native macOS apps still go
  through open_app/click_element instead — browser tools are only for web
  pages). browser_open and browser_click return the page's current named
  elements, the same way describe_ui does for native apps — target elements
  by that visible text, never by guessing position. Call browser_close when
  a web task is finished if the user doesn't need the page left open."""


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOLS = [
    _tool(
        "open_app",
        "Open a macOS application by name, e.g. 'Notes', 'Safari'.",
        {"name": {"type": "string"}},
        ["name"],
    ),
    _tool(
        "click_element",
        "Click a named UI element (button, checkbox, menu button, row...) in the given app's frontmost window.",
        {"app": {"type": "string"}, "name": {"type": "string"}},
        ["app", "name"],
    ),
    _tool(
        "click_menu_item",
        "Click an item in the app's menu bar, e.g. menu 'File', item 'New'.",
        {"app": {"type": "string"}, "menu": {"type": "string"}, "item": {"type": "string"}},
        ["app", "menu", "item"],
    ),
    _tool(
        "type_text",
        "Type text into the given app's currently focused field.",
        {"app": {"type": "string"}, "text": {"type": "string"}},
        ["app", "text"],
    ),
    _tool(
        "press_key",
        "Press a key or key combo in the given app, e.g. 'return', 'cmd+c', 'cmd+shift+4'.",
        {"app": {"type": "string"}, "keys": {"type": "string"}},
        ["app", "keys"],
    ),
    _tool(
        "wait",
        "Wait for the given number of seconds (e.g. for an app to load).",
        {"seconds": {"type": "number"}},
        ["seconds"],
    ),
    _tool(
        "done",
        "Call this when the task is complete.",
        {"message": {"type": "string"}},
        ["message"],
    ),
    _tool(
        "ask_user",
        "Call this if the command is ambiguous and you need clarification.",
        {"question": {"type": "string"}},
        ["question"],
    ),
    _tool(
        "take_screenshot",
        "Captures the entire screen to the clipboard (silently, no shutter sound). "
        "Paste it elsewhere with press_key 'cmd+v' after focusing the target field.",
        {},
        [],
    ),
    _tool(
        "crop_screenshot_with_gesture",
        "Takes a fresh screenshot and opens a webcam HUD window so the user can frame "
        "each hand with thumb + index finger, then pinch (touch thumb to index) on "
        "BOTH hands at once and release to capture the crop between the two pinch "
        "points, copying it to the clipboard. The window stays open for repeat crops "
        "from the same shot until the user presses Esc, closes either hand into a "
        "fist, or steps away — it does not close after one crop. Blocks until closed. "
        "Triggered by short phrases too, "
        "e.g. 'crop it', 'crop this', 'crop this screen' — the user does NOT need to "
        "say 'screenshot' or 'with my hands' every time; this tool already takes the "
        "screenshot itself, so a bare 'crop it' is enough.",
        {},
        [],
    ),
    _tool(
        "monitor_app_text",
        "Starts a background watch on an app's UI text (e.g. a chat or mail app) and "
        "notifies (spoken + banner) the first time new text containing the keyword "
        "appears. Runs in the background — returns immediately.",
        {
            "app": {"type": "string"},
            "keyword": {"type": "string", "description": "substring to watch for, e.g. a sender's name or word"},
        },
        ["app", "keyword"],
    ),
    _tool(
        "monitor_clipboard",
        "Starts a background watch on the clipboard. If `matching` is given, notifies "
        "and stops once new clipboard content contains that text; otherwise notifies "
        "on every change until stopped.",
        {"matching": {"type": "string"}},
        [],
    ),
    _tool(
        "monitor_app_launch",
        "Starts a background watch that notifies once when the named app is launched.",
        {"app": {"type": "string"}},
        ["app"],
    ),
    _tool(
        "set_reminder",
        "Schedules a one-time spoken + banner reminder at a specific clock time today "
        "(or tomorrow if that time already passed), e.g. '8pm', '20:00', '8:30 am'.",
        {"when": {"type": "string"}, "message": {"type": "string", "description": "what to say when it fires"}},
        ["when", "message"],
    ),
    _tool(
        "list_monitors",
        "Lists currently active background monitors/reminders with their ids.",
        {},
        [],
    ),
    _tool(
        "stop_monitor",
        "Stops/cancels an active monitor or reminder by its id (from list_monitors).",
        {"monitor_id": {"type": "string"}},
        ["monitor_id"],
    ),
    _tool(
        "remember_fact",
        "Saves a fact/preference to persistent memory so it's available in future commands, "
        "not just this one. Use whenever the user shares something worth keeping.",
        {"fact": {"type": "string"}},
        ["fact"],
    ),
    _tool(
        "list_memory",
        "Lists everything currently remembered.",
        {},
        [],
    ),
    _tool(
        "forget_fact",
        "Removes remembered fact(s) matching a substring/description.",
        {"fact_substring": {"type": "string"}},
        ["fact_substring"],
    ),
    _tool(
        "run_shell",
        "Runs a shell command inside the sandboxed jarvis_workspace/ folder. Refuses "
        "known-destructive patterns outright and asks the user to confirm out loud "
        "before running anything else — this confirmation happens automatically inside "
        "the tool.",
        {"command": {"type": "string"}},
        ["command"],
    ),
    _tool(
        "read_file",
        "Reads a file's contents from within jarvis_workspace/ (path is relative to it).",
        {"path": {"type": "string"}},
        ["path"],
    ),
    _tool(
        "write_file",
        "Writes content to a file within jarvis_workspace/ (path is relative to it), after "
        "asking the user to confirm out loud — this happens automatically inside the tool.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _tool(
        "list_workspace",
        "Lists files currently in the jarvis_workspace/ sandbox.",
        {},
        [],
    ),
    _tool(
        "browser_open",
        "Opens a URL in a real Chromium browser window (launches it if not already open) and "
        "returns the page's named interactive elements.",
        {"url": {"type": "string"}},
        ["url"],
    ),
    _tool(
        "browser_click",
        "Clicks a button/link/text on the current web page by its visible text (not "
        "coordinates). Returns the page's elements afterward.",
        {"text": {"type": "string"}},
        ["text"],
    ),
    _tool(
        "browser_type",
        "Types text into whatever field is currently focused on the web page.",
        {"text": {"type": "string"}},
        ["text"],
    ),
    _tool(
        "browser_press_key",
        "Presses a key/combo on the web page, e.g. 'Enter', 'Tab', 'Control+A'.",
        {"keys": {"type": "string"}},
        ["keys"],
    ),
    _tool(
        "browser_go_back",
        "Navigates the browser back one page.",
        {},
        [],
    ),
    _tool(
        "browser_close",
        "Closes the browser window/session.",
        {},
        [],
    ),
]

_clients = {
    name: OpenAI(api_key=api_key, base_url=base_url)
    for name, base_url, api_key, _ in config.VISION_PROVIDERS
}

# Tool names not handled by executor.execute() -- each module owns its own
# dispatch dict; merged once here so run_command doesn't need a growing
# if/elif chain every time a new tool module is added.
_EXTRA_ACTIONS = {
    **monitor.ACTIONS,
    **gesture_crop.ACTIONS,
    **memory.ACTIONS,
    **shell_tools.ACTIONS,
    **browser.ACTIONS,
}


def _call_with_fallback(messages: list[dict]):
    """Tries each configured provider in order for this one step; returns the
    first successful response. Raises the last error if all fail."""
    last_error = None
    for name, _, _, model in config.VISION_PROVIDERS:
        start = time.perf_counter()
        try:
            response = _clients[name].chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
            )
            print(f"[vision_agent] provider '{name}' responded in {time.perf_counter() - start:.2f}s")
            return response
        except Exception as exc:  # noqa: BLE001
            print(f"[vision_agent] provider '{name}' failed after {time.perf_counter() - start:.2f}s: {exc}")
            last_error = exc
    raise RuntimeError(f"All vision providers failed. Last error: {last_error}")


def run_command(command: str) -> str:
    """Drives the agent loop for a single spoken command. Returns a final
    spoken response for the user."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Remembered context from previous commands:\n{memory.read_context()}"},
        {"role": "user", "content": f"User said: {command!r}"},
    ]

    for _ in range(MAX_STEPS):
        step_start = time.perf_counter()
        current_app = executor.frontmost_app()

        # Jarvis itself (a menu-bar app with no real window) can briefly
        # become "frontmost" if you click its Dock/menu icon. Give focus a
        # moment to settle back to whatever you were actually using rather
        # than trying to read our own UI.
        settle_deadline = time.time() + 1.5
        while current_app == "Jarvis" and time.time() < settle_deadline:
            time.sleep(0.2)
            current_app = executor.frontmost_app()

        ui_start = time.perf_counter()
        try:
            ui_text = executor.describe_ui(current_app)
        except Exception as exc:  # noqa: BLE001
            ui_text = f"(could not read this app's UI: {exc})"
        print(f"[vision_agent] describe_ui({current_app!r}) took {time.perf_counter() - ui_start:.2f}s")

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Frontmost app: {current_app}\n"
                    f"Interactive elements:\n{ui_text}\n"
                    "Decide the next action."
                ),
            }
        )

        response = _call_with_fallback(messages)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        tool_calls = message.tool_calls
        if not tool_calls:
            return message.content or "I'm not sure what to do next."

        # Models are told to take one action at a time, but if one returns
        # several tool_calls in a single turn, every tool_call_id still needs
        # a matching tool response — only the first is actually executed.
        tool_call = tool_calls[0]
        name = tool_call.function.name
        try:
            params = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            params = {}

        if name == "done":
            return params.get("message", "Done.")
        if name == "ask_user":
            return params.get("question", "Could you clarify?")

        try:
            if name in _EXTRA_ACTIONS:
                result_text = str(_EXTRA_ACTIONS[name](**params))
            else:
                executor.execute(name, params)
                result_text = "ok"
        except Exception as exc:  # noqa: BLE001
            result_text = f"error: {exc}"

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            }
        )
        for extra_call in tool_calls[1:]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": extra_call.id,
                    "content": "skipped: only one action is executed per turn",
                }
            )
        print(f"[vision_agent] step ({name}) total took {time.perf_counter() - step_start:.2f}s")

    return "I stopped after taking a lot of steps without finishing — could you check what happened?"
