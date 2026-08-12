# Jarvis

A local, voice-controlled agent for macOS. Say a wake word, speak a command,
and it drives real Mac apps by name to carry it out.

## Pipeline

1. **Wake phrase** — "hey Jarvis", detected via energy-based voice-activity
   detection + fast local transcription (`wake_phrase.py`), not a pretrained
   wake-word classifier. This lets the phrase be anything, at the cost of
   more CPU/battery use and ~1-2s more latency than a dedicated classifier
   (openWakeWord's `hey_jarvis` model, which this replaced).
2. **Voice lock** — **⚠ not currently implemented** (`voice_lock.py`,
   `voice_lock_worker.py`, and `jarvis.enroll_voice` referenced below don't
   exist in this codebase — confirmed by searching the actual source, not
   just missing from the file listing). Right now *any* voice saying the
   wake phrase gets a response; the design below is the intended fix,
   worth building before relying on the newer shell/file tools day to day.
   Once built: after the phrase matches, the same audio clip would be
   checked against your enrolled voiceprint (SpeechBrain's pretrained
   ECAPA-TDNN model) before responding. Enrollment would run once via
   `./venv/bin/python -m jarvis.enroll_voice`.
   - An earlier dependency-free MFCC-based approach was tried first to avoid
     the ~1GB torch/speechbrain install, but live testing showed it didn't
     actually discriminate speakers at all (synthetic impostor voices scored
     *higher* than the genuine speaker). ECAPA-TDNN was validated live
     instead: genuine ~0.5-0.8, impostors ~0.08-0.23.
   - Verification runs in a subprocess (`voice_lock_worker.py`), not the
     main process — loading torch (speechbrain) and ctranslate2
     (faster-whisper, used for transcription) in the same process segfaults.
     Confirmed by direct testing, not just a warning.
4. **Reasoning** — sends your command, plus a text list of the frontmost
   app's named UI elements (buttons, menu items, fields — from the macOS
   Accessibility API), to a rotation of free LLM providers (Groq,
   OpenRouter, Gemini — see `config.VISION_PROVIDERS`). The model decides one
   action at a time by element *name*, never by guessing pixel coordinates.
   If one provider errors or hits a rate limit, the same step is retried
   against the next.
   - We tried screenshot-based "vision" reasoning first (local models, then
     Gemini, then Groq's vision model) — every free option misjudged pixel
     coordinates and reliably failed to click the right button. Targeting
     elements by name via the Accessibility API sidesteps that failure mode
     entirely, and works with plain text+tool-calling models.
5. **Execution** — `executor.py` re-activates the target app immediately
   before every action (focus can drift back to whatever else is running
   between steps), then performs it via System Events UI scripting or
   `pyautogui` keystrokes.
6. **Feedback** — macOS `say` speaks the result back to you.
7. Loop.

Every step in the pipeline above (wake word, transcription, each LLM
provider call, each executor action) prints a `took X.XXs` timing line to
stdout/`logs/` — useful for tracking down which step is actually responsible
when a command feels slow, rather than guessing.

## Extra actions

Beyond driving apps by UI element name, a few standalone actions are always
available as voice commands:

- **Screenshot** — "take a screenshot" silently captures the whole screen to
  the clipboard (`executor.take_screenshot`). To send it somewhere, follow up
  in the same breath, e.g. "take a screenshot and send it to John in
  Messages" — the model chains `open_app` → `click_element` (the compose
  field) → `press_key 'cmd+v'` (paste) → `press_key 'return'` (send).
- **Hand-gesture crop** — "crop it" / "crop this screen" opens a live webcam
  HUD window (`jarvis/gesture_crop.py`) over a fresh screenshot:
  - Pinch (touch thumb to index) on **both hands** at once to start framing,
    move your hands to size/reposition the box, then **release** to capture
    — the crop is copied to the clipboard as PNG.
  - The window stays open for repeat crops from the same shot.
  - Close it by pressing **Esc**, by making a **closed fist** with either
    hand (held briefly to confirm), or by leaving both hands out of frame
    for a while.
  - Needs Camera permission (see below) and pulls in `mediapipe` +
    `opencv-python` for hand tracking — already listed in
    `requirements.txt`.
- **Monitors & reminders** — `jarvis/monitor.py` runs background watches
  that don't block the normal wake-word loop, so you can keep giving other
  commands while one runs:
  - "notify me when Slack gets a message from Sarah" — watches an app's
    Accessibility UI text for a keyword (`monitor_app_text`).
  - "notify me when something new gets copied" — watches the clipboard
    (`monitor_clipboard`).
  - "tell me when Zoom opens" — watches for an app launching
    (`monitor_app_launch`).
  - "remind me at 8pm to call mom" — a one-time spoken + banner reminder at
    a clock time (`set_reminder`).
  - "what are you watching?" / "stop watching Slack" — list or cancel
    active monitors (`list_monitors`, `stop_monitor`).
  - Alerts fire both spoken (`say`) and as a macOS notification banner.
    Monitors are in-memory only — they don't survive a restart.
- **Persistent memory** — `jarvis/memory.py` keeps a plain-text file
  (`jarvis_workspace/MEMORY.md`) that's injected into *every* command's
  context, so facts survive between commands instead of vanishing the
  moment one finishes (the normal behavior — each command otherwise starts
  a brand-new conversation with no history). "Remember that I take my
  coffee black" gets saved via `remember_fact` and shows up in context from
  then on; "forget about the coffee thing" removes it via `forget_fact`;
  "what do you remember?" lists it all via `list_memory`. It's just a
  markdown file — read or edit it by hand any time.
- **Browser automation** — `jarvis/browser.py` drives a real Chromium
  window (via Playwright) for web tasks, the same "target by visible
  name, not pixel position" philosophy as native-app automation:
  `browser_open` navigates and `browser_click`/`browser_type`/
  `browser_press_key` act on the page, each returning the page's current
  accessibility tree (via `Locator.aria_snapshot()`) so the model can see
  what's clickable next. One browser window stays open across commands
  (like the native-app focus persisting between executor calls) until
  `browser_close`. Runs non-headless on purpose — you should see what it's
  doing. Needs the Chromium binary installed once:
  `./venv/bin/python -m playwright install chromium`.
- **Sandboxed shell & file access** — `jarvis/shell_tools.py` adds
  `run_shell`/`read_file`/`write_file`/`list_workspace`, confined entirely
  to a `jarvis_workspace/` folder (paths that would escape it via `..` or
  an absolute path are refused, not just warned about). **This exists
  despite Jarvis having no voice authentication today** (see "Wake phrase"
  above — the voice-lock step the earlier pipeline docs describe isn't
  actually implemented), so two more layers sit in front of it:
  - A denylist blocks known-destructive patterns (`sudo`, `rm -rf`, disk
    writes, `curl | bash`, etc.) outright, before anything else runs.
  - `run_shell` and `write_file` speak back exactly what they're about to
    do and require a spoken "yes" (reusing the normal mic/transcription
    pipeline) before doing it — silence or "no" cancels.
  This stops silent/accidental execution (a TV, a passerby triggering the
  wake word) but is **not** equivalent to real speaker verification —
  whoever speaks the confirmation gets to approve it. If you're going to
  rely on this day to day, build real voice lock first (see below).

## Setup

```bash
cd /Users/apple/Jarvis
cp .env.example .env   # then add at least one provider key
```

Add at least one of these to `.env` (more = better fallback coverage):
- `GROQ_API_KEY` — free at https://console.groq.com/keys
- `OPENROUTER_API_KEY` — free at https://openrouter.ai/keys
- `GEMINI_API_KEY` — free at https://aistudio.google.com (small daily quota)

Dependencies are already installed into `venv/` (Python 3.12).

### macOS permissions (required)

Whichever app runs this (Terminal, iTerm, etc.) needs to be granted, under
**System Settings → Privacy & Security**:

- **Microphone** — for wake word + command listening
- **Accessibility** — for UI scripting (System Events) and `pyautogui` keystrokes
- **Camera** — for hand-gesture screenshot cropping (`crop_screenshot_with_gesture`)

macOS will prompt automatically the first time each is needed; if a
permission is silently denied, add the app manually in those settings panels
and restart it.

### Run

**Menu-bar app (recommended)** — launch `Jarvis.app` from `/Applications`
(Spotlight: "Jarvis"), then click "○ Jarvis" in the menu bar and check
"Enable Jarvis":

```bash
open /Applications/Jarvis.app
```

Rebuild the app after changing code (alias mode — fast, references the venv
directly):

```bash
./venv/bin/python setup.py py2app -A
```

**Or as a plain terminal loop (no app bundle):**

```bash
./venv/bin/python -m jarvis.main
```

Say "hey jarvis" (the wake-word model's trigger phrase — kept as-is; see
note below), wait for "Yes?", then speak your command, e.g.:
- "Open TextEdit and write down: buy milk"
- "Open Safari and go to the address bar"

Note: the spoken activation phrase is "hey jarvis" because that's tied to
openWakeWord's pretrained model file — changing it would require training a
custom wake-word model (a multi-hour, multi-GB undertaking), which we
decided to skip for now.

## Safety notes

- `pyautogui.FAILSAFE` is on: slam the mouse into a screen corner to
  immediately abort an in-progress action.
- The agent stops after 15 steps per command if it hasn't finished, rather
  than looping forever.
- The menu-bar toggle fully releases the microphone when disabled — it's not
  listening in the background until you re-enable it.
- This works well for native Mac apps with standard controls (TextEdit,
  Notes, Mail, Finder...). Custom-rendered UI (web canvas content, games)
  isn't addressable by element name, so it's out of scope for now.
- This is v1: Mac-only. Persistent memory now exists (`jarvis/memory.py`) but
  is a single flat file, not the scoped/multi-workspace memory a platform
  like OpenClaw has. Windows support is a natural next phase.

## Project layout

```
jarvis/
  config.py         # env/config loading, VISION_PROVIDERS rotation list
  wake_word.py      # openWakeWord listener
  transcribe.py     # faster-whisper recording + transcription
  vision_agent.py   # UI-element text -> LLM function-calling loop
  executor.py       # Accessibility API (System Events) + keystroke actions,
                     # take_screenshot
  monitor.py        # background watches: app text, clipboard, app launch,
                     # timed reminders -- notify via `say` + notification banner
  gesture_crop.py   # webcam hand-tracking crop tool (pinch/drag/release,
                     # fist to exit) -> cropped PNG on the clipboard
  memory.py         # persistent MEMORY.md, injected into every command
  shell_tools.py    # sandboxed run_shell/read_file/write_file, denylist +
                     # spoken confirmation gate
  browser.py        # Playwright-driven Chromium automation for web tasks
  tts.py            # spoken feedback via `say`
  main.py           # plain terminal loop entry point
  tray_app.py       # menu-bar toggle UI entry point (rumps), class JarvisApp
setup.py            # py2app build script -> dist/Jarvis.app
AppIcon.icns        # app icon source, referenced by setup.py
Jarvis.app          # built app bundle (also copied to /Applications)
jarvis_workspace/   # sandbox root for shell_tools + memory.py's MEMORY.md
models/             # downloaded Whisper model weights
logs/
```

### Why a real py2app bundle instead of a shell-script wrapper

The first version wrapped the venv's python in a shell script inside a plain
`.app` folder. That mostly worked, but on this macOS version, hiding the Dock
icon (`LSUIElement`, or the runtime `NSApplicationActivationPolicyAccessory`
equivalent) corrupted the menu-bar status item's position — macOS placed it
off-screen. This reproduced identically with the shell-script bundle *and* a
properly-built py2app bundle, so it's an OS-level bug with accessory-policy
apps here, not fixable in our code. We accepted a visible Dock icon as the
trade-off and switched to py2app anyway because it fixes a separate, real
bug: a shell-script `exec`'d into the system python resolves
`NSBundle.mainBundle()` to the interpreter's own location, not the app
bundle, which subtly breaks macOS's app-identity bookkeeping. py2app's
compiled launcher stub keeps the bundle identity correct.
