"""macOS menu-bar UI: click 'Enable Jarvis' to turn voice listening on/off.

Run with: ./venv/bin/python -m jarvis.tray_app
"""

import os
import sys
import threading
import time

# ctranslate2 (faster-whisper's backend) can conflict with other OpenMP
# runtimes in the same process; harmless to set even without torch present.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# py2app's bundled bootstrap doesn't add the project root to sys.path, so the
# jarvis package isn't importable there without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rumps

from jarvis import tts, transcribe, vision_agent, wake_word

# Note: we deliberately do NOT hide the Dock icon (no LSUIElement, no
# NSApplicationActivationPolicyAccessory). Both approaches were tested and
# both corrupt this app's status bar item positioning on this macOS version
# (item gets placed off-screen) — a real OS-level bug with accessory-policy
# apps here, not something fixable in our code. A visible Dock icon is a
# reasonable trade-off: the app is fully functional, and it also makes it
# obvious at a glance that Jarvis actually launched.

ICON_OFF = "○"  # ◯
ICON_IDLE = "\U0001f7e2"  # 🟢 enabled, listening for wake word
ICON_COMMAND = "\U0001f3a4"  # 🎤 recording your command
ICON_THINKING = "\U0001f4ad"  # 💭 vision agent working


class JarvisApp(rumps.App):
    def __init__(self):
        super().__init__("Jarvis", title=f"{ICON_OFF} Jarvis")
        self.enabled = threading.Event()
        self.quitting = threading.Event()

        self.toggle_item = rumps.MenuItem("Enable Jarvis", callback=self.toggle)
        self.status_item = rumps.MenuItem("Status: Off")
        self.status_item.set_callback(None)  # non-clickable, just a label
        self.menu = [self.toggle_item, self.status_item]

        self.worker = threading.Thread(target=self._run_loop, daemon=True)
        self.worker.start()

    def toggle(self, sender):
        if self.enabled.is_set():
            self.enabled.clear()
            sender.state = False
            self._set_status("Off", ICON_OFF)
        else:
            self.enabled.set()
            sender.state = True
            self._set_status("Listening for wake word...", ICON_IDLE)

    def _set_status(self, text: str, icon: str) -> None:
        self.status_item.title = f"Status: {text}"
        self.title = f"{icon} Jarvis"

    def _run_loop(self) -> None:
        while not self.quitting.is_set():
            if not self.enabled.is_set():
                time.sleep(0.3)
                continue

            heard = wake_word.listen_for_wake_word(
                should_continue=lambda: self.enabled.is_set()
                and not self.quitting.is_set()
            )
            if heard is None:
                continue  # disabled mid-listen; loop back and idle

            tts.speak("Yes?")

            # Conversation mode: once woken, keep taking follow-up commands
            # directly (no repeated "hey jarvis") as long as you keep
            # speaking within each pause window. It only drops back to
            # needing the wake word after a real pause with nothing said.
            while self.enabled.is_set() and not self.quitting.is_set():
                self._set_status("Listening for command...", ICON_COMMAND)
                command = transcribe.listen_and_transcribe()

                if not command.strip():
                    self._set_status("Listening for wake word...", ICON_IDLE)
                    break

                self._set_status("Thinking...", ICON_THINKING)
                try:
                    response = vision_agent.run_command(command)
                except Exception as exc:  # noqa: BLE001
                    response = f"Something went wrong: {exc}"
                tts.speak(response)

            if not self.enabled.is_set():
                self._set_status("Off", ICON_OFF)


def main() -> None:
    print("[tray_app] Warming up Whisper model...")
    transcribe.get_model()
    print("[tray_app] Ready.")

    JarvisApp().run()


if __name__ == "__main__":
    main()
