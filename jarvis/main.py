"""Jarvis main loop: wake word -> listen -> transcribe -> vision agent -> speak."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from jarvis import tts, transcribe, vision_agent, wake_word


def main() -> None:
    tts.speak("Jarvis is online.")
    while True:
        try:
            heard = wake_word.listen_for_wake_word()
            if heard is None:
                continue
            tts.speak("Yes?")

            # Conversation mode: keep taking follow-up commands directly
            # (no repeated "hey jarvis") until a real pause with nothing said.
            while True:
                command = transcribe.listen_and_transcribe()
                if not command.strip():
                    break
                response = vision_agent.run_command(command)
                print(f"[main] {response}")
                tts.speak(response)
        except KeyboardInterrupt:
            tts.speak("Goodbye.")
            break


if __name__ == "__main__":
    main()
