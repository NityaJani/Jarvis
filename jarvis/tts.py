"""Spoken feedback using macOS's built-in `say` command."""

import subprocess


def speak(text: str) -> None:
    if not text:
        return
    subprocess.run(["say", text], check=False)
