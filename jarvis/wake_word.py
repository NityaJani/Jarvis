"""Local, always-on wake word detection using openWakeWord."""

from typing import Callable

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from jarvis import config

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # openwakeword expects 80ms chunks at 16kHz


def listen_for_wake_word(
    model_name: str = config.WAKE_WORD,
    should_continue: Callable[[], bool] = lambda: True,
) -> bool | None:
    """Blocks until the wake word is detected or `should_continue()` turns
    false (checked between audio chunks, so the mic can be released promptly
    when the UI disables Jarvis). Returns True if detected, None if stopped
    because `should_continue` returned false."""
    oww = Model(wakeword_models=[model_name], inference_framework="onnx")

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK_SAMPLES
    ) as stream:
        print(f"[wake_word] Listening for '{model_name}' (threshold={config.WAKE_WORD_THRESHOLD})...")
        while should_continue():
            audio_chunk, _ = stream.read(CHUNK_SAMPLES)
            audio_chunk = audio_chunk.flatten().astype(np.int16)
            predictions = oww.predict(audio_chunk)
            score = predictions.get(model_name, 0.0)
            if score > 0.1:
                print(f"[wake_word] score={score:.2f}")
            if score >= config.WAKE_WORD_THRESHOLD:
                print(f"[wake_word] Detected '{model_name}' (score={score:.2f})")
                return True
    return None
