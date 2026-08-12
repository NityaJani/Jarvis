"""Records a command after wake word and transcribes it locally with Whisper.

Recording is voice-activity-gated rather than a fixed-length window: it
waits (patiently) for you to actually start speaking, then keeps recording
until you pause. A fixed short window either started recording before you
were ready (losing the start of your command) or cut you off mid-sentence
if you took longer than expected — both made it seem like Jarvis "gave up"
waiting for input.
"""

import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from jarvis import config

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# Energy-based VAD, same approach as wake-word buffering: adapts to the
# room's ambient noise level rather than using a fixed volume cutoff.
SPEECH_MULTIPLIER = 3.0
MIN_SPEECH_ENERGY = 150.0
NOISE_FLOOR_ADAPT_RATE = 0.05

MAX_WAIT_FOR_SPEECH_SECONDS = 8.0  # patience before giving up with no input
SILENCE_HANGOVER_SECONDS = 1.2  # trailing pause that ends the command
MAX_COMMAND_SECONDS = 20.0  # hard cap so a stuck mic can't hang forever

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            download_root=str(config.MODELS_DIR),
        )
    return _model


def _frame_energy(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


def record_command() -> np.ndarray:
    """Waits for speech to start (up to MAX_WAIT_FOR_SPEECH_SECONDS), then
    records until a trailing pause or MAX_COMMAND_SECONDS is reached.
    Returns an empty array if nothing was said in time."""
    frames: list[np.ndarray] = []
    noise_floor = MIN_SPEECH_ENERGY
    silence_hangover_frames = int(SILENCE_HANGOVER_SECONDS * 1000 / FRAME_MS)
    max_wait_frames = int(MAX_WAIT_FOR_SPEECH_SECONDS * 1000 / FRAME_MS)
    max_total_frames = int(MAX_COMMAND_SECONDS * 1000 / FRAME_MS)

    print("[transcribe] Waiting for you to speak...")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES
    ) as stream:
        in_speech = False
        silence_run = 0
        waited_frames = 0
        while True:
            frame, _ = stream.read(FRAME_SAMPLES)
            energy = _frame_energy(frame)
            is_speech = energy > max(noise_floor * SPEECH_MULTIPLIER, MIN_SPEECH_ENERGY)

            if not in_speech:
                if not is_speech:
                    noise_floor += NOISE_FLOOR_ADAPT_RATE * (energy - noise_floor)
                    waited_frames += 1
                    if waited_frames >= max_wait_frames:
                        print("[transcribe] No speech detected, giving up.")
                        return np.array([], dtype=np.float32)
                    continue
                in_speech = True

            frames.append(frame.copy())
            silence_run = 0 if is_speech else silence_run + 1
            if silence_run >= silence_hangover_frames or len(frames) >= max_total_frames:
                break

    clip = np.concatenate(frames).flatten().astype(np.float32) / 32768.0
    print(f"[transcribe] Captured {len(clip) / SAMPLE_RATE:.1f}s of command audio.")
    return clip


def transcribe(audio: np.ndarray) -> str:
    start = time.perf_counter()
    model = get_model()
    segments, _ = model.transcribe(audio, language="en")
    text = " ".join(segment.text.strip() for segment in segments)
    print(f"[transcribe] Heard: {text!r} (whisper took {time.perf_counter() - start:.2f}s)")
    return text


def listen_and_transcribe() -> str:
    record_start = time.perf_counter()
    audio = record_command()
    print(f"[transcribe] record_command took {time.perf_counter() - record_start:.2f}s")
    if len(audio) == 0:
        return ""
    return transcribe(audio)
