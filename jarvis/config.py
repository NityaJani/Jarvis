import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

WAKE_WORD = os.environ.get("JARVIS_WAKE_WORD", "hey_jarvis")
# Lowered from the library's default 0.5 — user reported needing many
# repeated attempts before one crossed threshold. Trades a bit more
# sensitivity to background noise for far fewer required attempts.
WAKE_WORD_THRESHOLD = float(os.environ.get("JARVIS_WAKE_THRESHOLD", "0.3"))

WHISPER_MODEL_SIZE = os.environ.get("JARVIS_WHISPER_MODEL", "medium.en")

MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Vision-reasoning providers, tried in order; a step falls through to the next
# entry if one is rate-limited or errors. All speak the OpenAI-compatible
# chat-completions format, so they share one code path in vision_agent.py.
# Only entries whose API key is actually set are used.
_PROVIDER_CANDIDATES = [
    ("groq", "https://api.groq.com/openai/v1", GROQ_API_KEY, "meta-llama/llama-4-scout-17b-16e-instruct"),
    ("openrouter-nemotron-vl", "https://openrouter.ai/api/v1", OPENROUTER_API_KEY, "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("openrouter-gemma", "https://openrouter.ai/api/v1", OPENROUTER_API_KEY, "google/gemma-4-31b-it:free"),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", GEMINI_API_KEY, "gemini-2.5-flash"),
]
VISION_PROVIDERS = [p for p in _PROVIDER_CANDIDATES if p[2]]

if not VISION_PROVIDERS:
    raise RuntimeError(
        "No vision provider API keys are set. Add at least one of "
        "GROQ_API_KEY, OPENROUTER_API_KEY, or GEMINI_API_KEY to .env."
    )
