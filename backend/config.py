"""
Central configuration for the voice RAG agent.
All secrets are loaded from environment variables (see .env.example).
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return val


@dataclass(frozen=True)
class Settings:
    # --- STT ---
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY", "")
    deepgram_model: str = "nova-3"

    # --- TTS ---
    # "cartesia" or "deepgram" — switchable via .env, no code changes needed.
    # Deepgram TTS (Aura-2) reuses the same DEEPGRAM_API_KEY as STT above —
    # no separate signup required.
    tts_provider: str = os.getenv("TTS_PROVIDER", "cartesia")

    cartesia_api_key: str = os.getenv("CARTESIA_API_KEY", "")
    cartesia_model: str = "sonic-2"
    # Default Cartesia voice id (British female, "Helpful Woman"). Swap freely.
    cartesia_voice_id: str = os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22")

    # Deepgram Aura-2 model = voice, in one string: "aura-2-<voicename>-en".
    # Browse voices at https://developers.deepgram.com/docs/tts-models
    deepgram_tts_model: str = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")

    # --- LLM (Groq) ---
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "qwen/qwen3.6-27b"

    # --- Embeddings ---
    embedding_model_name: str = "minishlab/potion-retrieval-32M"
    embedding_dim: int = 512  # potion-retrieval-32M output dimensionality

    # --- Vector DB (Qdrant) ---
    qdrant_mode: str = os.getenv("QDRANT_MODE", "local")  # "local" (docker) or "cloud"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "pride_and_prejudice")

    # --- App database (users, profiles, conversations) ---
    db_path: str = os.getenv("DB_PATH", "data/app.db")

    # --- RAG chunking ---
    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 150
    retrieval_top_k: int = 5

    # --- Turn-taking ---
    # How many seconds of silence Silero VAD requires before it considers you
    # done talking (and hands off to smart-turn-v3 for the final call). The
    # library default (0.2s) is too twitchy for natural speech with pauses —
    # raise this if the agent still interrupts you; lower it if it feels
    # sluggish to respond after you actually finish.
    vad_stop_secs: float = float(os.getenv("VAD_STOP_SECS", "0.8"))

    # --- Server ---
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "7860"))

    def validate_runtime_keys(self) -> None:
        """Call this before starting the live pipeline (not needed for ingestion)."""
        required = [
            ("DEEPGRAM_API_KEY", self.deepgram_api_key),
            ("GROQ_API_KEY", self.groq_api_key),
        ]
        if self.tts_provider == "cartesia":
            required.append(("CARTESIA_API_KEY", self.cartesia_api_key))

        missing = [name for name, val in required if not val]
        if missing:
            raise RuntimeError(f"Missing required API keys in .env: {', '.join(missing)}")


settings = Settings()
