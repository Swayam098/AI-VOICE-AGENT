"""
Configuration module for the AI Voice Call Agent.
Uses Pydantic Settings for validated environment-based configuration.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache

# Project root is one level above the backend/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Twilio ---
    twilio_account_sid: str = Field(default="", description="Twilio Account SID")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token")
    twilio_phone_number: str = Field(default="", description="Twilio Phone Number")

    # --- Groq (Cloud LLM) ---
    groq_api_key: str = Field(default="", description="Groq API Key")

    # --- Supabase ---
    supabase_url: str = Field(default="", description="Supabase Project URL")
    supabase_key: str = Field(default="", description="Supabase Anon Key")
    supabase_publishable_key: str = Field(default="", description="Supabase Publishable Key")
    supabase_service_key: str = Field(default="", description="Supabase Service Role Key")

    # --- Ollama (Local LLM) ---
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_primary_model: str = Field(default="mistral", description="Primary local model")
    ollama_lightweight_model: str = Field(default="phi3:mini", description="Lightweight local model")

    # --- Speech ---
    whisper_model_size: str = Field(default="small", description="Whisper model size: tiny/small/medium")
    whisper_device: str = Field(default="cuda", description="Whisper device: cuda/cpu")
    whisper_compute_type: str = Field(default="float16", description="Compute type for Whisper")
    whisper_language: str = Field(default="", description="Whisper language code — leave empty for auto-detect (Hindi/English)")
    piper_model_path: str = Field(default="models/piper/en_US-lessac-medium.onnx", description="Path to English Piper TTS model")
    piper_hindi_model_path: str = Field(default="models/piper/hi_IN-pratham-medium.onnx", description="Path to Hindi Piper TTS model")

    # --- Language / Multilingual ---
    supported_languages: str = Field(default="en,hi,hi-en", description="Supported languages: en, hi, hi-en (Hinglish)")
    default_language: str = Field(default="en", description="Default response language")

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="info", description="Logging level")

    # --- ngrok ---
    ngrok_auth_token: str = Field(default="", description="ngrok auth token for tunneling")

    # --- Smart Router Thresholds ---
    router_short_query_max_words: int = Field(default=10, description="Max words for a 'short' query")
    router_complex_min_words: int = Field(default=50, description="Min words to consider a query 'complex'")
    router_latency_timeout_ms: int = Field(default=5000, description="Max latency before fallback (ms)")

    # --- Feature Flags ---
    enable_local_models: bool = Field(default=True, description="Enable local Ollama models")
    enable_web_search: bool = Field(default=True, description="Enable web search for real-time queries")
    enable_emotion_detection: bool = Field(default=True, description="Enable text-based sentiment analysis")

    @property
    def twilio_configured(self) -> bool:
        """Check if Twilio credentials are set."""
        return bool(self.twilio_account_sid and self.twilio_auth_token)

    @property
    def groq_configured(self) -> bool:
        """Check if Groq API is configured."""
        return bool(self.groq_api_key)

    @property
    def supabase_configured(self) -> bool:
        """Check if Supabase is configured."""
        return bool(self.supabase_url and self.supabase_key)

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
