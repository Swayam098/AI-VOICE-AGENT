"""
Speech-to-Text Engine using faster-whisper.
Optimized for RTX 4050 (6GB VRAM) with CUDA acceleration.

Multilingual: Automatically detects Hindi and English speech.
"""

import asyncio
import io
import time
from typing import Optional

import numpy as np
from loguru import logger
import os

# Fix for [WinError 1455] The paging file is too small for this operation to complete
# This prevents PyTorch from loading all CUDA DLLs into virtual memory at once
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

from config import get_settings


class STTEngine:
    """
    Speech-to-Text engine powered by faster-whisper.
    Supports GPU acceleration and configurable model sizes.
    """

    def __init__(self):
        self.model = None
        self.settings = get_settings()
        self._initialized = False

    async def initialize(self):
        """Load the Whisper model (runs in executor to avoid blocking)."""
        if self._initialized:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self):
        """Synchronous model loading."""
        try:
            from faster_whisper import WhisperModel

            model_size = self.settings.whisper_model_size
            device = self.settings.whisper_device
            compute_type = self.settings.whisper_compute_type

            logger.info(f"Loading Whisper model: {model_size} on {device} ({compute_type})")

            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )

            self._initialized = True
            logger.success(f"Whisper model loaded successfully: {model_size}")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.info("Falling back to CPU with int8 compute type")

            try:
                from faster_whisper import WhisperModel

                self.model = WhisperModel(
                    self.settings.whisper_model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self._initialized = True
                logger.success("Whisper model loaded on CPU (fallback)")
            except Exception as e2:
                logger.error(f"Whisper model loading completely failed: {e2}")

    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Transcribe audio bytes to text with language detection.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)
            sample_rate: Audio sample rate (default 16000 Hz)
            language: Optional language code (e.g., 'en', 'hi').
                      If None, Whisper auto-detects between Hindi and English.

        Returns:
            Dict with 'text' and 'language' keys, or None if failed.
            Example: {'text': 'namaste', 'language': 'hi'}
        """
        if not self._initialized or not self.model:
            logger.warning("STT engine not initialized")
            return None

        if len(audio_data) < 1600:  # Less than 0.1s of audio at 16kHz
            return None

        # Use configured whisper language if no override provided
        if language is None:
            configured_lang = self.settings.whisper_language
            language = configured_lang if configured_lang else None  # None = auto-detect

        start_time = time.time()

        try:
            # Convert bytes to numpy float32 array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Run transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._transcribe_sync(audio_array, language),
            )

            elapsed = (time.time() - start_time) * 1000
            if result:
                logger.debug(
                    f"STT completed in {elapsed:.0f}ms "
                    f"[{result['language']}]: '{result['text'][:80]}'"
                )
            return result

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _transcribe_sync(
        self, audio_array: np.ndarray, language: Optional[str]
    ) -> Optional[dict]:
        """Synchronous transcription with language info."""
        segments, info = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        # Whisper detected language
        detected_lang = info.language if info else (language or "en")
        detected_lang_prob = info.language_probability if info else 0.0

        logger.debug(
            f"Whisper detected language: {detected_lang} "
            f"(probability: {detected_lang_prob:.2f})"
        )

        # Collect all segment texts
        texts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)

        full_text = " ".join(texts).strip()
        if not full_text:
            return None

        return {
            "text": full_text,
            "language": detected_lang,
            "language_probability": round(detected_lang_prob, 3),
        }

    async def transcribe_from_mulaw(
        self,
        mulaw_data: bytes,
        sample_rate: int = 8000,
    ) -> Optional[dict]:
        """
        Transcribe μ-law encoded audio (Twilio format).
        Decodes μ-law to PCM first, then transcribes.
        Returns dict with 'text' and 'language' keys.
        """
        try:
            import audioop

            # Decode μ-law to linear PCM (16-bit)
            pcm_data = audioop.ulaw2lin(mulaw_data, 2)

            # Resample from 8kHz (Twilio) to 16kHz (Whisper)
            pcm_data, _ = audioop.ratecv(pcm_data, 2, 1, sample_rate, 16000, None)

            return await self.transcribe(pcm_data, sample_rate=16000)

        except Exception as e:
            logger.error(f"μ-law transcription error: {e}")
            return None

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.model is not None
