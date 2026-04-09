"""
Text-to-Speech Engine using Piper TTS.
Fast, local speech synthesis optimized for real-time voice responses.
Falls back to a simple placeholder if Piper is not available.
"""

import asyncio
import io
import time
from typing import Optional

import numpy as np
from loguru import logger

from config import get_settings


class TTSEngine:
    """
    Text-to-Speech engine powered by Piper TTS.
    Produces raw PCM audio bytes for streaming back to the caller.
    Multilingual: Supports English and Hindi voices.
    """

    def __init__(self):
        self.voice_en = None
        self.voice_hi = None
        self.settings = get_settings()
        self._initialized = False
        self._sample_rate_en = 22050
        self._sample_rate_hi = 22050

    async def initialize(self):
        """Load the Piper TTS voice model."""
        if self._initialized:
            return

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self):
        """Synchronous model loading for both languages."""
        model_path_en = self.settings.piper_model_path
        model_path_hi = self.settings.piper_hindi_model_path

        try:
            from piper.voice import PiperVoice

            # Load English Model
            logger.info(f"Loading Piper TTS English model: {model_path_en}")
            try:
                self.voice_en = PiperVoice.load(model_path_en)
                self._sample_rate_en = self.voice_en.config.sample_rate
                logger.success(f"Piper TTS (EN) loaded — sample rate: {self._sample_rate_en}")
            except Exception as e:
                logger.warning(f"Failed to load English TTS model: {e}")

            # Load Hindi Model
            logger.info(f"Loading Piper TTS Hindi model: {model_path_hi}")
            try:
                self.voice_hi = PiperVoice.load(model_path_hi)
                self._sample_rate_hi = self.voice_hi.config.sample_rate
                logger.success(f"Piper TTS (HI) loaded — sample rate: {self._sample_rate_hi}")
            except Exception as e:
                logger.warning(f"Failed to load Hindi TTS model: {e}")

            # Mark initialized if at least one model loaded
            if self.voice_en or self.voice_hi:
                self._initialized = True
            else:
                logger.error("Failed to load any Piper TTS models.")
                self._initialized = False

        except ImportError:
            logger.warning("piper-tts not installed — TTS will use placeholder")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to load Piper TTS module: {e}")
            self._initialized = False

    async def synthesize(self, text: str, language: str = "en") -> bytes:
        """
        Synthesize text to raw PCM audio bytes.

        Args:
            text: The text to synthesize
            language: The language code ('en', 'hi', 'hi-en')

        Returns:
            Raw PCM audio bytes (16-bit, mono)
        """
        if not text.strip():
            return b""

        start_time = time.time()

        # Select the appropriate voice model based on language
        voice_to_use = self.voice_hi if language == "hi" else self.voice_en
        # Fallback if preferred voice is missing
        if not voice_to_use:
            voice_to_use = self.voice_en or self.voice_hi
            
        sample_rate = self._sample_rate_hi if voice_to_use == self.voice_hi else self._sample_rate_en

        if self._initialized and voice_to_use:
            try:
                loop = asyncio.get_event_loop()
                audio_bytes = await loop.run_in_executor(
                    None,
                    lambda: self._synthesize_sync(text, voice_to_use),
                )
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"TTS synthesized in {elapsed:.0f}ms ({len(audio_bytes)} bytes, lang: {language})")
                return audio_bytes

            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")

        # Fallback: return silence placeholder
        logger.debug("TTS fallback — returning silence")
        return self._generate_silence(duration_s=0.5, sample_rate=sample_rate)

    def _synthesize_sync(self, text: str, voice) -> bytes:
        """Synchronous speech synthesis."""
        audio_chunks = []

        for audio_bytes in voice.synthesize_stream_raw(text):
            audio_chunks.append(audio_bytes)

        return b"".join(audio_chunks)

    async def synthesize_to_mulaw(self, text: str, target_rate: int = 8000, language: str = "en") -> bytes:
        """
        Synthesize text and convert to μ-law encoding (Twilio format).

        Args:
            text: Text to synthesize
            target_rate: Target sample rate (8000 for Twilio)
            language: The language code ('en', 'hi', 'hi-en')

        Returns:
            μ-law encoded audio bytes
        """
        pcm_data = await self.synthesize(text, language=language)
        if not pcm_data:
            return b""

        try:
            import audioop

            # Determine original sample rate based on selected voice
            voice_to_use = self.voice_hi if language == "hi" else self.voice_en
            if not voice_to_use:
                voice_to_use = self.voice_en or self.voice_hi
            original_rate = self._sample_rate_hi if voice_to_use == self.voice_hi else self._sample_rate_en

            # Resample from Piper sample rate to Twilio's 8kHz
            if original_rate != target_rate:
                pcm_data, _ = audioop.ratecv(
                    pcm_data, 2, 1, original_rate, target_rate, None
                )

            # Encode to μ-law
            mulaw_data = audioop.lin2ulaw(pcm_data, 2)
            return mulaw_data

        except Exception as e:
            logger.error(f"μ-law conversion error: {e}")
            return b""

    def _generate_silence(self, duration_s: float = 1.0, sample_rate: int = 22050) -> bytes:
        """Generate silent audio bytes."""
        num_samples = int(sample_rate * duration_s)
        silence = np.zeros(num_samples, dtype=np.int16)
        return silence.tobytes()

    @property
    def is_ready(self) -> bool:
        return self._initialized and (self.voice_en is not None or self.voice_hi is not None)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_en
