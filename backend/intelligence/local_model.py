"""
Local Model Interface — Ollama (Mistral & Phi-3).
GPU-optimized local inference for fast, offline reasoning.
"""

import asyncio
import time
from typing import Optional

from loguru import logger

from config import get_settings


class LocalModelEngine:
    """
    Interface to local LLMs via Ollama.
    Supports Mistral (primary) and Phi-3 (lightweight).
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialized = False

    async def initialize(self):
        """Initialize Ollama client and verify model availability."""
        try:
            import ollama

            self.client = ollama
            
            # Verify connection and models
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(None, self._list_models)

            model_names = [m.get("name", "").split(":")[0] for m in models]
            logger.info(f"Ollama models available: {model_names}")

            primary = self.settings.ollama_primary_model
            lightweight = self.settings.ollama_lightweight_model

            if primary not in model_names and f"{primary}" not in str(models):
                logger.warning(f"Primary model '{primary}' not found in Ollama")
            if lightweight not in model_names and f"{lightweight}" not in str(models):
                logger.warning(f"Lightweight model '{lightweight}' not found in Ollama")

            self._initialized = True
            logger.success("Ollama local models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.info("Make sure Ollama is running: 'ollama serve'")
            self._initialized = False

    def _list_models(self) -> list:
        """List available Ollama models (sync)."""
        try:
            response = self.client.list()
            # Handle both old and new ollama SDK response formats
            if hasattr(response, "models"):
                return [{"name": m.model} for m in response.models]
            elif isinstance(response, dict) and "models" in response:
                return response["models"]
            return []
        except Exception:
            return []

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: str = "",
        conversation_history: list = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict:
        """
        Generate a response from a local model.

        Args:
            prompt: User's message
            model: Model name (defaults to primary model)
            system_prompt: System instructions
            conversation_history: Previous messages
            max_tokens: Maximum response tokens
            temperature: Creativity (0.0 - 1.0)

        Returns:
            Dict with 'text', 'model', 'latency_ms'
        """
        if not self._initialized:
            return {"text": "Local model not available.", "model": "none", "latency_ms": 0}

        model = model or self.settings.ollama_primary_model
        start_time = time.time()

        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if conversation_history:
            # Include last 6 messages for context
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
        else:
            messages.append({"role": "user", "content": prompt})

        # If the last message in history isn't the current prompt, add it
        if messages and messages[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._chat_sync(model, messages, max_tokens, temperature),
            )

            latency = int((time.time() - start_time) * 1000)

            # Extract response text
            if isinstance(response, dict):
                text = response.get("message", {}).get("content", "")
            elif hasattr(response, "message"):
                text = response.message.content
            else:
                text = str(response)

            logger.debug(f"Local model ({model}) responded in {latency}ms")

            return {
                "text": text.strip(),
                "model": model,
                "latency_ms": latency,
            }

        except Exception as e:
            logger.error(f"Local model error ({model}): {e}")
            return {
                "text": f"I'm having trouble with my local processing. Error: {str(e)[:100]}",
                "model": model,
                "latency_ms": int((time.time() - start_time) * 1000),
                "error": str(e),
            }

    def _chat_sync(self, model: str, messages: list, max_tokens: int, temperature: float):
        """Synchronous chat completion."""
        return self.client.chat(
            model=model,
            messages=messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )

    async def generate_with_lightweight(self, prompt: str, **kwargs) -> dict:
        """Generate using the lightweight model (Phi-3)."""
        return await self.generate(
            prompt=prompt,
            model=self.settings.ollama_lightweight_model,
            max_tokens=256,  # Shorter for lightweight
            **kwargs,
        )

    async def generate_with_primary(self, prompt: str, **kwargs) -> dict:
        """Generate using the primary model (Mistral)."""
        return await self.generate(
            prompt=prompt,
            model=self.settings.ollama_primary_model,
            **kwargs,
        )

    @property
    def is_ready(self) -> bool:
        return self._initialized
