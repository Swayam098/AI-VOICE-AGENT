"""
Cloud Model Interface — Groq API.
High-quality, low-latency cloud LLM for complex queries.
"""

import asyncio
import time
from typing import Optional

from loguru import logger

from config import get_settings


class CloudModelEngine:
    """
    Interface to Groq's ultra-fast LLM inference API.
    Used for complex queries that benefit from larger models.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialized = False
        self.default_model = "llama-3.3-70b-versatile"
        self.total_tokens_used = 0

    async def initialize(self):
        """Initialize Groq client."""
        if not self.settings.groq_configured:
            logger.warning("Groq API key not configured — cloud model unavailable")
            return

        try:
            from groq import Groq

            self.client = Groq(api_key=self.settings.groq_api_key)
            self._initialized = True
            logger.success("Groq cloud model initialized")

        except ImportError:
            logger.error("groq package not installed — run: pip install groq")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        conversation_history: list = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> dict:
        """
        Generate a response from Groq's cloud LLM.

        Args:
            prompt: User's message
            system_prompt: System instructions
            conversation_history: Previous messages
            model: Groq model name
            max_tokens: Maximum response tokens
            temperature: Creativity (0.0 - 1.0)

        Returns:
            Dict with 'text', 'model', 'latency_ms', 'tokens_used'
        """
        if not self._initialized:
            return {
                "text": "Cloud model not available. Please configure GROQ_API_KEY.",
                "model": "none",
                "latency_ms": 0,
                "tokens_used": 0,
            }

        model = model or self.default_model
        start_time = time.time()

        # Build message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if conversation_history:
            for msg in conversation_history[-8:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        if not messages or messages[-1].get("content") != prompt:
            messages.append({"role": "user", "content": prompt})

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._chat_sync(model, messages, max_tokens, temperature),
            )

            latency = int((time.time() - start_time) * 1000)

            # Extract response
            text = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0
            self.total_tokens_used += tokens

            logger.debug(
                f"Groq ({model}) responded in {latency}ms — {tokens} tokens"
            )

            return {
                "text": text,
                "model": model,
                "latency_ms": latency,
                "tokens_used": tokens,
            }

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {
                "text": f"Cloud processing error: {str(e)[:100]}",
                "model": model,
                "latency_ms": int((time.time() - start_time) * 1000),
                "tokens_used": 0,
                "error": str(e),
            }

    def _chat_sync(self, model: str, messages: list, max_tokens: int, temperature: float):
        """Synchronous Groq chat completion."""
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def get_usage_stats(self) -> dict:
        """Get total API usage statistics."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "model": self.default_model,
            "configured": self._initialized,
        }

    @property
    def is_ready(self) -> bool:
        return self._initialized
