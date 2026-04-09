"""
Intelligence Pipeline — Orchestrates the full query processing flow.
Ties together Smart Router, Local Models, Cloud Models, Web Search,
Emotion Detection, and Language Detection.

Now fully multilingual — supports Hindi, English, and Hinglish.
"""

import time
from typing import Optional

from loguru import logger

from intelligence.router import SmartRouter, QueryType
from intelligence.local_model import LocalModelEngine
from intelligence.cloud_model import CloudModelEngine
from intelligence.web_search import WebSearchEngine
from intelligence.prompt_templates import build_system_prompt
from intelligence.language_detector import LanguageDetector, Language
from emotion.sentiment import SentimentAnalyzer
from config import get_settings


class IntelligencePipeline:
    """
    The central orchestrator that routes queries through the appropriate
    intelligence source and returns responses.

    Multilingual: Automatically detects language and responds in kind.
    """

    def __init__(
        self,
        router: SmartRouter,
        local_model: LocalModelEngine,
        cloud_model: CloudModelEngine,
        web_search: WebSearchEngine,
        sentiment: SentimentAnalyzer,
    ):
        self.router = router
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.web_search = web_search
        self.sentiment = sentiment
        self.settings = get_settings()

    async def process(
        self,
        text: str,
        conversation_history: list = None,
        session_id: str = "",
        caller_context: str = "",
    ) -> dict:
        """
        Process a user query through the full intelligence pipeline.

        Flow:
        1. Classify query via Smart Router (includes language detection)
        2. (Optional) Check sentiment/urgency
        3. Route to appropriate model with language-aware prompts
        4. Return response with metadata

        Args:
            text: User's query text
            conversation_history: Previous messages
            session_id: Current session identifier
            caller_context: Info about the caller

        Returns:
            Dict with 'text', 'route', 'model', 'latency_ms', 'language', 'sentiment'
        """
        start_time = time.time()

        # Step 1: Classify the query (includes language detection)
        decision = self.router.classify(text, conversation_history)
        route = decision.query_type
        detected_language = decision.language  # 'en', 'hi', or 'hi-en'

        # Step 2: Check sentiment (if enabled)
        sentiment_result = None
        if self.settings.enable_emotion_detection and self.sentiment.is_ready:
            sentiment_result = self.sentiment.analyze(text)

            # Check for urgency — could trigger escalation
            if sentiment_result.get("is_urgent"):
                logger.warning(f"[{session_id}] URGENT query detected! (lang: {detected_language})")

        # Step 3: Route to appropriate intelligence source (with language)
        response = await self._route_query(
            text=text,
            route=route,
            conversation_history=conversation_history or [],
            caller_context=caller_context,
            language=detected_language,
        )

        total_latency = int((time.time() - start_time) * 1000)

        result = {
            "text": response.get("text", "I'm sorry, I couldn't process that."),
            "route": route.value,
            "model": response.get("model", "unknown"),
            "latency_ms": total_latency,
            "language": detected_language,
            "routing_confidence": decision.confidence,
            "routing_reason": decision.reason,
        }

        if sentiment_result:
            result["sentiment"] = sentiment_result

        logger.info(
            f"[{session_id}] Pipeline complete — "
            f"route: {route.value}, model: {response.get('model')}, "
            f"lang: {detected_language}, latency: {total_latency}ms"
        )

        return result

    async def _route_query(
        self,
        text: str,
        route: QueryType,
        conversation_history: list,
        caller_context: str,
        language: str = "en",
    ) -> dict:
        """Route the query to the appropriate intelligence source."""

        if route == QueryType.REALTIME:
            return await self._handle_realtime(text, conversation_history, caller_context, language)

        elif route == QueryType.COMPLEX:
            return await self._handle_complex(text, conversation_history, caller_context, language)

        elif route == QueryType.SHORT:
            return await self._handle_short(text, conversation_history, caller_context, language)

        else:  # NORMAL
            return await self._handle_normal(text, conversation_history, caller_context, language)

    async def _handle_realtime(
        self, text: str, history: list, caller_context: str, language: str
    ) -> dict:
        """Handle real-time queries: Web Search + Model."""
        # Step 1: Fetch web search results
        search_results = []
        search_context = ""

        if self.web_search.is_ready:
            search_results = await self.web_search.search(text, max_results=3)
            search_context = self.web_search.format_results_for_prompt(search_results)

        # Step 2: Build prompt with search context AND language instruction
        system_prompt = build_system_prompt(
            route="realtime",
            caller_context=caller_context,
            search_results=search_context,
            language=language,
        )

        # Step 3: Use cloud model if available (better at synthesizing search results)
        if self.cloud_model.is_ready:
            return await self.cloud_model.generate(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
            )
        elif self.local_model.is_ready:
            return await self.local_model.generate(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
            )
        else:
            # Last resort: return raw search results
            if search_results:
                summary = "; ".join([r["snippet"][:100] for r in search_results[:2]])
                if language == "hi":
                    return {"text": f"ये मिला मुझे: {summary}", "model": "search-only"}
                return {"text": f"Here's what I found: {summary}", "model": "search-only"}
            if language == "hi":
                return {"text": "मुझे इस बारे में अभी कोई जानकारी नहीं मिल पाई।", "model": "none"}
            return {"text": "I couldn't find current information on that.", "model": "none"}

    async def _handle_complex(
        self, text: str, history: list, caller_context: str, language: str
    ) -> dict:
        """Handle complex queries: Cloud Model (Groq)."""
        system_prompt = build_system_prompt(
            route="complex",
            caller_context=caller_context,
            language=language,
        )

        if self.cloud_model.is_ready:
            return await self.cloud_model.generate(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
                max_tokens=1024,
            )
        elif self.local_model.is_ready:
            # Fallback to primary local model
            return await self.local_model.generate_with_primary(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
            )
        else:
            if language == "hi":
                return {"text": "इस समय कोई मॉडल उपलब्ध नहीं है।", "model": "none"}
            return {"text": "No models available for complex processing.", "model": "none"}

    async def _handle_normal(
        self, text: str, history: list, caller_context: str, language: str
    ) -> dict:
        """Handle normal queries: Ollama Mistral."""
        system_prompt = build_system_prompt(
            route="normal",
            caller_context=caller_context,
            language=language,
        )

        if self.local_model.is_ready:
            return await self.local_model.generate_with_primary(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
            )
        elif self.cloud_model.is_ready:
            # Fallback to cloud
            return await self.cloud_model.generate(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
            )
        else:
            if language == "hi":
                return {"text": "कोई मॉडल उपलब्ध नहीं है।", "model": "none"}
            return {"text": "No models available for processing.", "model": "none"}

    async def _handle_short(
        self, text: str, history: list, caller_context: str, language: str
    ) -> dict:
        """Handle short queries: Ollama Phi-3."""
        system_prompt = build_system_prompt(
            route="short",
            caller_context=caller_context,
            language=language,
        )

        if self.local_model.is_ready:
            return await self.local_model.generate_with_lightweight(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
            )
        elif self.cloud_model.is_ready:
            return await self.cloud_model.generate(
                prompt=text,
                system_prompt=system_prompt,
                conversation_history=history,
                max_tokens=256,
            )
        else:
            if language == "hi":
                return {"text": "कोई मॉडल उपलब्ध नहीं है।", "model": "none"}
            return {"text": "No models available.", "model": "none"}
