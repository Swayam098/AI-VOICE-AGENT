"""
Smart Routing Engine — The Key Differentiator.
Now with multilingual support (Hindi/English/Hinglish).

Automatically classifies queries and routes them to the optimal intelligence source.

Routing Logic:
  IF real-time query → Web Search + Model
  ELIF complex query → Groq API (Cloud)
  ELIF short query   → Ollama Phi-3 (Lightweight)
  ELSE               → Ollama Mistral (Primary)
"""

import re
import time
from enum import Enum
from typing import Optional

from loguru import logger

from config import get_settings
from intelligence.language_detector import (
    LanguageDetector,
    Language,
    LanguageDetectionResult,
)


class QueryType(Enum):
    """Classification of query types for routing."""
    REALTIME = "realtime"    # → Web search + model
    COMPLEX = "complex"      # → Groq API
    NORMAL = "normal"        # → Ollama Mistral
    SHORT = "short"          # → Ollama Phi-3


class RoutingDecision:
    """Represents a routing decision with metadata."""

    def __init__(
        self,
        query_type: QueryType,
        confidence: float,
        reason: str,
        language: str = "en",
    ):
        self.query_type = query_type
        self.confidence = confidence
        self.reason = reason
        self.language = language
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "route": self.query_type.value,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "language": self.language,
            "timestamp": self.timestamp,
        }


# Keywords that indicate the user wants real-time / current information
REALTIME_KEYWORDS = {
    "latest", "today", "current", "now", "recent", "news",
    "weather", "price", "stock", "score", "update", "live",
    "happening", "right now", "this week", "this month",
    "trending", "breaking", "forecast", "tomorrow",
}

# Keywords that suggest complex reasoning is required
COMPLEXITY_KEYWORDS = {
    "explain", "analyze", "compare", "contrast", "evaluate",
    "describe in detail", "step by step", "pros and cons",
    "why does", "how does", "what would happen if",
    "theory", "algorithm", "architecture", "design",
    "write a", "create a", "build a", "implement",
    "debug", "fix", "solve", "calculate",
}

# Keywords indicating urgency (for escalation)
URGENCY_KEYWORDS = {
    "emergency", "urgent", "help me", "please help",
    "critical", "immediately", "asap", "right away",
    "danger", "accident", "911",
}


class SmartRouter:
    """
    Classifies incoming queries and routes them to the appropriate
    intelligence source for optimal speed, quality, and freshness.

    Now multilingual — detects Hindi, English, and Hinglish automatically.
    """

    def __init__(self):
        self.settings = get_settings()
        self.routing_history: list[dict] = []
        self.language_detector = LanguageDetector()

    async def initialize(self):
        """Initialize the language detector for multilingual routing."""
        await self.language_detector.initialize()

    def classify(
        self,
        text: str,
        conversation_history: list = None,
    ) -> RoutingDecision:
        """
        Classify a query and return a routing decision.
        Automatically detects language (Hindi/English/Hinglish).

        Priority order:
        1. Language detection
        2. Real-time detection (keyword-based — both languages)
        3. Complexity estimation (length + keywords)
        4. Short query detection (word count)
        5. Default → Normal (Mistral)

        Args:
            text: The user's query text
            conversation_history: Previous messages for context

        Returns:
            RoutingDecision with the chosen route and detected language
        """
        text_lower = text.lower().strip()
        word_count = len(text_lower.split())

        # --- Step 0: Detect language ---
        lang_result = self.language_detector.detect(text)
        detected_lang = lang_result.language.value

        logger.info(
            f"Language detected: {detected_lang} "
            f"(confidence: {lang_result.confidence:.2f}, "
            f"hindi_ratio: {lang_result.hindi_ratio:.2f})"
        )

        # --- Step 1: Check for real-time intent (both languages) ---
        if self.settings.enable_web_search:
            realtime_matches = self._check_realtime(text, text_lower, lang_result)
            if realtime_matches:
                confidence = min(0.95, 0.6 + 0.1 * len(realtime_matches))
                decision = RoutingDecision(
                    QueryType.REALTIME,
                    confidence,
                    f"Real-time keywords detected: {', '.join(realtime_matches[:3])}",
                    language=detected_lang,
                )
                self._log_decision(text, decision)
                return decision

        # --- Step 2: Check for complexity (both languages) ---
        complexity_score = self._compute_complexity(
            text, text_lower, word_count, conversation_history, lang_result
        )

        if complexity_score >= 0.7:
            if self.settings.groq_configured:
                decision = RoutingDecision(
                    QueryType.COMPLEX,
                    complexity_score,
                    f"High complexity score: {complexity_score:.2f} (words: {word_count})",
                    language=detected_lang,
                )
                self._log_decision(text, decision)
                return decision

        # --- Step 3: Check for short/simple query ---
        if word_count <= self.settings.router_short_query_max_words:
            if self.settings.enable_local_models:
                decision = RoutingDecision(
                    QueryType.SHORT,
                    0.85,
                    f"Short query ({word_count} words ≤ {self.settings.router_short_query_max_words})",
                    language=detected_lang,
                )
                self._log_decision(text, decision)
                return decision

        # --- Step 4: Default → Normal (Mistral) ---
        if self.settings.enable_local_models:
            decision = RoutingDecision(
                QueryType.NORMAL,
                0.80,
                f"Default routing — standard query ({word_count} words)",
                language=detected_lang,
            )
        else:
            decision = RoutingDecision(
                QueryType.COMPLEX,
                0.70,
                "Local models disabled — using cloud fallback",
                language=detected_lang,
            )

        self._log_decision(text, decision)
        return decision

    def _check_realtime(
        self,
        text: str,
        text_lower: str,
        lang_result: LanguageDetectionResult,
    ) -> list[str]:
        """Check for real-time keywords in both English and Hindi."""
        matches = []

        # English keywords
        matches.extend([kw for kw in REALTIME_KEYWORDS if kw in text_lower])

        # Hindi keywords (Devanagari + romanized)
        if lang_result.language in (Language.HINDI, Language.HINGLISH):
            hindi_matches = self.language_detector.get_realtime_keywords(text)
            matches.extend(hindi_matches)

        return list(set(matches))  # deduplicate

    def _compute_complexity(
        self,
        text: str,
        text_lower: str,
        word_count: int,
        history: list = None,
        lang_result: LanguageDetectionResult = None,
    ) -> float:
        """
        Compute a complexity score (0.0 - 1.0) for the query.
        Supports both English and Hindi complexity markers.
        """
        score = 0.0

        # Factor 1: Word count contribution
        if word_count >= self.settings.router_complex_min_words:
            score += 0.4
        elif word_count >= 30:
            score += 0.2
        elif word_count >= 15:
            score += 0.1

        # Factor 2: English complexity keywords
        complexity_matches = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in text_lower)
        score += min(0.3, complexity_matches * 0.1)

        # Factor 3: Hindi complexity keywords
        if lang_result and lang_result.language in (Language.HINDI, Language.HINGLISH):
            hindi_matches = self.language_detector.get_complexity_keywords(text)
            score += min(0.3, len(hindi_matches) * 0.12)

        # Factor 4: Multi-part questions
        question_marks = text.count("?")
        conjunctions = len(re.findall(r'\b(and|also|additionally|furthermore|moreover)\b', text_lower))
        # Hindi conjunctions
        hindi_conjunctions = len(re.findall(r'(और|तथा|साथ ही|इसके अलावा)', text))
        if question_marks > 1 or conjunctions > 1 or hindi_conjunctions > 1:
            score += 0.15

        # Factor 5: Conversation context depth
        if history and len(history) > 6:
            score += 0.1

        return min(1.0, score)

    def check_urgency(self, text: str) -> tuple[bool, float]:
        """
        Check if the text contains urgency indicators.
        Supports both English and Hindi urgency keywords.

        Returns:
            (is_urgent: bool, urgency_score: float)
        """
        text_lower = text.lower()
        score = 0.0

        # English urgency
        eng_matches = [kw for kw in URGENCY_KEYWORDS if kw in text_lower]
        if eng_matches:
            score += min(1.0, 0.5 + 0.15 * len(eng_matches))

        # Hindi urgency
        hindi_urgent, hindi_score = self.language_detector.get_urgency_keywords(text)
        score = max(score, hindi_score)

        return (score >= 0.5, min(1.0, score))

    def _log_decision(self, text: str, decision: RoutingDecision):
        """Log routing decision for analytics."""
        entry = {
            "query_preview": text[:100],
            **decision.to_dict(),
        }
        self.routing_history.append(entry)
        logger.info(
            f"Route: {decision.query_type.value} "
            f"(conf: {decision.confidence:.2f}, lang: {decision.language}) "
            f"— {decision.reason}"
        )

    def get_stats(self) -> dict:
        """Get routing distribution statistics including language breakdown."""
        if not self.routing_history:
            return {"total": 0, "distribution": {}, "languages": {}}

        total = len(self.routing_history)
        distribution = {}
        languages = {}

        for entry in self.routing_history:
            route = entry["route"]
            lang = entry.get("language", "en")
            distribution[route] = distribution.get(route, 0) + 1
            languages[lang] = languages.get(lang, 0) + 1

        return {
            "total": total,
            "distribution": {k: round(v / total * 100, 1) for k, v in distribution.items()},
            "counts": distribution,
            "languages": {k: round(v / total * 100, 1) for k, v in languages.items()},
            "language_counts": languages,
        }
