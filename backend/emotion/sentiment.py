"""
Text-Based Sentiment Analysis (Phase 1).
Analyzes transcribed text for emotional tone and urgency.
Multilingual: Supports both Hindi and English urgency detection.
"""

import re
from typing import Optional

from loguru import logger


class SentimentAnalyzer:
    """
    Analyzes text sentiment using TextBlob.
    Provides polarity, subjectivity, and urgency scoring.
    """

    def __init__(self):
        self.analyzer = None
        self._initialized = False

    async def initialize(self):
        """Initialize the sentiment analysis engine."""
        try:
            from textblob import TextBlob

            self.analyzer = TextBlob
            self._initialized = True
            logger.success("Sentiment analyzer initialized")
        except ImportError:
            logger.warning("textblob not installed — sentiment analysis disabled")
            self._initialized = False

    def analyze(self, text: str) -> dict:
        """
        Analyze the sentiment of a text string.

        Args:
            text: The text to analyze

        Returns:
            Dict with:
            - polarity: -1.0 (negative) to 1.0 (positive)
            - subjectivity: 0.0 (objective) to 1.0 (subjective)
            - mood: 'positive', 'negative', or 'neutral'
            - urgency: 0.0 to 1.0
            - is_urgent: bool
        """
        if not self._initialized or not text.strip():
            return self._default_result()

        try:
            blob = self.analyzer(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Determine mood
            if polarity > 0.1:
                mood = "positive"
            elif polarity < -0.1:
                mood = "negative"
            else:
                mood = "neutral"

            # Check urgency
            urgency_score = self._calculate_urgency(text)
            is_urgent = urgency_score >= 0.6

            return {
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "mood": mood,
                "urgency": round(urgency_score, 3),
                "is_urgent": is_urgent,
            }

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return self._default_result()

    def _calculate_urgency(self, text: str) -> float:
        """
        Calculate urgency score based on keyword presence and text features.
        Supports both English and Hindi urgency keywords.

        Returns:
            Urgency score from 0.0 to 1.0
        """
        text_lower = text.lower()
        score = 0.0

        # English urgent keywords and their weights
        urgent_keywords = {
            "emergency": 0.4,
            "urgent": 0.3,
            "help": 0.2,
            "please help": 0.35,
            "critical": 0.3,
            "immediately": 0.3,
            "asap": 0.25,
            "right away": 0.25,
            "danger": 0.4,
            "accident": 0.4,
            "911": 0.5,
            "dying": 0.5,
            "hospital": 0.3,
            "police": 0.3,
            "fire": 0.3,
        }

        for keyword, weight in urgent_keywords.items():
            if keyword in text_lower:
                score += weight

        # Hindi urgent keywords (Devanagari)
        hindi_urgency = {
            "मदद": 0.35,
            "बचाओ": 0.45,
            "जल्दी": 0.25,
            "तुरंत": 0.3,
            "खतरा": 0.4,
            "आपातकाल": 0.5,
            "एम्बुलेंस": 0.45,
            "पुलिस": 0.35,
            "अस्पताल": 0.35,
            "आग": 0.3,
            "दुर्घटना": 0.4,
        }

        for keyword, weight in hindi_urgency.items():
            if keyword in text:
                score += weight

        # Romanized Hindi urgency keywords
        hindi_romanized_urgency = {
            "madad": 0.35,
            "bachao": 0.45,
            "jaldi": 0.25,
            "turant": 0.3,
            "khatara": 0.4,
            "ambulance": 0.4,
        }

        for keyword, weight in hindi_romanized_urgency.items():
            if keyword in text_lower:
                score += weight

        # Exclamation marks increase urgency
        exclamations = text.count("!")
        score += min(0.2, exclamations * 0.05)

        # ALL CAPS words increase urgency
        caps_words = len(re.findall(r"\b[A-Z]{2,}\b", text))
        score += min(0.15, caps_words * 0.05)

        return min(1.0, score)

    def analyze_conversation_sentiment(self, messages: list[dict]) -> dict:
        """
        Analyze the overall sentiment of a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Overall conversation sentiment summary
        """
        if not messages:
            return self._default_result()

        # Analyze only user messages
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return self._default_result()

        sentiments = [self.analyze(m["content"]) for m in user_messages]

        avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)
        avg_urgency = sum(s["urgency"] for s in sentiments) / len(sentiments)
        max_urgency = max(s["urgency"] for s in sentiments)

        # Overall mood
        if avg_polarity > 0.1:
            overall_mood = "positive"
        elif avg_polarity < -0.1:
            overall_mood = "negative"
        else:
            overall_mood = "neutral"

        return {
            "overall_polarity": round(avg_polarity, 3),
            "overall_mood": overall_mood,
            "average_urgency": round(avg_urgency, 3),
            "max_urgency": round(max_urgency, 3),
            "is_urgent": max_urgency >= 0.6,
            "message_count": len(user_messages),
        }

    def _default_result(self) -> dict:
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "mood": "neutral",
            "urgency": 0.0,
            "is_urgent": False,
        }

    @property
    def is_ready(self) -> bool:
        return self._initialized
