"""
Language Detection Engine — Hindi / English / Hinglish.
Automatically detects the user's language and provides metadata for routing.

Supported Languages:
  - English (en)
  - Hindi (hi)
  - Hinglish (hi-en) — mixed Hindi + English

Uses Unicode script detection (Devanagari) combined with keyword-based heuristics.
No external API calls — runs 100% locally with zero latency.
"""

import re
import unicodedata
from enum import Enum
from typing import Optional

from loguru import logger


class Language(Enum):
    """Detected language codes."""
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hi-en"
    UNKNOWN = "unknown"


class LanguageDetectionResult:
    """Result of a language detection operation."""

    def __init__(
        self,
        language: Language,
        confidence: float,
        hindi_ratio: float,
        english_ratio: float,
        script_info: str,
    ):
        self.language = language
        self.confidence = confidence
        self.hindi_ratio = hindi_ratio
        self.english_ratio = english_ratio
        self.script_info = script_info

    def to_dict(self) -> dict:
        return {
            "language": self.language.value,
            "confidence": round(self.confidence, 2),
            "hindi_ratio": round(self.hindi_ratio, 2),
            "english_ratio": round(self.english_ratio, 2),
            "script_info": self.script_info,
        }


# ─── Common Hindi Words (Romanized) for Hinglish Detection ───
HINDI_ROMANIZED_WORDS = {
    # Greetings & common phrases
    "namaste", "namaskar", "dhanyavaad", "dhanyawad", "shukriya",
    "alvida", "phir milenge", "theek", "accha", "achha",
    # Pronouns & particles
    "kya", "kaise", "kaisa", "kahan", "kab", "kyun", "kyu",
    "mujhe", "mera", "meri", "mere", "tumhara", "tumhari",
    "uska", "uski", "unka", "unki", "hamara", "hamari",
    "aap", "tum", "hum", "woh", "yeh", "ye",
    # Verbs
    "hai", "hain", "tha", "thi", "hoga", "hogi",
    "karo", "karna", "karenge", "karega", "karegi",
    "bolo", "bolna", "bata", "batao", "batana",
    "dekho", "dekhna", "suno", "sunna", "jao", "jana",
    "aao", "aana", "chalo", "chaliye", "ruko", "rukna",
    "samajh", "samjho", "samajhna", "padhna", "likhna",
    "khana", "peena", "sona", "jaagna",
    # Common words
    "bahut", "bohot", "zyada", "kam", "thoda", "thodi",
    "acha", "bura", "nahi", "nahin", "haan", "ji",
    "lekin", "par", "aur", "ya", "phir", "abhi",
    "yahan", "wahan", "idhar", "udhar",
    "kuch", "sab", "sabhi", "koi", "kaun",
    "paisa", "paise", "rupaye", "rupee",
    "beta", "bhai", "didi", "behen", "chacha",
    "matlab", "iska", "wala", "wali", "wale",
    "bilkul", "zaroor", "jaroor", "pakka",
    # Question patterns
    "kaise ho", "kya haal", "kidhar", "kitna", "kitne", "kitni",
    # Sentence particles
    "na", "naa", "re", "yaar", "arre", "arrey",
    # Time-related
    "aaj", "kal", "parso", "abhi", "baad",
    # Urgency
    "jaldi", "turant", "fauran", "madad",
}

# Hindi urgency keywords (Devanagari script)
HINDI_URGENCY_KEYWORDS = {
    "मदद": 0.35,          # help
    "बचाओ": 0.45,         # save me
    "जल्दी": 0.25,         # quickly
    "तुरंत": 0.3,          # immediately
    "खतरा": 0.4,           # danger
    "आपातकाल": 0.5,       # emergency
    "एम्बुलेंस": 0.45,    # ambulance
    "पुलिस": 0.35,         # police
    "अस्पताल": 0.35,      # hospital
    "आग": 0.3,             # fire
    "दुर्घटना": 0.4,       # accident
    "कृपया मदद करें": 0.45, # please help
}

# Romanized Hindi urgency keywords
HINDI_URGENCY_ROMANIZED = {
    "madad": 0.35,
    "bachao": 0.45,
    "jaldi": 0.25,
    "turant": 0.3,
    "khatara": 0.4,
    "emergency": 0.4,
    "ambulance": 0.4,
    "police": 0.35,
    "hospital": 0.35,
    "aag": 0.3,
}

# Hindi real-time keywords
HINDI_REALTIME_KEYWORDS = {
    # Devanagari
    "आज", "अभी", "ताज़ा", "नवीनतम", "समाचार", "मौसम",
    "कीमत", "दाम", "स्कोर", "लाइव",
    # Romanized
    "aaj", "abhi", "taaza", "taza", "samachar", "news",
    "mausam", "keemat", "daam", "score", "live",
    "aaj ka", "abhi ka", "latest",
}

# Hindi complexity keywords
HINDI_COMPLEXITY_KEYWORDS = {
    # Devanagari
    "समझाइए", "विश्लेषण", "तुलना", "विस्तार से",
    "कदम दर कदम", "फायदे और नुकसान",
    # Romanized
    "samjhaiye", "samjhao", "vishleshan", "tulna",
    "vistar se", "detail mein", "step by step",
    "fayde aur nuksan",
}


class LanguageDetector:
    """
    Detects language from text using Unicode analysis and keyword matching.
    Zero-dependency — no API calls, no ML model needed.
    """

    def __init__(self):
        self._initialized = False

    async def initialize(self):
        """Initialize the language detector."""
        self._initialized = True
        logger.success("Language detector initialized (Hindi/English/Hinglish)")

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect the language of the input text.

        Strategy:
        1. Check for Devanagari script (→ Hindi or Hinglish)
        2. Check for romanized Hindi words (→ Hinglish)
        3. Default to English

        Args:
            text: Input text to analyze

        Returns:
            LanguageDetectionResult with detected language and metadata
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                Language.UNKNOWN, 0.0, 0.0, 0.0, "empty input"
            )

        text = text.strip()
        chars = [c for c in text if not c.isspace() and not c.isdigit() and c not in '.,!?;:\'"()-']

        if not chars:
            return LanguageDetectionResult(
                Language.ENGLISH, 0.5, 0.0, 1.0, "only punctuation/numbers"
            )

        # Count Devanagari vs Latin characters
        devanagari_count = sum(1 for c in chars if self._is_devanagari(c))
        latin_count = sum(1 for c in chars if c.isascii() and c.isalpha())
        total_alpha = devanagari_count + latin_count

        if total_alpha == 0:
            return LanguageDetectionResult(
                Language.UNKNOWN, 0.3, 0.0, 0.0, "no alphabetic characters"
            )

        hindi_ratio = devanagari_count / total_alpha
        english_ratio = latin_count / total_alpha

        # Check for romanized Hindi words
        romanized_hindi_score = self._check_romanized_hindi(text)

        # ─── Decision Logic ───

        # Case 1: Predominantly Devanagari script → Hindi
        if hindi_ratio >= 0.7:
            return LanguageDetectionResult(
                Language.HINDI,
                min(0.95, 0.6 + hindi_ratio * 0.3),
                hindi_ratio,
                english_ratio,
                "predominantly Devanagari script",
            )

        # Case 2: Mixed Devanagari + Latin → Hinglish
        if hindi_ratio >= 0.2 and english_ratio >= 0.2:
            return LanguageDetectionResult(
                Language.HINGLISH,
                0.85,
                hindi_ratio,
                english_ratio,
                "mixed Devanagari and Latin script",
            )

        # Case 3: Small amount of Devanagari with Latin → Hinglish
        if hindi_ratio > 0.05:
            return LanguageDetectionResult(
                Language.HINGLISH,
                0.75,
                hindi_ratio,
                english_ratio,
                "some Devanagari with Latin text",
            )

        # Case 4: All Latin but contains romanized Hindi → Hinglish
        if romanized_hindi_score >= 0.3:
            return LanguageDetectionResult(
                Language.HINGLISH,
                min(0.85, 0.5 + romanized_hindi_score * 0.4),
                0.0,
                english_ratio,
                f"romanized Hindi detected (score: {romanized_hindi_score:.2f})",
            )

        # Case 5: Default → English
        return LanguageDetectionResult(
            Language.ENGLISH,
            min(0.95, 0.7 + english_ratio * 0.2),
            hindi_ratio,
            english_ratio,
            "Latin script, no Hindi markers",
        )

    def _is_devanagari(self, char: str) -> bool:
        """Check if a character is in the Devanagari Unicode block."""
        try:
            return unicodedata.name(char, "").startswith("DEVANAGARI")
        except (TypeError, ValueError):
            return False

    def _check_romanized_hindi(self, text: str) -> float:
        """
        Check for romanized Hindi words in Latin-script text.
        Returns a score from 0.0 to 1.0.
        """
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]+\b', text_lower)

        if not words:
            return 0.0

        hindi_word_count = sum(1 for w in words if w in HINDI_ROMANIZED_WORDS)
        return min(1.0, hindi_word_count / max(1, len(words)))

    def get_urgency_keywords(self, text: str) -> tuple[bool, float]:
        """
        Check for Hindi urgency keywords (both scripts).

        Returns:
            (is_urgent, urgency_score)
        """
        text_lower = text.lower()
        score = 0.0

        # Check Devanagari urgency keywords
        for keyword, weight in HINDI_URGENCY_KEYWORDS.items():
            if keyword in text:
                score += weight

        # Check romanized urgency keywords
        for keyword, weight in HINDI_URGENCY_ROMANIZED.items():
            if keyword in text_lower:
                score += weight

        return (score >= 0.5, min(1.0, score))

    def get_realtime_keywords(self, text: str) -> list[str]:
        """Find Hindi real-time keywords in text."""
        text_lower = text.lower()
        matches = []
        for kw in HINDI_REALTIME_KEYWORDS:
            if kw in text or kw in text_lower:
                matches.append(kw)
        return matches

    def get_complexity_keywords(self, text: str) -> list[str]:
        """Find Hindi complexity keywords in text."""
        text_lower = text.lower()
        matches = []
        for kw in HINDI_COMPLEXITY_KEYWORDS:
            if kw in text or kw in text_lower:
                matches.append(kw)
        return matches

    @property
    def is_ready(self) -> bool:
        return self._initialized
