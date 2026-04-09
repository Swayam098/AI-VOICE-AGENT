"""
Conversation Manager.
Handles conversation context, history retrieval, and summary generation.
"""

import time
from typing import Optional

from loguru import logger

from memory.supabase_client import SupabaseClient


class ConversationManager:
    """
    Manages conversation context for active and historical calls.
    Maintains rolling context windows and generates summaries.
    """

    def __init__(self, db: SupabaseClient):
        self.db = db
        self.max_context_messages = 10  # Rolling context window

    async def get_context(
        self,
        session_id: str,
        conversation_history: list[dict] = None,
    ) -> str:
        """
        Build context string for the current conversation.
        Includes recent messages and any caller history.

        Args:
            session_id: Current session ID
            conversation_history: In-memory message history

        Returns:
            Formatted context string for prompt injection
        """
        context_parts = []

        # Use in-memory history if available
        if conversation_history:
            recent = conversation_history[-self.max_context_messages:]
            if recent:
                context_parts.append(f"Current conversation ({len(recent)} messages in context).")

        # Try to fetch caller history from DB
        try:
            previous_messages = await self.db.get_messages_for_call(session_id)
            if previous_messages:
                context_parts.append(
                    f"This caller has {len(previous_messages)} messages in this session."
                )
        except Exception:
            pass

        return " ".join(context_parts) if context_parts else "New caller, no previous history."

    async def store_exchange(
        self,
        session_id: str,
        call_sid: str,
        user_text: str,
        agent_text: str,
        route_used: str = "",
        latency_ms: int = 0,
    ):
        """Store a complete user-agent exchange."""
        await self.db.store_message(
            session_id=session_id,
            call_sid=call_sid,
            role="user",
            content=user_text,
            route_used="",
            latency_ms=0,
        )

        await self.db.store_message(
            session_id=session_id,
            call_sid=call_sid,
            role="assistant",
            content=agent_text,
            route_used=route_used,
            latency_ms=latency_ms,
        )

    async def generate_summary(
        self,
        session_id: str,
        conversation_history: list[dict],
        model_engine=None,
    ) -> Optional[str]:
        """
        Generate a conversation summary using the AI model.

        Args:
            session_id: Session to summarize
            conversation_history: Full message history
            model_engine: LLM engine to use for summarization

        Returns:
            Summary text or None
        """
        if not conversation_history or len(conversation_history) < 2:
            return None

        from intelligence.prompt_templates import (
            SUMMARY_PROMPT,
            format_conversation_for_summary,
        )

        conversation_text = format_conversation_for_summary(conversation_history)
        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

        if model_engine:
            try:
                result = await model_engine.generate(
                    prompt=prompt,
                    system_prompt="You summarize phone conversations concisely.",
                    max_tokens=256,
                    temperature=0.3,
                )
                return result.get("text", None)
            except Exception as e:
                logger.error(f"Summary generation failed: {e}")

        return None

    async def get_call_transcript(self, session_id: str) -> list[dict]:
        """Get the full transcript for a session."""
        return await self.db.get_messages_for_call(session_id)
