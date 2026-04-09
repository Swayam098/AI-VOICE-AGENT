"""
Supabase Client — Database connection and operations.
Manages persistent storage for calls, messages, and summaries.
Falls back to in-memory storage if Supabase is not configured.
"""

import time
import uuid
from typing import Optional
from datetime import datetime, timezone

from loguru import logger

from config import get_settings


class SupabaseClient:
    """
    Supabase database client for conversation persistence.
    Provides CRUD operations for calls, messages, and summaries.
    Falls back to local in-memory storage when Supabase isn't configured.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self._initialized = False

        # In-memory fallback storage
        self._local_calls: list[dict] = []
        self._local_messages: list[dict] = []
        self._local_summaries: list[dict] = []
        self._use_local = False

    async def initialize(self):
        """Initialize Supabase client connection."""
        if not self.settings.supabase_configured:
            logger.warning("Supabase not configured — using in-memory storage")
            self._use_local = True
            self._initialized = True
            return

        try:
            from supabase import create_client

            self.client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_key,
            )
            self._initialized = True
            self._use_local = False
            logger.success("Supabase client initialized")

            # Verify tables exist by doing a simple query
            try:
                self.client.table("calls").select("id").limit(1).execute()
                logger.info("Supabase tables verified")
            except Exception as e:
                logger.warning(f"Supabase tables may not exist yet: {e}")
                logger.info("Run the SQL migration to create tables")

        except ImportError:
            logger.error("supabase package not installed — using in-memory storage")
            self._use_local = True
            self._initialized = True
        except Exception as e:
            logger.error(f"Supabase initialization failed: {e} — using in-memory storage")
            self._use_local = True
            self._initialized = True

    # ─── Calls ───

    async def create_call(
        self,
        call_sid: str,
        caller_phone: str = "unknown",
        session_id: str = "",
    ) -> dict:
        """Create a new call record."""
        call = {
            "id": str(uuid.uuid4()),
            "call_sid": call_sid,
            "session_id": session_id,
            "caller_phone": caller_phone,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "duration_seconds": 0,
            "routing_decisions": [],
            "sentiment_score": None,
        }

        if self._use_local:
            self._local_calls.append(call)
            return call

        try:
            result = self.client.table("calls").insert(call).execute()
            return result.data[0] if result.data else call
        except Exception as e:
            logger.error(f"Failed to create call record: {e}")
            self._local_calls.append(call)
            return call

    async def end_call(self, call_id: str, sentiment_score: float = None):
        """Mark a call as ended."""
        update = {
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
        if sentiment_score is not None:
            update["sentiment_score"] = sentiment_score

        if self._use_local:
            for call in self._local_calls:
                if call["id"] == call_id:
                    call.update(update)
                    started = datetime.fromisoformat(call["started_at"])
                    call["duration_seconds"] = int(
                        (datetime.now(timezone.utc) - started).total_seconds()
                    )
                    break
            return

        try:
            self.client.table("calls").update(update).eq("id", call_id).execute()
        except Exception as e:
            logger.error(f"Failed to end call: {e}")

    async def get_calls(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """Retrieve recent calls."""
        if self._use_local:
            sorted_calls = sorted(
                self._local_calls, key=lambda x: x["started_at"], reverse=True
            )
            return sorted_calls[offset : offset + limit]

        try:
            result = (
                self.client.table("calls")
                .select("*")
                .order("started_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get calls: {e}")
            return self._local_calls[-limit:]

    # ─── Messages ───

    async def store_message(
        self,
        session_id: str,
        call_sid: str,
        role: str,
        content: str,
        route_used: str = "",
        latency_ms: int = 0,
    ) -> dict:
        """Store a conversation message."""
        message = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "call_sid": call_sid,
            "role": role,
            "content": content,
            "route_used": route_used,
            "latency_ms": latency_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._use_local:
            self._local_messages.append(message)
            return message

        try:
            result = self.client.table("messages").insert(message).execute()
            return result.data[0] if result.data else message
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            self._local_messages.append(message)
            return message

    async def get_messages_for_call(self, session_id: str) -> list[dict]:
        """Retrieve all messages for a given session."""
        if self._use_local:
            return [m for m in self._local_messages if m["session_id"] == session_id]

        try:
            result = (
                self.client.table("messages")
                .select("*")
                .eq("session_id", session_id)
                .order("created_at")
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get messages: {e}")
            return []

    async def get_recent_messages(self, limit: int = 100) -> list[dict]:
        """Retrieve recent messages across all calls."""
        if self._use_local:
            return sorted(
                self._local_messages, key=lambda x: x["created_at"], reverse=True
            )[:limit]

        try:
            result = (
                self.client.table("messages")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            return []

    # ─── Summaries ───

    async def store_summary(
        self,
        call_id: str,
        session_id: str,
        summary: str,
        key_topics: list[str] = None,
        action_items: list[str] = None,
    ) -> dict:
        """Store a conversation summary."""
        summary_record = {
            "id": str(uuid.uuid4()),
            "call_id": call_id,
            "session_id": session_id,
            "summary": summary,
            "key_topics": key_topics or [],
            "action_items": action_items or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._use_local:
            self._local_summaries.append(summary_record)
            return summary_record

        try:
            result = self.client.table("summaries").insert(summary_record).execute()
            return result.data[0] if result.data else summary_record
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")
            self._local_summaries.append(summary_record)
            return summary_record

    # ─── Stats ───

    async def get_stats(self) -> dict:
        """Get database statistics."""
        if self._use_local:
            return {
                "storage": "in-memory",
                "total_calls": len(self._local_calls),
                "total_messages": len(self._local_messages),
                "total_summaries": len(self._local_summaries),
            }

        try:
            calls = self.client.table("calls").select("id", count="exact").execute()
            messages = self.client.table("messages").select("id", count="exact").execute()
            return {
                "storage": "supabase",
                "total_calls": calls.count or 0,
                "total_messages": messages.count or 0,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"storage": "error", "error": str(e)}

    @property
    def is_ready(self) -> bool:
        return self._initialized
