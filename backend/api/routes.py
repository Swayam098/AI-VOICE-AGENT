"""
REST API Routes for the Dashboard.
Provides endpoints for call logs, analytics, settings, and system status.
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import Optional
from loguru import logger

router = APIRouter(prefix="/api", tags=["dashboard"])

# These will be injected at startup from main.py
_components = {}


def register_components(components: dict):
    """Register system components for API access."""
    global _components
    _components = components


# ─── Pydantic Models ───

class TestMessageRequest(BaseModel):
    text: str
    session_id: Optional[str] = "api-test"


class SettingsUpdate(BaseModel):
    enable_local_models: Optional[bool] = None
    enable_web_search: Optional[bool] = None
    enable_emotion_detection: Optional[bool] = None
    router_short_query_max_words: Optional[int] = None
    router_complex_min_words: Optional[int] = None


# ─── System Status ───

@router.get("/status")
async def system_status():
    """Get comprehensive system status."""
    router_engine = _components.get("router")
    local_model = _components.get("local_model")
    cloud_model = _components.get("cloud_model")
    web_search = _components.get("web_search")
    stt = _components.get("stt")
    tts = _components.get("tts")
    db = _components.get("db")
    sentiment = _components.get("sentiment")
    media_handler = _components.get("media_handler")

    active_sessions = media_handler.get_active_sessions() if media_handler else []

    return {
        "status": "operational",
        "components": {
            "stt": {"ready": stt.is_ready if stt else False, "model": "faster-whisper"},
            "tts": {"ready": tts.is_ready if tts else False, "model": "piper"},
            "local_model": {"ready": local_model.is_ready if local_model else False, "engine": "ollama"},
            "cloud_model": {"ready": cloud_model.is_ready if cloud_model else False, "engine": "groq"},
            "web_search": {"ready": web_search.is_ready if web_search else False, "engine": "duckduckgo"},
            "database": {"ready": db.is_ready if db else False},
            "sentiment": {"ready": sentiment.is_ready if sentiment else False},
        },
        "active_calls": len(active_sessions),
        "active_sessions": active_sessions,
    }


# ─── Intelligence Testing ───

@router.post("/test/route")
async def test_routing(request: TestMessageRequest):
    """Test the smart routing engine with a text query."""
    router_engine = _components.get("router")
    if not router_engine:
        raise HTTPException(status_code=503, detail="Router not initialized")

    decision = router_engine.classify(request.text)
    return {
        "query": request.text,
        "route": decision.query_type.value,
        "confidence": decision.confidence,
        "reason": decision.reason,
    }


@router.post("/test/chat")
async def test_chat(request: TestMessageRequest):
    """
    Test the full intelligence pipeline with a text query.
    Routes through Smart Router → appropriate model → response.
    """
    intelligence = _components.get("intelligence_pipeline")
    if not intelligence:
        raise HTTPException(status_code=503, detail="Intelligence pipeline not initialized")

    result = await intelligence.process(
        text=request.text,
        conversation_history=[],
        session_id=request.session_id,
    )

    return {
        "query": request.text,
        "response": result.get("text", ""),
        "route": result.get("route", ""),
        "language": result.get("language", "en"),
        "latency_ms": result.get("latency_ms", 0),
        "model": result.get("model", ""),
    }


@router.post("/test/sentiment")
async def test_sentiment(request: TestMessageRequest):
    """Test sentiment analysis on a text string."""
    sentiment = _components.get("sentiment")
    if not sentiment:
        raise HTTPException(status_code=503, detail="Sentiment analyzer not initialized")

    result = sentiment.analyze(request.text)
    return {"text": request.text, "sentiment": result}


@router.post("/test/search")
async def test_web_search(request: TestMessageRequest):
    """Test web search with a query."""
    web_search = _components.get("web_search")
    if not web_search:
        raise HTTPException(status_code=503, detail="Web search not initialized")

    results = await web_search.search(request.text)
    return {"query": request.text, "results": results}


# ─── Call Logs ───

@router.get("/calls")
async def get_calls(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Retrieve call history."""
    db = _components.get("db")
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    calls = await db.get_calls(limit=limit, offset=offset)
    return {"calls": calls, "count": len(calls), "offset": offset}


@router.get("/calls/{session_id}/messages")
async def get_call_messages(session_id: str):
    """Retrieve all messages for a specific call session."""
    db = _components.get("db")
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    messages = await db.get_messages_for_call(session_id)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


# ─── Analytics ───

@router.get("/analytics/routing")
async def get_routing_analytics():
    """Get routing distribution statistics."""
    router_engine = _components.get("router")
    if not router_engine:
        return {"stats": {}}

    return {"stats": router_engine.get_stats()}


@router.get("/analytics/overview")
async def get_analytics_overview():
    """Get overall system analytics."""
    db = _components.get("db")
    router_engine = _components.get("router")
    cloud_model = _components.get("cloud_model")

    db_stats = await db.get_stats() if db else {}
    routing_stats = router_engine.get_stats() if router_engine else {}
    cloud_usage = cloud_model.get_usage_stats() if cloud_model else {}

    return {
        "database": db_stats,
        "routing": routing_stats,
        "cloud_usage": cloud_usage,
    }


# ─── Messages ───

@router.get("/messages/recent")
async def get_recent_messages(limit: int = Query(default=50, ge=1, le=200)):
    """Retrieve recent messages across all calls."""
    db = _components.get("db")
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    messages = await db.get_recent_messages(limit=limit)
    return {"messages": messages, "count": len(messages)}
