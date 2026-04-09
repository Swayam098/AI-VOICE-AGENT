"""
AI Voice Call Agent — Main Application Entry Point.
FastAPI server orchestrating all components:
  Telephony ↔ Speech ↔ Intelligence ↔ Memory
"""

import asyncio
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Ensure backend dir is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_settings
from speech.stt import STTEngine
from speech.tts import TTSEngine
from intelligence.router import SmartRouter
from intelligence.local_model import LocalModelEngine
from intelligence.cloud_model import CloudModelEngine
from intelligence.web_search import WebSearchEngine
from intelligence.pipeline import IntelligencePipeline
from emotion.sentiment import SentimentAnalyzer
from memory.supabase_client import SupabaseClient
from memory.conversation import ConversationManager
from telephony.twilio_handler import router as twilio_router
from telephony.media_stream import MediaStreamHandler
from api.routes import router as api_router, register_components
from api.ws_routes import live_status, LiveStatusManager


# ─── Global Component Registry ───
components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("🧠 AI Voice Call Agent — Starting Up")
    logger.info("=" * 60)

    # ── Initialize all components ──

    # 1. Database (Memory)
    db = SupabaseClient()
    await db.initialize()
    components["db"] = db

    conversation_mgr = ConversationManager(db)
    components["conversation"] = conversation_mgr

    # 2. Speech engines
    stt = STTEngine()
    tts = TTSEngine()

    # Initialize speech in background (model loading is slow)
    logger.info("Loading speech models (this may take a moment)...")
    try:
        await stt.initialize()
    except Exception as e:
        logger.warning(f"STT initialization skipped: {e}")

    try:
        await tts.initialize()
    except Exception as e:
        logger.warning(f"TTS initialization skipped: {e}")

    components["stt"] = stt
    components["tts"] = tts

    # 3. Intelligence layer
    router_engine = SmartRouter()
    components["router"] = router_engine

    local_model = LocalModelEngine()
    if settings.enable_local_models:
        await local_model.initialize()
    components["local_model"] = local_model

    cloud_model = CloudModelEngine()
    await cloud_model.initialize()
    components["cloud_model"] = cloud_model

    web_search = WebSearchEngine()
    if settings.enable_web_search:
        await web_search.initialize()
    components["web_search"] = web_search

    sentiment = SentimentAnalyzer()
    if settings.enable_emotion_detection:
        await sentiment.initialize()
    components["sentiment"] = sentiment

    # 4. Intelligence Pipeline (ties everything together)
    pipeline = IntelligencePipeline(
        router=router_engine,
        local_model=local_model,
        cloud_model=cloud_model,
        web_search=web_search,
        sentiment=sentiment,
    )
    components["intelligence_pipeline"] = pipeline

    # 5. Media Stream Handler (WebSocket audio)
    media_handler = MediaStreamHandler(
        stt_engine=stt,
        tts_engine=tts,
        intelligence=pipeline,
        memory=db,
    )
    components["media_handler"] = media_handler

    # 6. Register components with API routes
    register_components(components)

    # 7. Start live status broadcast loop
    status_task = asyncio.create_task(
        live_status.status_loop(components, interval=5.0)
    )

    # ── Startup complete ──
    logger.info("=" * 60)
    logger.info("✅ All components initialized")
    logger.info(f"   STT:         {'✅ Ready' if stt.is_ready else '⚠️  Not loaded'}")
    logger.info(f"   TTS:         {'✅ Ready' if tts.is_ready else '⚠️  Not loaded'}")
    logger.info(f"   Local LLM:   {'✅ Ready' if local_model.is_ready else '⚠️  Not available'}")
    logger.info(f"   Cloud LLM:   {'✅ Ready' if cloud_model.is_ready else '⚠️  Not configured'}")
    logger.info(f"   Web Search:  {'✅ Ready' if web_search.is_ready else '⚠️  Disabled'}")
    logger.info(f"   Sentiment:   {'✅ Ready' if sentiment.is_ready else '⚠️  Disabled'}")
    logger.info(f"   Database:    {'✅ Supabase' if not db._use_local else '📦 In-Memory'}")
    logger.info(f"   Server:      http://{settings.host}:{settings.port}")
    logger.info("=" * 60)
    logger.info("🚀 AI Voice Agent is LIVE — Ready to handle calls!")
    logger.info("   Test endpoint: POST /api/test/chat")
    logger.info("   WebSocket test: ws://localhost:8000/test-stream")
    logger.info("=" * 60)

    yield

    # ── Shutdown ──
    logger.info("Shutting down AI Voice Agent...")
    status_task.cancel()
    try:
        await status_task
    except asyncio.CancelledError:
        pass
    logger.info("Goodbye! 👋")


# ─── Create FastAPI App ───
app = FastAPI(
    title="AI Voice Call Agent",
    description="Cross-Platform AI Voice Agent with Smart Routing & Real-Time Knowledge",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS Middleware ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Mount Routers ───
app.include_router(twilio_router)
app.include_router(api_router)


# ─── WebSocket Endpoints ───

@app.websocket("/media-stream")
async def media_stream_ws(websocket: WebSocket):
    """Twilio Media Stream WebSocket endpoint."""
    handler = components.get("media_handler")
    if handler:
        await handler.handle_twilio_stream(websocket)
    else:
        await websocket.close(code=1011, reason="Media handler not initialized")


@app.websocket("/test-stream")
async def test_stream_ws(websocket: WebSocket):
    """Direct test WebSocket — send text, get AI responses (no audio needed)."""
    handler = components.get("media_handler")
    if handler:
        await handler.handle_test_stream(websocket)
    else:
        await websocket.close(code=1011, reason="Media handler not initialized")


@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    """Dashboard live status WebSocket."""
    await live_status.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages
            data = await websocket.receive_text()
            # Client can send ping/pong or commands
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
    except Exception:
        live_status.disconnect(websocket)


# ─── Health Check ───

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-voice-agent",
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "AI Voice Call Agent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_status": "/api/status",
        "test_chat": "POST /api/test/chat",
        "test_ws": "ws://localhost:8000/test-stream",
    }


# ─── Run ───
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level,
    )
