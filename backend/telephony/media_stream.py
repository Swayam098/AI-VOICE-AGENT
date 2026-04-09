"""
WebSocket Media Stream Handler.
Manages bidirectional audio streaming between Twilio and our AI pipeline.
Also supports a direct test mode via WebSocket for development without Twilio.
"""

import asyncio
import base64
import json
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class CallSession:
    """Represents an active call session with audio buffering."""

    def __init__(self, call_sid: str, stream_sid: str = ""):
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.session_id = str(uuid.uuid4())
        self.started_at = time.time()
        self.audio_buffer = bytearray()
        self.is_speaking = False
        self.last_speech_time = 0.0
        self.silence_threshold = 1.5  # seconds of silence to trigger processing
        self.messages = []  # conversation history

    @property
    def duration(self) -> float:
        return time.time() - self.started_at


class MediaStreamHandler:
    """
    Handles WebSocket media stream connections.
    
    Supports two modes:
    1. Twilio Mode: Receives Twilio media stream events (JSON with base64 audio)
    2. Test Mode: Direct PCM audio for development/testing without Twilio
    """

    def __init__(self, stt_engine=None, tts_engine=None, intelligence=None, memory=None):
        self.stt_engine = stt_engine
        self.tts_engine = tts_engine
        self.intelligence = intelligence
        self.memory = memory
        self.active_sessions: dict[str, CallSession] = {}

    async def handle_twilio_stream(self, websocket: WebSocket):
        """Handle a Twilio Media Stream WebSocket connection."""
        await websocket.accept()
        session: Optional[CallSession] = None

        logger.info("Twilio Media Stream WebSocket connected")

        try:
            async for raw_message in websocket.iter_text():
                message = json.loads(raw_message)
                event = message.get("event")

                if event == "connected":
                    logger.info("Twilio stream connected")

                elif event == "start":
                    start_data = message.get("start", {})
                    call_sid = start_data.get("callSid", str(uuid.uuid4()))
                    stream_sid = start_data.get("streamSid", "")
                    session = CallSession(call_sid=call_sid, stream_sid=stream_sid)
                    self.active_sessions[session.session_id] = session
                    logger.info(f"Stream started — Session: {session.session_id}, Call: {call_sid}")

                elif event == "media" and session:
                    # Decode base64 μ-law audio payload
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        audio_bytes = base64.b64decode(payload)
                        session.audio_buffer.extend(audio_bytes)
                        session.last_speech_time = time.time()
                        session.is_speaking = True

                    # Check for silence → process accumulated audio
                    await self._check_and_process(session, websocket, mode="twilio")

                elif event == "stop":
                    logger.info(f"Stream stopped — Session: {session.session_id if session else 'unknown'}")
                    if session:
                        # Process any remaining audio
                        if len(session.audio_buffer) > 0:
                            await self._process_audio(session, websocket, mode="twilio")
                        self.active_sessions.pop(session.session_id, None)
                    break

        except WebSocketDisconnect:
            logger.info("Twilio Media Stream WebSocket disconnected")
        except Exception as e:
            logger.error(f"Media stream error: {e}")
        finally:
            if session:
                self.active_sessions.pop(session.session_id, None)

    async def handle_test_stream(self, websocket: WebSocket):
        """
        Handle a direct test WebSocket connection.
        Accepts text messages (simulating transcribed speech) for testing without audio.
        """
        await websocket.accept()
        session = CallSession(call_sid=f"test-{uuid.uuid4().hex[:8]}")
        self.active_sessions[session.session_id] = session

        logger.info(f"Test stream connected — Session: {session.session_id}")

        try:
            # Send welcome message
            await self._send_agent_response(
                session,
                websocket,
                "Hello! I'm your AI voice assistant. How can I help you today?",
                mode="test",
            )

            async for message in websocket.iter_text():
                data = json.loads(message)
                user_text = data.get("text", "").strip()

                if not user_text:
                    continue

                logger.info(f"[{session.session_id}] User: {user_text}")

                # Process through intelligence pipeline
                await self._process_text(session, websocket, user_text, mode="test")

        except WebSocketDisconnect:
            logger.info(f"Test stream disconnected — Session: {session.session_id}")
        except Exception as e:
            logger.error(f"Test stream error: {e}")
        finally:
            self.active_sessions.pop(session.session_id, None)

    async def _check_and_process(self, session: CallSession, websocket: WebSocket, mode: str):
        """Check if there's enough silence after speech to trigger processing."""
        if not session.is_speaking:
            return

        time_since_speech = time.time() - session.last_speech_time
        if time_since_speech >= session.silence_threshold and len(session.audio_buffer) > 0:
            session.is_speaking = False
            await self._process_audio(session, websocket, mode)

    async def _process_audio(self, session: CallSession, websocket: WebSocket, mode: str):
        """Process accumulated audio buffer through STT → Intelligence → TTS pipeline."""
        if not self.stt_engine:
            logger.warning("No STT engine configured — skipping audio processing")
            session.audio_buffer.clear()
            return

        start_time = time.time()

        # Step 1: Speech-to-Text
        audio_data = bytes(session.audio_buffer)
        session.audio_buffer.clear()

        try:
            if mode == "twilio":
                stt_result = await self.stt_engine.transcribe_from_mulaw(audio_data)
            else:
                stt_result = await self.stt_engine.transcribe(audio_data)

            if not stt_result or not stt_result.get("text", "").strip():
                return

            transcription = stt_result["text"]
            detected_lang = stt_result.get("language", "en")

            logger.info(f"[{session.session_id}] Transcribed ({detected_lang}): {transcription}")

            # Step 2: Process through intelligence
            await self._process_text(session, websocket, transcription, mode)

            latency = (time.time() - start_time) * 1000
            logger.info(f"[{session.session_id}] Total pipeline latency: {latency:.0f}ms")

        except Exception as e:
            logger.error(f"Audio processing error: {e}")

    async def _process_text(self, session: CallSession, websocket: WebSocket, text: str, mode: str):
        """Process transcribed text through Intelligence → TTS pipeline."""
        start_time = time.time()

        # Track user message
        session.messages.append({"role": "user", "content": text})

        # Get AI response
        if self.intelligence:
            response = await self.intelligence.process(
                text=text,
                conversation_history=session.messages,
                session_id=session.session_id,
            )
        else:
            response = {
                "text": f"I heard you say: '{text}'. Intelligence engine not configured.",
                "route": "echo",
                "latency_ms": 0,
            }

        agent_text = response.get("text", "I'm sorry, I couldn't process that.")
        route_used = response.get("route", "unknown")
        language_used = response.get("language", "en")

        logger.info(f"[{session.session_id}] Agent ({route_used}, lang: {language_used}): {agent_text}")

        # Track agent message
        session.messages.append({"role": "assistant", "content": agent_text})

        # Store in memory
        if self.memory:
            await self.memory.store_message(
                session_id=session.session_id,
                call_sid=session.call_sid,
                role="user",
                content=text,
                route_used="",
                latency_ms=0,
            )
            await self.memory.store_message(
                session_id=session.session_id,
                call_sid=session.call_sid,
                role="assistant",
                content=agent_text,
                route_used=route_used,
                latency_ms=int((time.time() - start_time) * 1000),
            )

        # Send response
        await self._send_agent_response(session, websocket, agent_text, mode, language=language_used)

    async def _send_agent_response(
        self, session: CallSession, websocket: WebSocket, text: str, mode: str, language: str = "en"
    ):
        """Send agent response — as audio (Twilio) or text (test mode)."""
        if mode == "twilio" and self.tts_engine:
            # Synthesize speech and send as base64 audio
            try:
                # Twilio expects 8kHz μ-law audio
                audio_bytes = await self.tts_engine.synthesize_to_mulaw(text, language=language)
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

                media_message = {
                    "event": "media",
                    "streamSid": session.stream_sid,
                    "media": {"payload": audio_b64},
                }
                await websocket.send_text(json.dumps(media_message))
            except Exception as e:
                logger.error(f"TTS error: {e}")
        else:
            # Test mode — send text response
            await websocket.send_text(json.dumps({
                "type": "response",
                "text": text,
                "session_id": session.session_id,
                "timestamp": time.time(),
            }))

    def get_active_sessions(self) -> list[dict]:
        """Return info about all active call sessions."""
        return [
            {
                "session_id": s.session_id,
                "call_sid": s.call_sid,
                "duration": round(s.duration, 1),
                "message_count": len(s.messages),
                "is_speaking": s.is_speaking,
            }
            for s in self.active_sessions.values()
        ]
