"""
Twilio Voice Webhook Handlers.
Manages incoming call routing and TwiML responses.
Ready to wire up once a Twilio account is configured.
"""

from fastapi import APIRouter, Request, Response, HTTPException
from loguru import logger
import xml.etree.ElementTree as ET

from config import get_settings

router = APIRouter(prefix="/voice", tags=["telephony"])


def generate_twiml_connect(ws_url: str) -> str:
    """Generate TwiML XML to connect a call to a WebSocket media stream."""
    response = ET.Element("Response")
    say = ET.SubElement(response, "Say", voice="Polly.Amy")
    say.text = "Connecting you to the AI assistant. Please hold."
    connect = ET.SubElement(response, "Connect")
    stream = ET.SubElement(connect, "Stream", url=ws_url)
    ET.SubElement(stream, "Parameter", name="direction", value="both")
    return '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(response, encoding="unicode")


@router.post("/incoming")
async def handle_incoming_call(request: Request):
    """
    Handle incoming voice calls from Twilio.
    Returns TwiML that connects the call to our WebSocket media stream.
    """
    settings = get_settings()

    if not settings.twilio_configured:
        logger.warning("Twilio not configured — returning mock TwiML response")

    # Build WebSocket URL for media streaming
    # In production, this would use the ngrok or public server URL
    host = request.headers.get("host", f"localhost:{settings.port}")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{scheme}://{host}/media-stream"

    logger.info(f"Incoming call — connecting to WebSocket: {ws_url}")

    # Extract caller info from Twilio request
    form_data = await request.form()
    caller = form_data.get("From", "unknown")
    called = form_data.get("To", "unknown")
    call_sid = form_data.get("CallSid", "unknown")

    logger.info(f"Call SID: {call_sid} | From: {caller} | To: {called}")

    twiml = generate_twiml_connect(ws_url)
    return Response(content=twiml, media_type="application/xml")


@router.post("/status")
async def handle_call_status(request: Request):
    """
    Handle call status callbacks from Twilio.
    Tracks call lifecycle: initiated → ringing → in-progress → completed.
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    call_status = form_data.get("CallStatus", "unknown")
    duration = form_data.get("CallDuration", "0")

    logger.info(f"Call {call_sid} status: {call_status} (duration: {duration}s)")

    return Response(content="OK", status_code=200)


@router.get("/health")
async def telephony_health():
    """Check telephony configuration status."""
    settings = get_settings()
    return {
        "twilio_configured": settings.twilio_configured,
        "status": "ready" if settings.twilio_configured else "not_configured",
        "message": "Add Twilio credentials to .env to enable call handling"
        if not settings.twilio_configured
        else "Twilio ready",
    }
