# 🧠 AI Voice Call Agent

Cross-Platform AI Voice Agent with Smart Routing & Real-Time Knowledge.

## Architecture

```
Incoming Call → Twilio → WebSocket → STT (Whisper) → Smart Router → Model → TTS (Piper) → Response
                                                        ↓
                                              ┌─────────┼─────────┐
                                              │         │         │
                                          Real-time  Complex   Normal/Short
                                          Web+Model   Groq    Ollama Mistral/Phi-3
```

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Ollama with `mistral` and `phi3` models
- NVIDIA GPU (RTX 4050 or similar) for Whisper + local LLM

### 2. Setup
```bash
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env
# Edit .env with your API keys
```

### 3. Run
```bash
cd backend
python main.py
```

### 4. Test
- **Swagger Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **System Status**: http://localhost:8000/api/status
- **Test Chat**: `POST /api/test/chat` with `{"text": "Hello!"}`
- **WebSocket**: Connect to `ws://localhost:8000/test-stream`

### 5. Supabase Setup
1. Create a project at [supabase.com](https://supabase.com)
2. Run `supabase_migration.sql` in the SQL Editor
3. Add `SUPABASE_URL` and `SUPABASE_KEY` to `.env`

## Smart Routing

| Query Type | Trigger | Model |
|-----------|---------|-------|
| Real-time | "latest", "news", "today" keywords | Web Search + Cloud/Local |
| Complex | Long queries, reasoning keywords | Groq API |
| Normal | Standard conversation | Ollama Mistral |
| Short | < 10 words | Ollama Phi-3 |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/status` | Full system status |
| POST | `/api/test/chat` | Test intelligence pipeline |
| POST | `/api/test/route` | Test smart router |
| POST | `/api/test/sentiment` | Test sentiment analysis |
| POST | `/api/test/search` | Test web search |
| GET | `/api/calls` | Call history |
| GET | `/api/analytics/overview` | System analytics |
| WS | `/test-stream` | Interactive text chat test |
| WS | `/media-stream` | Twilio audio stream |
| WS | `/ws/dashboard` | Live dashboard updates |
