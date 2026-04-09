-- ============================================
--  AI Voice Call Agent — Supabase Schema
--  Run this in your Supabase SQL Editor
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Calls Table ───
CREATE TABLE IF NOT EXISTS calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_sid TEXT NOT NULL,
    session_id TEXT,
    caller_phone TEXT DEFAULT 'unknown',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds INT DEFAULT 0,
    routing_decisions JSONB DEFAULT '[]'::jsonb,
    sentiment_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Messages Table ───
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id TEXT NOT NULL,
    call_sid TEXT,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    route_used TEXT DEFAULT '',
    latency_ms INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Summaries Table ───
CREATE TABLE IF NOT EXISTS summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    session_id TEXT,
    summary TEXT NOT NULL,
    key_topics TEXT[] DEFAULT '{}',
    action_items TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Indexes for Performance ───
CREATE INDEX IF NOT EXISTS idx_calls_started_at ON calls(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_calls_session_id ON calls(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_call_id ON summaries(call_id);

-- ─── Row Level Security ───
ALTER TABLE calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE summaries ENABLE ROW LEVEL SECURITY;

-- Allow service role full access (use service key on backend)
CREATE POLICY "Service role full access on calls"
    ON calls FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access on messages"
    ON messages FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Service role full access on summaries"
    ON summaries FOR ALL
    USING (true)
    WITH CHECK (true);

-- ─── Enable Realtime for messages table ───
ALTER PUBLICATION supabase_realtime ADD TABLE messages;
