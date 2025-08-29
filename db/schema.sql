CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS recordings (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  s3_key TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS transcripts (
  id UUID PRIMARY KEY,
  recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
  user_id UUID NOT NULL,
  lang TEXT,
  text TEXT NOT NULL,
  words_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS transcript_chunks (
  id BIGSERIAL PRIMARY KEY,
  transcript_id UUID NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
  idx INT NOT NULL,
  text TEXT NOT NULL,
  embedding VECTOR(1536)
);
CREATE INDEX IF NOT EXISTS transcript_chunks_embedding_idx
  ON transcript_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX IF NOT EXISTS transcript_chunks_tidx ON transcript_chunks(transcript_id, idx);

CREATE TABLE IF NOT EXISTS memories (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  transcript_id UUID REFERENCES transcripts(id) ON DELETE SET NULL,
  summary TEXT,
  action_items JSONB,
  tags JSONB,
  flags JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
