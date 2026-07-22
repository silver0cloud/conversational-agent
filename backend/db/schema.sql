-- Phase 0 schema.
--
-- Every user-owned table carries a `user_id` column even though today there's
-- only ever one user (DEFAULT_USER_ID in db/__init__.py). That means adding
-- real accounts later is: add an auth layer that resolves a real user_id
-- instead of the hardcoded default — no schema migration, no data reshaping.

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    display_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per user: the accumulated "who is this person" profile that the
-- onboarding conversation (Phase 1) seeds and every conversation's
-- post-processing (Phase 2) refines.
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY REFERENCES users(id),
    -- JSON array of short tags, e.g. ["enjoys moral ambiguity", "romance genre fan"]
    tags_json TEXT NOT NULL DEFAULT '[]',
    -- Freeform running notes the LLM appends/revises after each conversation.
    notes TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per call.
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(id),
    topic TEXT,  -- what the user chose to discuss, filled in once known
    -- in_progress -> ended -> processing -> done  (or -> error)
    status TEXT NOT NULL DEFAULT 'in_progress',
    summary TEXT,          -- LLM-generated post-call summary (Phase 2)
    substack_draft TEXT,   -- LLM-generated markdown draft (Phase 2)
    error_message TEXT,    -- populated only if status = 'error'
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT,
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);

-- One row per spoken turn, in order. Querying "give me the transcript for
-- conversation X" is just SELECT ... WHERE conversation_id = X ORDER BY id.
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation ON conversation_turns(conversation_id);
