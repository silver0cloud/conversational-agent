"""
SQLite access layer.

Single-file DB (see config.db_path), stdlib sqlite3 only — no ORM, no extra
dependency, appropriate for a single-process personal project. Every
function here takes/returns plain dicts, not custom model objects, to keep
this easy to consume from the pipeline, the post-call processor, and the
FastAPI dashboard routes alike.

DEFAULT_USER_ID is used everywhere in place of a real logged-in user for
now. When real auth is added later, callers swap in the resolved user id —
nothing in the schema or these functions needs to change.
"""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from backend.config import settings

DEFAULT_USER_ID = "default_user"

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db() -> None:
    """Create the DB file and tables if they don't exist yet. Safe to call
    every time the server starts — CREATE TABLE IF NOT EXISTS is a no-op on
    an already-initialized database."""
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript(_SCHEMA_PATH.read_text())

    ensure_default_user()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --------------------------------------------------------------- users ----

def ensure_default_user(display_name: Optional[str] = None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (id, display_name) VALUES (?, ?)",
            (DEFAULT_USER_ID, display_name),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_profiles (user_id) VALUES (?)",
            (DEFAULT_USER_ID,),
        )


# ------------------------------------------------------------- profile ----

def get_profile(user_id: str = DEFAULT_USER_ID) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT user_id, tags_json, notes, updated_at FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return {"user_id": user_id, "tags": [], "notes": "", "updated_at": None}
    return {
        "user_id": row["user_id"],
        "tags": json.loads(row["tags_json"]),
        "notes": row["notes"],
        "updated_at": row["updated_at"],
    }


def save_profile(tags: list[str], notes: str, user_id: str = DEFAULT_USER_ID) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE user_profiles
            SET tags_json = ?, notes = ?, updated_at = datetime('now')
            WHERE user_id = ?
            """,
            (json.dumps(tags), notes, user_id),
        )


def has_completed_conversation(user_id: str = DEFAULT_USER_ID) -> bool:
    """Used by Phase 1 to decide: onboarding flow (first ever call) vs the
    normal 'what do you want to talk about today' flow (returning user)."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE user_id = ? AND status != 'in_progress' LIMIT 1",
            (user_id,),
        ).fetchone()
    return row is not None


# -------------------------------------------------------- conversations ---

def create_conversation(user_id: str = DEFAULT_USER_ID, topic: Optional[str] = None) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO conversations (user_id, topic) VALUES (?, ?)",
            (user_id, topic),
        )
        return cursor.lastrowid


def set_conversation_topic(conversation_id: int, topic: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET topic = ? WHERE id = ?",
            (topic, conversation_id),
        )


def end_conversation(conversation_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET status = 'ended', ended_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )


def mark_processing(conversation_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET status = 'processing' WHERE id = ?",
            (conversation_id,),
        )


def save_generated_content(conversation_id: int, summary: str, substack_draft: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET status = 'done', summary = ?, substack_draft = ?, processed_at = datetime('now')
            WHERE id = ?
            """,
            (summary, substack_draft, conversation_id),
        )


def mark_error(conversation_id: int, error_message: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE conversations SET status = 'error', error_message = ? WHERE id = ?",
            (error_message, conversation_id),
        )


def get_conversation(conversation_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return dict(row) if row else None


def list_conversations(user_id: str = DEFAULT_USER_ID, limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, topic, status, started_at, ended_at, processed_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------- turns ----

def add_turn(conversation_id: int, role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO conversation_turns (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )


def get_turns(conversation_id: int) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM conversation_turns
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_transcript_text(conversation_id: int) -> str:
    """Flattened 'User: ...\\nAgent: ...' form, ready to hand to an LLM
    prompt for summary/draft generation (Phase 2)."""
    turns = get_turns(conversation_id)
    speaker = {"user": "User", "assistant": "Agent"}
    return "\n".join(f"{speaker[t['role']]}: {t['content']}" for t in turns)
