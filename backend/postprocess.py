"""
Post-call processing (Phase 2).

Runs once, in the background, right after a call ends (see main.py):
  1. Pull the full transcript from the DB
  2. One JSON-mode LLM call: extract the topic + an updated, CONSOLIDATED
     user profile (tags + notes) — the model rewrites/merges with the
     existing profile rather than just appending, so it stays clean over
     many conversations instead of accumulating near-duplicate tags
  3. One prose LLM call: a crisp personal-notes summary AND a polished,
     publish-ready Substack draft, in a single response separated by a
     delimiter (kept as plain prose rather than JSON so markdown
     formatting, quotes, and line breaks in the creative writing never
     fight with JSON string-escaping)
  4. Persist everything: conversations.topic/summary/substack_draft, and
     user_profiles.tags/notes

Talks to Groq directly via the `openai` SDK (not through the Pipecat
pipeline) since this isn't a real-time voice turn — it's a one-shot
background job with no latency constraint, so unlike the live conversation
it can afford full thinking-mode reasoning for better writing quality.
"""

import json
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI

from backend import db
from backend.config import settings

_client: Optional[AsyncOpenAI] = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
    return _client


# ------------------------------------------------------- structured pass --

EXTRACTION_SYSTEM_PROMPT = """You analyze a transcript of a spoken conversation \
between a voice AI companion and a user discussing Jane Austen's Pride and \
Prejudice. You output ONLY a single valid JSON object, nothing else before \
or after it, with exactly these keys:

{
  "topic": "<a short, natural phrase describing what was actually discussed>",
  "tags": ["<consolidated list of concise tags reflecting this person's \
reading interests and personality, merged with the existing tags given \
below — rephrase or combine duplicates rather than just appending; keep it \
to 10 tags or fewer>"],
  "notes": "<a concise 2-4 sentence running note about this person's \
reading interests and personality, REWRITING/SYNTHESIZING the existing \
notes below together with what this conversation revealed — not just \
appending a new sentence>"
}"""


def _build_extraction_user_prompt(transcript: str, existing_profile: dict) -> str:
    tags = existing_profile.get("tags") or []
    notes = existing_profile.get("notes") or ""
    return (
        f"Existing tags: {json.dumps(tags)}\n"
        f"Existing notes: {notes or '(none yet)'}\n\n"
        f"Transcript:\n{transcript}"
    )


async def _extract_structured_info(transcript: str, existing_profile: dict) -> dict:
    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_extraction_user_prompt(transcript, existing_profile)},
        ],
        response_format={"type": "json_object"},
        # Purely extractive/structural — no need for deep reasoning, and
        # keeping it fast matters even in the background since this call
        # blocks the second (drafting) call from starting.
        extra_body={"reasoning_effort": "none"},
    )
    return json.loads(response.choices[0].message.content)


# --------------------------------------------------------- drafting pass --

DRAFTING_SYSTEM_PROMPT = """You are a thoughtful writer helping someone turn a spoken \
conversation about Jane Austen's Pride and Prejudice into two pieces of \
written output. Respond in EXACTLY this format, nothing before or after it:

SUMMARY:
<summary text>

===SUBSTACK_DRAFT_BELOW===
<substack draft in markdown>

Requirements for SUMMARY: a crisp, information-dense synthesis of the key \
points, insights, and interesting moments from the conversation — written \
so the user can paste it directly into their personal notes. A few \
sentences to a short paragraph. Not a transcript recap; a distillation.

Requirements for the Substack draft: a polished, engaging, publish-ready \
blog post inspired by the ideas and views raised in the conversation — NOT \
a transcript or a "user said / I said" recap. Start with a compelling \
headline (as a markdown H1) and a hook that draws the reader in. Develop \
the ideas into a coherent, well-structured essay with your own authorial \
voice, varied sentence rhythm, and correct grammar. End on a satisfying, \
thought-provoking closing line. It should stand alone and be enjoyable to \
someone who never heard the conversation. Aim for roughly 400-700 words."""

_DRAFT_DELIMITER = "===SUBSTACK_DRAFT_BELOW==="


async def _generate_summary_and_draft(transcript: str) -> tuple[str, str]:
    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": DRAFTING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
        # Full thinking mode here (reasoning_effort intentionally omitted) —
        # this is a background job with no latency constraint, so let the
        # model reason through structure/voice before writing.
        # reasoning_format="parsed" makes Groq strip the <think> block for
        # us server-side, so .content is just the final answer.
        extra_body={"reasoning_format": "parsed"},
    )
    raw = (response.choices[0].message.content or "").strip()

    if _DRAFT_DELIMITER in raw:
        summary_part, draft_part = raw.split(_DRAFT_DELIMITER, 1)
    else:
        # Model didn't follow the format exactly — fail safe rather than
        # crash: keep the whole response as the draft and leave summary
        # empty, rather than silently duplicating content into both fields
        # or guessing where to split.
        logger.warning("[postprocess] delimiter missing in drafting response; using fallback")
        summary_part, draft_part = "", raw

    summary = summary_part.replace("SUMMARY:", "", 1).strip()
    draft = draft_part.strip()
    return summary, draft


# ------------------------------------------------------------- orchestrate

async def process_conversation(conversation_id: int) -> None:
    """Entry point: fire-and-forget this in the background right after a
    call ends (see main.py). Never raises — failures are recorded on the
    conversation row via db.mark_error rather than propagating, since this
    always runs detached from any request/response cycle."""
    try:
        db.mark_processing(conversation_id)

        transcript = db.get_transcript_text(conversation_id)
        if not transcript.strip():
            logger.info(f"[postprocess] conversation {conversation_id} had no real exchange, skipping")
            db.mark_error(conversation_id, "empty transcript — nothing to summarize")
            return

        existing_profile = db.get_profile()

        structured = await _extract_structured_info(transcript, existing_profile)
        summary, draft = await _generate_summary_and_draft(transcript)

        topic = (structured.get("topic") or "").strip() or "Untitled discussion"
        tags = structured.get("tags") or existing_profile.get("tags", [])
        notes = (structured.get("notes") or existing_profile.get("notes", "")).strip()

        db.set_conversation_topic(conversation_id, topic)
        db.save_generated_content(conversation_id, summary=summary, substack_draft=draft)
        db.save_profile(tags=tags, notes=notes)

        logger.info(f"[postprocess] conversation {conversation_id} processed successfully")

    except Exception as exc:  # noqa: BLE001
        logger.exception(f"[postprocess] failed for conversation {conversation_id}: {exc}")
        db.mark_error(conversation_id, str(exc)[:500])