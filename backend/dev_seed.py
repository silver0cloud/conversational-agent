"""
Dev-only helper: seed a fake, fully-processed conversation without doing a
real voice call.

Creates a conversation with a realistic sample transcript, ends it, then
runs the REAL Phase 2 post-processing against it (actual Groq API calls,
using your real key from .env) — so you get a genuine summary + Substack
draft to look at, not placeholder text.

Usage:
    python -m backend.dev_seed

Once Phase 3 (the /profile dashboard) exists, this conversation will just
be sitting there in the list, fully processed, ready to view — no need to
do a live voice call every time you want to test the dashboard UI.

Edit SAMPLE_TRANSCRIPT below to try different conversation shapes (e.g. a
much longer discussion, or one that's mostly chitchat) if you want to see
how the summary/draft generation handles different inputs.
"""

import asyncio

from loguru import logger

from backend import db, postprocess

SAMPLE_TRANSCRIPT: list[tuple[str, str]] = [
    ("assistant", "Hey! Great to talk with you. What drew you to Pride and Prejudice in the first place?"),
    ("user", "Honestly I just love how petty and catty everyone is at the balls. It's like reality TV."),
    ("assistant", "Ha, fair! Austen basically invented the reality-TV confessional two centuries early. "
                  "Any particular scene that stuck with you?"),
    ("user", "The one where Mr. Collins proposes to Elizabeth. It's so cringe but also kind of sad for him."),
    ("assistant", "It's a masterclass in secondhand embarrassment. What do you make of Elizabeth's reasons "
                  "for turning him down — is it just that he's awkward, or something deeper?"),
    ("user", "I think it's that she refuses to marry just for security, even though everyone around her thinks "
             "she's crazy not to. Charlotte does the opposite and just takes the practical option."),
    ("assistant", "Right, and Austen doesn't really let you fully condemn Charlotte for it either — she's "
                  "pretty clear-eyed about her own choice. It sets up a nice contrast with Elizabeth's idealism."),
    ("user", "Yeah, I actually felt for Charlotte more than I expected to. Anyway, I should get going — "
             "this was fun though."),
    ("assistant", "It really was — talk soon!"),
]


async def seed() -> int:
    db.init_db()

    conversation_id = db.create_conversation()
    for role, content in SAMPLE_TRANSCRIPT:
        db.add_turn(conversation_id, role, content)
    db.end_conversation(conversation_id)

    logger.info(f"[dev_seed] created conversation {conversation_id}, running real post-processing...")
    await postprocess.process_conversation(conversation_id)

    conv = db.get_conversation(conversation_id)
    profile = db.get_profile()

    print()
    print(f"=== Conversation {conversation_id} ({conv['status']}) ===")
    print(f"Topic: {conv['topic']}")
    print()
    print("--- Summary ---")
    print(conv["summary"])
    print()
    print("--- Substack draft ---")
    print(conv["substack_draft"])
    print()
    print("--- Updated profile ---")
    print("Tags:", profile["tags"])
    print("Notes:", profile["notes"])
    print()

    if conv["status"] == "error":
        print(f"NOTE: processing failed — {conv['error_message']}")
        print("Check your GROQ_API_KEY in .env and that the server can reach api.groq.com.")

    return conversation_id


if __name__ == "__main__":
    asyncio.run(seed())