"""
Phase 3 API routes: everything the /profile dashboard needs to read.

All read-only. Kept as a separate router (rather than piling into main.py)
since this is a distinct, growing surface area from the WebRTC signaling
endpoint.
"""

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from backend import db

router = APIRouter(prefix="/api", tags=["profile"])


@router.get("/profile")
async def get_profile():
    return await asyncio.to_thread(db.get_profile)


@router.get("/conversations")
async def list_conversations():
    return await asyncio.to_thread(db.list_conversations)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int):
    conv = await asyncio.to_thread(db.get_conversation, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/conversations/{conversation_id}/draft.md")
async def download_draft(conversation_id: int):
    conv = await asyncio.to_thread(db.get_conversation, conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if not conv.get("substack_draft"):
        raise HTTPException(status_code=404, detail="No draft generated yet for this conversation")

    filename = f"substack-draft-{conversation_id}.md"
    return Response(
        content=conv["substack_draft"],
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
