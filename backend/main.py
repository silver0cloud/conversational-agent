"""
Entry point.

    python -m backend.main

Serves:
  - the frontend (orb UI) at /
  - the profile dashboard at /profile
  - POST /api/offer  -> WebRTC signaling (SmallWebRTC, no Daily.co, no cost)
  - GET  /api/profile, /api/conversations, /api/conversations/{id},
    /api/conversations/{id}/draft.md -> dashboard data (see routes_profile.py)

Each browser tab that hits "start call" gets its own SmallWebRTCConnection
and its own Pipecat pipeline instance (STT -> router -> LLM -> TTS).
"""

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from backend.config import settings
from backend.pipeline import build_pipeline
from backend import db, postprocess, routes_profile

app = FastAPI(title="Pride and Prejudice Voice Agent")
app.include_router(routes_profile.router)


@app.on_event("startup")
async def on_startup() -> None:
    db.init_db()
    logger.info(f"Database ready at {settings.db_path}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

request_handler = SmallWebRTCRequestHandler()

# Strong references to in-flight background postprocessing tasks. Without
# this, asyncio.create_task()'s result can be garbage-collected before it
# finishes — a real risk here since each task runs multiple LLM calls over
# many seconds. Tasks remove themselves from this set on completion.
_background_tasks: set[asyncio.Task] = set()


def _run_in_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.post("/api/offer")
async def offer(request: dict):
    """Standard SmallWebRTC offer/answer signaling endpoint."""
    # Fail fast with a clear HTTP error if keys are missing, rather than
    # accepting a WebRTC connection that will silently never produce audio
    # (the previous behavior: a 200 OK with a connection that just times out).
    try:
        settings.validate_runtime_keys()
    except RuntimeError as exc:
        logger.error(f"Refusing /api/offer: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    small_request = SmallWebRTCRequest.from_dict(request)

    async def on_connection(webrtc_connection):
        task, runner, conversation_id = build_pipeline(webrtc_connection)

        async def run_pipeline():
            try:
                await runner.run(task)
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Pipeline error: {exc}")
            finally:
                # Runs regardless of how the call ended (goodbye detection,
                # orb click, or a dropped connection) so no conversation is
                # ever left stuck in 'in_progress'.
                await asyncio.to_thread(db.end_conversation, conversation_id)
                logger.info(f"[main] conversation {conversation_id} closed")

                # Fire-and-forget: summary/draft generation happens in the
                # background so hanging up feels instant. Errors are caught
                # and recorded on the conversation row inside
                # process_conversation itself, never raised here.
                _run_in_background(postprocess.process_conversation(conversation_id))

        _run_in_background(run_pipeline())

    answer = await request_handler.handle_web_request(small_request, on_connection)
    return answer


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/profile")
async def profile_page():
    """Serves the dashboard page at a clean URL (no .html extension).
    Registered before the StaticFiles mount below so it takes precedence."""
    return FileResponse(FRONTEND_DIR / "profile.html")


# Serve the orb UI (and profile.html/css/js as static assets). Mounted last
# so /api/* and /profile above take precedence.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
