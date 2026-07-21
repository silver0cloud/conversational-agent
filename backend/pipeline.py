"""
The voice agent pipeline.

Flow:
  mic audio
    -> VAD / turn detection (smart-turn-v3)
    -> STT (Deepgram nova-3)
    -> RAGRouterProcessor   <-- decides per-turn: chitchat or book lookup,
                                 whether the user is ending the call, and
                                 logs the user's turn to the conversation DB
    -> LLM context aggregation (system prompt varies: first-time users get
       a warm "get to know you" persona; returning users get a persona that
       knows their stored profile and eases into "what to talk about today")
    -> LLM (Qwen3.6-27B via Groq)
    -> AssistantTurnLogger  <-- buffers streamed LLM text into one clean
                                 turn and logs it to the conversation DB
    -> TTS (Cartesia sonic-2)
    -> EndCallProcessor     <-- hangs up right after the farewell audio
                                 finishes playing, if the user said bye
    -> speaker audio (over the SmallWebRTC transport, no Daily.co)

Each call gets its own `conversations` row (backend/db) created at pipeline
build time and closed out by the caller (see main.py) once the pipeline task
finishes, regardless of whether the call ended via the "goodbye" detection,
a click on the orb, or a dropped connection.
"""

import asyncio

from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from backend import db
from backend.config import settings
from backend.rag.retriever import BookRetriever
from backend.rag.router import TurnRouter

BASE_INSTRUCTIONS = """You are a warm, knowledgeable voice companion who has read \
Jane Austen's "Pride and Prejudice" closely and loves discussing it.

Keep replies SHORT and conversational — you are being spoken aloud, not read. \
One to three sentences unless the user clearly wants more detail. No lists, \
no markdown, no stage directions.

For greetings, small talk, or questions about yourself, just chat naturally.

When the message includes a block starting with "Relevant passages from \
Pride and Prejudice:", ground your answer in those passages. If they don't \
actually answer the question, say you're not sure rather than making up plot \
details.

If the user seems to be ending the conversation, give a brief, warm goodbye \
in one short sentence — do not ask a follow-up question."""

ONBOARDING_PERSONA = """This is your very first conversation with this person — you've \
never spoken before. Do NOT open by asking what they want to discuss, and do \
NOT jump straight into the book. Instead, be genuinely warm and curious: \
briefly introduce yourself as a companion for discussing Pride and Prejudice, \
then get to know them a little — ask about their interests, why they picked \
up this book, what genres or themes they're drawn to, what kind of reader \
they are. Ask ONE question at a time and react naturally to what they say, \
like a real first conversation, not an interview. Only after a few genuine \
exchanges getting to know them should you naturally ease into asking what \
part of the book, or which character or theme, they'd like to dig into today."""

RETURNING_PERSONA_TEMPLATE = """You've spoken with this person before. Here's what you \
know about them so far:
{profile_block}

Open warmly with a quick, genuine check-in — reference something you know \
about them if it fits naturally, but don't recite it back like a report. \
After a bit of natural warm-up, ease into asking what they'd like to talk \
about today — don't lead with that question bluntly as your very first line."""

# Keep injected profile text bounded so it can't grow the prompt unbounded
# over many conversations.
_MAX_NOTES_CHARS = 600


def build_system_prompt(is_first_time: bool, profile: dict) -> str:
    if is_first_time:
        persona = ONBOARDING_PERSONA
    else:
        tags = profile.get("tags") or []
        notes = (profile.get("notes") or "").strip()[:_MAX_NOTES_CHARS]

        lines = []
        if tags:
            lines.append(f"- Interests: {', '.join(tags)}")
        if notes:
            lines.append(f"- Notes from past conversations: {notes}")
        profile_block = "\n".join(lines) if lines else "- (no notes yet)"

        persona = RETURNING_PERSONA_TEMPLATE.format(profile_block=profile_block)

    return f"{persona}\n\n{BASE_INSTRUCTIONS}"


class RAGRouterProcessor(FrameProcessor):
    """
    Sits just before the LLM context aggregator consumes the user's turn.
    For each final transcription:
      - decides (via cheap embedding similarity, no LLM call) whether to
        pull context from Qdrant, and if so, stashes it so it can be
        stitched onto the next LLMContextFrame
      - decides whether the user is signalling the end of the call, and if
        so, tags the frame so the pipeline can hang up after the farewell
    """

    def __init__(
        self,
        retriever: BookRetriever,
        router: TurnRouter,
        on_ending_detected,
        conversation_id: int,
    ) -> None:
        super().__init__()
        self._retriever = retriever
        self._router = router
        self._pending_context: str | None = None
        self._on_ending_detected = on_ending_detected
        self._conversation_id = conversation_id

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            user_text = frame.text.strip()

            # sqlite3 is blocking I/O — run it off the event loop so a slow
            # disk write never introduces audio hitches. This frame isn't on
            # the raw audio hot path, so a brief await here is safe.
            await asyncio.to_thread(db.add_turn, self._conversation_id, "user", user_text)

            if self._router.should_retrieve(user_text):
                logger.debug(f"[router] retrieving context for: {user_text!r}")
                self._pending_context = self._retriever.build_context_block(user_text)
            else:
                logger.debug(f"[router] chitchat, skipping RAG for: {user_text!r}")
                self._pending_context = None

            if self._router.is_conversation_ending(user_text):
                logger.info(f"[router] end-of-conversation detected: {user_text!r}")
                await self._on_ending_detected()

        elif isinstance(frame, LLMContextFrame) and self._pending_context:
            # Append retrieved passages to the latest user message so the
            # LLM sees them as part of what the user just said.
            messages = frame.context.get_messages()
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    msg["content"] = f"{msg['content']}\n\n{self._pending_context}"
                    break
            self._pending_context = None

        await self.push_frame(frame, direction)


class AssistantTurnLogger(FrameProcessor):
    """
    Positioned right after the LLM. Buffers the streamed LLMTextFrame chunks
    between LLMFullResponseStartFrame/EndFrame into one clean turn of text
    (rather than logging fragmented streaming pieces) and persists it.
    """

    def __init__(self, conversation_id: int) -> None:
        super().__init__()
        self._conversation_id = conversation_id
        self._buffer: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
        elif isinstance(frame, LLMTextFrame):
            self._buffer.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            full_text = "".join(self._buffer).strip()
            self._buffer = []
            if full_text:
                await asyncio.to_thread(db.add_turn, self._conversation_id, "assistant", full_text)

        await self.push_frame(frame, direction)


class EndCallProcessor(FrameProcessor):
    """
    Once the router flags that the user is ending the call, this waits for
    the bot to finish speaking its farewell line and then cleanly ends the
    pipeline task (closing the WebRTC connection).
    """

    def __init__(self) -> None:
        super().__init__()
        self._should_end_after_next_bot_turn = False

    def arm(self) -> None:
        self._should_end_after_next_bot_turn = True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame) and self._should_end_after_next_bot_turn:
            logger.info("[end_call] farewell finished playing, ending session")
            await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)


def build_pipeline(webrtc_connection) -> tuple[PipelineTask, PipelineRunner, int]:
    settings.validate_runtime_keys()

    is_first_time = not db.has_completed_conversation()
    profile = db.get_profile()
    conversation_id = db.create_conversation()
    logger.info(
        f"[pipeline] conversation {conversation_id} starting "
        f"({'first-time' if is_first_time else 'returning'} user)"
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=settings.vad_stop_secs)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    )

    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(model=settings.deepgram_model, smart_format=True),
    )

    tts = CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        settings=CartesiaTTSService.Settings(voice=settings.cartesia_voice_id, model=settings.cartesia_model),
    )

    llm = GroqLLMService(
        api_key=settings.groq_api_key,
        settings=GroqLLMService.Settings(
            model=settings.groq_model,
            # Critical for a voice agent: qwen3.6-27b is a reasoning model that
            # otherwise emits its <think>...</think> chain-of-thought as plain
            # text, which our pipeline streams straight to TTS — meaning the
            # agent would literally speak its internal reasoning out loud.
            # "none" puts it in non-thinking/conversational mode: faster,
            # cheaper, and it only ever says the actual answer.
            extra={"reasoning_effort": "none"},
        ),
    )

    retriever = BookRetriever()
    router = TurnRouter(retriever._model)  # reuse the same embedding model instance

    system_prompt = build_system_prompt(is_first_time, profile)
    context = LLMContext([{"role": "system", "content": system_prompt}])
    context_aggregator = LLMContextAggregatorPair(context)

    end_call_processor = EndCallProcessor()

    async def on_ending_detected() -> None:
        end_call_processor.arm()

    rag_router = RAGRouterProcessor(retriever, router, on_ending_detected, conversation_id)
    assistant_logger = AssistantTurnLogger(conversation_id)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rag_router,
            context_aggregator.user(),
            llm,
            assistant_logger,
            tts,
            end_call_processor,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()
    return task, runner, conversation_id