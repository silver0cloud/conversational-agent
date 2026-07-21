# Pride & Prejudice — Voice RAG Companion

A voice agent you talk to about *Pride and Prejudice*, built entirely on your
own infra — no Daily.co, no per-minute call fees. WebRTC runs through
Pipecat's `SmallWebRTCTransport`, self-hosted on your FastAPI server.

## Stack

| Component     | Choice                                      |
|---------------|----------------------------------------------|
| STT           | Deepgram `nova-3`                             |
| Embeddings    | `minishlab/potion-retrieval-32M` (model2vec)  |
| LLM           | Qwen3-235B-A22B-Instruct-2507, via **Groq**   |
| TTS           | Cartesia `sonic-2`                            |
| Turn detection| Pipecat `smart-turn-v3`                       |
| Vector DB     | Qdrant (local, via Docker)                    |
| Transport     | Pipecat `SmallWebRTCTransport` (free, self-hosted) |

## How the "smart routing" works

`backend/rag/router.py` embeds every user utterance with the same
potion-retrieval model already loaded for retrieval (no second model, no
extra network call) and compares it against small anchor-phrase sets for
"chitchat" vs. "book question" using cosine similarity. Only utterances that
land closer to the book-question cluster trigger a Qdrant lookup — greetings,
fillers, thanks, and meta questions skip retrieval entirely, so latency for
casual turns stays low.

The same trick detects end-of-conversation intent semantically (not just
keyword matching on "bye") — see `ENDING_ANCHORS` in that file. When it
fires, the pipeline lets the LLM produce one short farewell line, waits for
`BotStoppedSpeakingFrame` (i.e. the farewell has finished playing), then
sends `EndFrame` to close the session — no click required.

**Tuning note:** the similarity thresholds (`BOOK_QUERY_THRESHOLD`,
`ENDING_THRESHOLD`) were set conservatively based on the anchor phrases, but
I couldn't download the actual potion-retrieval-32M weights in the sandbox
this was built in (no Hugging Face network access there) to empirically
tune them. Run `python -m backend.rag.router_selftest` (see below) after
setup to sanity-check the thresholds against your own test phrases before
relying on it live.

## Setup

```bash
cd voice-rag-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env
# fill in DEEPGRAM_API_KEY, CARTESIA_API_KEY, GROQ_API_KEY in .env

# start local Qdrant
docker compose up -d

# one-time: ingest the book into Qdrant
python -m backend.rag.ingest --pdf data/pride-and-prejudice.pdf

# run the server
python -m backend.main
```

Open **http://localhost:7860** — tap the orb's mic button, allow mic access,
and start talking.

## UI behavior

- **Idle**: mic button under the orb. Tap to start.
- **In call**: mic button disappears, a horizontal waveform of your live mic
  input takes its place, scrolling right→left.
- The orb itself grows/pulses with whichever side is currently louder — you
  or the agent — driven directly by live audio amplitude (Web Audio API
  `AnalyserNode`), no extra signaling needed from the backend for this.
- **Ending a call**: click the orb any time, or just say something like
  "alright, I've got to go — bye!" and the agent will say a short goodbye and
  hang up on its own.

## Known gaps / things to verify on your machine

I built and import-checked every backend module against the actual installed
`pipecat-ai==1.5.0` API (verified exact constructor signatures, frame types,
etc. — not from memory) — see inline comments. What I could **not** do in
this sandbox, since it only has network access to package registries, not to
Deepgram/Cartesia/Groq/Hugging Face:

- Run a live end-to-end call
- Download `potion-retrieval-32M` to tune the router's similarity thresholds
- Confirm the exact Cartesia `voice_id` in `.env.example` is still valid on
  your account (swap it for any voice id from your Cartesia dashboard)

Budget your first run as a debugging pass, not a guaranteed one-shot —
real-time voice pipelines with five external services almost always need at
least one round of "check the logs, fix a param name" even when every piece
is individually correct.

## Extending beyond one book

To go multi-book: add a `book_id` payload field in `ingest.py`, filter
`retriever.py`'s Qdrant query by it, and pass the active book into the
system prompt / a session-level variable. At that scale you'd probably also
want Qdrant Cloud instead of local Docker.
