"""
Turn router.

Two jobs, both done with cheap local embedding similarity (no network call,
no extra LLM round-trip) so routing adds negligible latency:

1. should_retrieve(utterance) -> bool
   Decides whether this turn needs a Qdrant lookup, or can go straight to
   the LLM as plain chitchat (greetings, fillers, "uh-huh", thanks, meta
   questions about the agent itself, etc).

2. is_conversation_ending(utterance) -> bool
   Semantic (not just keyword) detection of the user signalling they want
   to end the call — "I should get going", "talk to you later", "bye",
   "that's all for now", etc. Works on meaning, not exact phrases, so it
   generalizes beyond the anchor examples below.

Both use the SAME potion-retrieval-32M model instance already loaded for
retrieval, so there's no second model to load.
"""

import numpy as np
from model2vec import StaticModel

# --- Anchor phrases -----------------------------------------------------
# Short prototype utterances for each class. The incoming utterance is
# embedded and compared (cosine similarity) against the centroid of each
# class. This generalizes to paraphrases without needing an exact match.

CHITCHAT_ANCHORS = [
    "hello", "hi there", "hey", "good morning", "how are you",
    "thank you", "thanks a lot", "okay", "got it", "cool",
    "that's interesting", "haha nice", "who are you", "what can you do",
    "can you hear me", "can you repeat that", "sorry what",
    "what's your name", "are you an AI",
]

BOOK_QUERY_ANCHORS = [
    "what happens in the book", "tell me about a character",
    "why does Elizabeth refuse Mr. Darcy's proposal",
    "what is the relationship between Jane and Bingley",
    "describe Mr. Collins", "what does Mrs. Bennet want",
    "summarize chapter one", "what is the theme of the novel",
    "quote something Mr. Darcy said", "what happens at the ball",
    "how does the story end", "what is Wickham's role in the plot",
    "tell me about Pemberley", "why is Lady Catherine angry",
]

ENDING_ANCHORS = [
    "goodbye", "bye for now", "I have to go", "talk to you later",
    "I've got some work to do, it was nice talking to you",
    "that's all for now, thanks", "have a good one", "see you later",
    "I should get going", "let's wrap this up", "that's it for today",
    "thanks for the chat, bye", "okay I'm done, goodbye",
]

# Similarity thresholds, tuned to be conservative (favor chitchat / not-ending
# on ambiguous input, since a false "should_retrieve" just costs a bit of
# latency, while a false "is_ending" would cut the user off mid-sentence).
BOOK_QUERY_THRESHOLD = 0.42
ENDING_THRESHOLD = 0.55


def _centroid(model: StaticModel, phrases: list[str]) -> np.ndarray:
    vecs = model.encode(phrases)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.mean(axis=0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class TurnRouter:
    def __init__(self, model: StaticModel) -> None:
        self._model = model
        self._chitchat_centroid = _centroid(model, CHITCHAT_ANCHORS)
        self._book_centroid = _centroid(model, BOOK_QUERY_ANCHORS)
        self._ending_centroid = _centroid(model, ENDING_ANCHORS)

    def _embed(self, text: str) -> np.ndarray:
        return self._model.encode([text])[0]

    def should_retrieve(self, utterance: str) -> bool:
        # Very short utterances (fillers, acks) never need retrieval.
        if len(utterance.strip().split()) <= 2:
            return False
        vec = self._embed(utterance)
        book_sim = _cosine(vec, self._book_centroid)
        chitchat_sim = _cosine(vec, self._chitchat_centroid)
        return book_sim > BOOK_QUERY_THRESHOLD and book_sim > chitchat_sim

    def is_conversation_ending(self, utterance: str) -> bool:
        vec = self._embed(utterance)
        ending_sim = _cosine(vec, self._ending_centroid)
        return ending_sim > ENDING_THRESHOLD
