"""
Quick manual sanity check for TurnRouter's routing decisions.
Run after setup:

    python -m backend.rag.router_selftest

Prints should_retrieve / is_conversation_ending for a mix of test utterances
so you can eyeball whether the thresholds in router.py need adjusting for
your accent/phrasing before relying on this live.
"""

from model2vec import StaticModel

from backend.config import settings
from backend.rag.router import TurnRouter

TEST_UTTERANCES = [
    "hey how's it going",
    "thanks, that makes sense",
    "why did Elizabeth reject Darcy's first proposal?",
    "what's Mr. Collins like as a person",
    "can you tell me about the Bennet family",
    "uh-huh",
    "cool, got it",
    "alright I've got some stuff to do, it was nice talking to you",
    "bye, have a good one",
    "so what happens after the ball at Netherfield",
]


def main() -> None:
    model = StaticModel.from_pretrained(settings.embedding_model_name)
    router = TurnRouter(model)

    print(f"{'utterance':<65} {'retrieve?':<10} {'ending?'}")
    print("-" * 90)
    for u in TEST_UTTERANCES:
        print(f"{u:<65} {str(router.should_retrieve(u)):<10} {router.is_conversation_ending(u)}")


if __name__ == "__main__":
    main()
