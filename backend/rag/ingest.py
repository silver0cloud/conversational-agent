"""
One-time ingestion script.

Reads the book PDF, splits it into overlapping chunks along paragraph/sentence
boundaries, embeds each chunk with model2vec's potion-retrieval-32M, and
upserts everything into a Qdrant collection.

Run once (or whenever the source PDF changes):

    python -m backend.rag.ingest --pdf data/pride-and-prejudice.pdf
"""

import argparse
import re
import sys
from pathlib import Path

from model2vec import StaticModel
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.config import settings  # noqa: E402


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    # Collapse the "Free eBooks at Planet eBook.com"-style running headers/footers
    # and normalize whitespace.
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Chunk on paragraph boundaries first, then greedily pack paragraphs into
    ~chunk_size windows with a sliding overlap so retrieval doesn't lose
    context at chunk edges (important for dialogue-heavy prose like Austen).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying the tail of the previous one for overlap
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n{para}".strip()
    if current:
        chunks.append(current)

    return chunks


def build_collection(client: QdrantClient, dim: int) -> None:
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default="data/pride-and-prejudice.pdf")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    print(f"[ingest] Extracting text from {pdf_path} ...")
    text = extract_text(pdf_path)

    print(f"[ingest] Chunking (size={settings.chunk_size_chars}, overlap={settings.chunk_overlap_chars}) ...")
    chunks = chunk_text(text, settings.chunk_size_chars, settings.chunk_overlap_chars)
    print(f"[ingest] Produced {len(chunks)} chunks.")

    print(f"[ingest] Loading embedding model {settings.embedding_model_name} ...")
    model = StaticModel.from_pretrained(settings.embedding_model_name)

    print("[ingest] Embedding chunks ...")
    vectors = model.encode(chunks, show_progress_bar=True)
    dim = len(vectors[0])

    if settings.qdrant_mode == "local":
        client = QdrantClient(url=settings.qdrant_url)
    else:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    print(f"[ingest] Creating Qdrant collection '{settings.qdrant_collection}' (dim={dim}) ...")
    build_collection(client, dim)

    print("[ingest] Upserting points ...")
    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": chunks[i], "chunk_index": i})
        for i in range(len(chunks))
    ]
    # batch upsert to keep request sizes sane
    batch = 128
    for start in range(0, len(points), batch):
        client.upsert(collection_name=settings.qdrant_collection, points=points[start : start + batch])

    print(f"[ingest] Done. {len(points)} chunks indexed into '{settings.qdrant_collection}'.")


if __name__ == "__main__":
    main()
