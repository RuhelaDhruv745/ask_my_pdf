"""Index every PDF in data/ into a persistent ChromaDB collection.

Run with: python ingest.py

Safe to re-run: each chunk has a fixed ID, so existing chunks are updated in
place, and chunks that no longer exist are deleted.
"""

import sys
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 800  # maximum characters per chunk
CHUNK_OVERLAP = 120  # characters shared by neighbouring chunks (less than half of CHUNK_SIZE)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "pdf_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once per process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()


def get_client():
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )


def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks, breaking between words where possible."""
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            # Back up to the last whitespace in the second half of the chunk.
            while end > start + CHUNK_SIZE // 2 and not text[end].isspace():
                end -= 1
            if not text[end].isspace():
                end = start + CHUNK_SIZE  # no whitespace found: cut mid-word

        chunks.append(text[start:end].strip())
        if end >= len(text):
            break

        overlap_start = end - CHUNK_OVERLAP
        start = overlap_start
        # Move forward so the next chunk doesn't begin in the middle of a word.
        while start < end and not text[start - 1].isspace():
            start += 1
        if start == end:
            start = overlap_start
    return [chunk for chunk in chunks if chunk]


def chunk_pdf(path: Path) -> list[dict]:
    """Return the chunks for one PDF. Page numbers start at 1."""
    chunks = []
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        for index, text in enumerate(split_text(page.extract_text() or "")):
            chunks.append(
                {
                    "id": f"{path.name}::p{page_number}::c{index}",
                    "text": text,
                    "metadata": {"source": path.name, "page": page_number},
                }
            )
    return chunks


def ingest() -> None:
    pdf_paths = (
        sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
        if DATA_DIR.is_dir()
        else []
    )
    if not pdf_paths:
        # Stop here so a missing or empty folder can't wipe the existing index.
        sys.exit(f"No PDFs found in {DATA_DIR}. The index was not changed.")

    chunks = []
    failed = set()
    for path in pdf_paths:
        try:
            pdf_chunks = chunk_pdf(path)
        except Exception as error:  # one bad PDF shouldn't stop the others
            failed.add(path.name)
            print(f"Could not read {path.name}: {error}", file=sys.stderr)
            continue
        if not pdf_chunks:
            print(f"No text found in {path.name} (it may be a scanned PDF).", file=sys.stderr)
        print(f"{path.name}: {len(pdf_chunks)} chunks")
        chunks.extend(pdf_chunks)

    client = get_client()
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,  # embeddings are computed here and passed in
    )

    batch_size = client.get_max_batch_size()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=texts,
            embeddings=embed(texts),
            metadatas=[chunk["metadata"] for chunk in batch],
        )

    # Delete leftovers from removed or changed PDFs. Files that failed to read
    # keep their old chunks.
    current_ids = {chunk["id"] for chunk in chunks}
    stored = collection.get(include=["metadatas"])
    stale_ids = [
        chunk_id
        for chunk_id, metadata in zip(stored["ids"], stored["metadatas"])
        if chunk_id not in current_ids and metadata["source"] not in failed
    ]
    for start in range(0, len(stale_ids), batch_size):
        collection.delete(ids=stale_ids[start : start + batch_size])

    print(f"Removed {len(stale_ids)} old chunks. The index now has {collection.count()}.")
    if failed:
        sys.exit(f"{len(failed)} PDF(s) could not be read.")


if __name__ == "__main__":
    ingest()
