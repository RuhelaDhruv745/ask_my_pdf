"""Find the chunks that best match a question.

Run with: python retriever.py "your question"
"""

import argparse
import sys

import numpy as np
from chromadb.errors import NotFoundError

from ingest import COLLECTION_NAME, embed, get_client


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Return up to k chunks, closest first.

    Each result has text, source, page and distance. Distance is cosine
    distance, so lower means a closer match.

    The search is exact: the question is compared with every stored chunk.
    Chroma's approximate query sometimes missed close matches in this project.
    """
    if not query.strip():
        raise ValueError("The query is empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    not_indexed = "No index found. Run: python ingest.py"
    try:
        collection = get_client().get_collection(COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        raise RuntimeError(not_indexed) from None
    stored = collection.get(include=["embeddings", "documents", "metadatas"])
    if not stored["ids"]:
        raise RuntimeError(not_indexed)

    # Cosine distance = 1 - cosine similarity.
    vectors = np.asarray(stored["embeddings"], dtype=float)
    question = np.asarray(embed([query])[0], dtype=float)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(question)
    distances = 1 - (vectors @ question) / norms

    # Sort by distance, then by ID so that ties always come out in the same order.
    closest = np.lexsort((stored["ids"], distances))[:k]
    return [
        {
            "text": stored["documents"][i],
            "source": stored["metadatas"][i]["source"],
            "page": stored["metadatas"][i]["page"],
            "distance": float(distances[i]),
        }
        for i in closest
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the indexed PDFs.")
    parser.add_argument("query", help="your question, in quotes")
    parser.add_argument("-k", type=int, default=4, help="number of results (default: 4)")
    args = parser.parse_args()

    # PDF text can contain characters the Windows console can't display.
    sys.stdout.reconfigure(errors="replace")

    try:
        results = retrieve(args.query, args.k)
    except (RuntimeError, ValueError) as error:
        sys.exit(str(error))

    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] {result['source']} (page {result['page']}), "
              f"distance {result['distance']:.3f}")
        print(result["text"])


if __name__ == "__main__":
    main()
