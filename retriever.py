"""Find the chunks that best match a question.

Run with: python retriever.py "your question"
"""

import argparse
import sys

from chromadb.errors import NotFoundError

from ingest import COLLECTION_NAME, embed, get_client


def retrieve(query: str, k: int = 4) -> list[dict]:
    """Return up to k chunks, closest first.

    Each result has text, source, page and distance. Distance is cosine
    distance, so lower means a closer match.
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
    count = collection.count()
    if count == 0:
        raise RuntimeError(not_indexed)

    results = collection.query(
        query_embeddings=embed([query]),
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance,
        }
        for text, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
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
