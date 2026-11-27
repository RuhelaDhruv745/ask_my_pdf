"""Measure retrieval quality against the golden questions. No LLM calls.

Run with: python eval.py
Compare chunk settings: python eval.py --chunk-size 1600 --overlap 200

With --chunk-size or --overlap, a temporary in-memory index is built from data/
and chroma_db/ is never opened. Without them, the existing chroma_db/ is used.
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

import ingest
import retriever
from answer import MAX_DISTANCE

GOLDEN_PATH = Path(__file__).resolve().parent / "eval" / "golden.jsonl"
TOP_K = 6


def load_golden(path: Path) -> list[dict]:
    """Read one JSON object per line. Raise ValueError on any invalid line."""
    cases = []
    lines = path.read_text(encoding="utf-8-sig").split("\n")
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        try:
            case = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: invalid JSON ({error.msg}).") from None
        if not isinstance(case, dict):
            raise ValueError(f"{where}: expected a JSON object.")

        question = case.get("question")
        source = case.get("expected_source")
        pages = case.get("expected_pages")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{where}: question must be non-empty text.")
        if not isinstance(source, str):
            raise ValueError(f"{where}: expected_source must be text.")
        if not isinstance(pages, list) or not all(type(p) is int and p >= 1 for p in pages):
            raise ValueError(f"{where}: expected_pages must be a list of page numbers.")
        if pages and not source:
            raise ValueError(f"{where}: expected_source is required when pages are listed.")
        cases.append(case)

    if not cases:
        raise ValueError(f"{path.name} has no questions.")
    return cases


def build_temporary_index(chunk_size: int, overlap: int) -> None:
    """Index data/ in memory and point ingestion and retrieval at that index."""
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    # retriever imported get_client by name, so both modules need the replacement.
    ingest.get_client = retriever.get_client = lambda: client
    ingest.CHUNK_SIZE = chunk_size
    ingest.CHUNK_OVERLAP = overlap
    ingest.ingest()  # exits if a PDF can't be read, so a partial index isn't scored


def index_contents() -> tuple[int, set[str]]:
    """Return the number of chunks and the file names in the index being evaluated."""
    try:
        collection = retriever.get_client().get_collection(
            ingest.COLLECTION_NAME, embedding_function=None
        )
    except NotFoundError:
        raise RuntimeError("No index found. Run: python ingest.py") from None
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    return len(metadatas), {metadata["source"] for metadata in metadatas}


def evaluate(case: dict) -> dict:
    """Retrieve for one question. rank is the position of the first hit, or None."""
    results = retriever.retrieve(case["question"], k=TOP_K)
    rank = next(
        (
            position
            for position, result in enumerate(results, start=1)
            # Match the file too: different PDFs share page numbers.
            if result["source"] == case["expected_source"]
            and result["page"] in case["expected_pages"]
        ),
        None,
    )
    return {**case, "results": results, "rank": rank, "top": results[0]["distance"]}


def average_top(rows: list[dict]) -> str:
    return f"{sum(row['top'] for row in rows) / len(rows):.4f}" if rows else "n/a"


def print_report(rows: list[dict]) -> None:
    answerable = [row for row in rows if row["expected_pages"]]
    unanswerable = [row for row in rows if not row["expected_pages"]]

    print(f"\nAnswerable questions (hit = expected file and page in the top {TOP_K}):")
    for row in answerable:
        status = "HIT " if row["rank"] else "MISS"
        pages = ", ".join(str(page) for page in row["expected_pages"])
        found = f"first hit at rank {row['rank']}" if row["rank"] else "not retrieved"
        print(f"  {status}  top {row['top']:.4f}  {row['question']}")
        print(f"          expected page(s) {pages}: {found}")
        if not row["rank"]:
            print(f"          expected file: {row['expected_source']}")
            for result in row["results"]:
                print(f"          got: {result['source']} p{result['page']}")

    print(f"\nUnanswerable questions (cutoff {MAX_DISTANCE}):")
    for row in unanswerable:
        status = "BELOW" if row["top"] < MAX_DISTANCE else "ABOVE"
        print(f"  {status}  top {row['top']:.4f}  {row['question']}")

    hits = sum(1 for row in answerable if row["rank"])
    rate = f"{hits}/{len(answerable)} ({hits / len(answerable):.1%})" if answerable else "n/a"
    below = sum(1 for row in unanswerable if row["top"] < MAX_DISTANCE)
    print("\nSummary")
    print(f"  Hit rate, answerable questions: {rate}")
    print(f"  Unanswerable with top chunk below {MAX_DISTANCE}: {below}/{len(unanswerable)}")
    print(f"  Average top-chunk distance, answerable:   {average_top(answerable)}")
    print(f"  Average top-chunk distance, unanswerable: {average_top(unanswerable)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on eval/golden.jsonl.")
    parser.add_argument("--chunk-size", type=int, help="build a temporary index with this size")
    parser.add_argument("--overlap", type=int, help="build a temporary index with this overlap")
    args = parser.parse_args()

    temporary = args.chunk_size is not None or args.overlap is not None
    chunk_size = ingest.CHUNK_SIZE if args.chunk_size is None else args.chunk_size
    overlap = ingest.CHUNK_OVERLAP if args.overlap is None else args.overlap
    # split_text only keeps moving forward if the overlap is under half the size.
    if temporary and not (chunk_size > 0 and 0 <= overlap < chunk_size // 2):
        parser.error(
            f"need chunk size > 0 and 0 <= overlap < half the chunk size "
            f"(got size {chunk_size}, overlap {overlap})"
        )

    # Questions and file names can contain characters the Windows console can't display.
    sys.stdout.reconfigure(errors="replace")

    try:
        cases = load_golden(GOLDEN_PATH)
        if temporary:
            build_temporary_index(chunk_size, overlap)
        count, indexed = index_contents()

        # A typo or an out-of-date index would otherwise look like a retrieval miss.
        missing = {c["expected_source"] for c in cases if c["expected_pages"]} - indexed
        if missing:
            sys.exit(f"Expected files not in the index: {', '.join(sorted(missing))}")

        rows = [evaluate(case) for case in cases]
    except (OSError, RuntimeError, ValueError) as error:
        sys.exit(str(error))

    if temporary:
        description = f"temporary, chunk size {chunk_size}, overlap {overlap}"
    else:
        description = "existing chroma_db/ (its chunk settings aren't recorded)"
    print(f"\nIndex: {description}; {count} chunks from {len(indexed)} files")
    print_report(rows)


if __name__ == "__main__":
    main()
