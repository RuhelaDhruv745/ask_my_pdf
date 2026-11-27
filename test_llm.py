"""Smoke test: send one short question to the LLM and print the reply.

Run with: python test_llm.py
"""

import sys

from openai import APIError

from llm import ask_llm


def main() -> None:
    try:
        reply = ask_llm(
            system="You are a helpful assistant. Answer in one short sentence.",
            user="What is the capital of France?",
        )
    except APIError as error:
        # Don't show the full message or traceback: some providers echo part of the key.
        status = getattr(error, "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        sys.exit(f"LLM request failed: {type(error).__name__}{detail}")
    except RuntimeError as error:
        sys.exit(str(error))

    print(reply)


if __name__ == "__main__":
    main()
