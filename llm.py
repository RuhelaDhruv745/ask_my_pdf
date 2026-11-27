"""The single entry point for LLM calls in this project."""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

# utf-8-sig also handles a .env saved with a byte-order mark (common on Windows).
load_dotenv(Path(__file__).with_name(".env"), encoding="utf-8-sig")

MAX_RETRIES = 3  # retries after the first attempt
BASE_DELAY_SECONDS = 1.0  # waits are 1s, 2s, 4s
TIMEOUT_SECONDS = 60.0

# Temporary failures worth retrying. APIConnectionError includes timeouts, and
# InternalServerError covers 5xx responses. Other errors (bad key, unknown
# model, bad request) are raised immediately.
RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, InternalServerError)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


def ask_llm(system: str, user: str, temperature: float = 0.2) -> str:
    """Send a system and user message to the model and return the reply text."""
    base_url = _require_env("LLM_BASE_URL")
    model = _require_env("LLM_MODEL")
    api_key = _require_env("LLM_API_KEY")

    # max_retries=0 turns off the client's own retries so they don't stack with ours.
    with OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
    ) as client:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                )
                break
            except RETRYABLE_ERRORS as error:
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_DELAY_SECONDS * 2**attempt
                # Log only the error type: provider messages may echo request details.
                print(
                    f"LLM call failed ({type(error).__name__}). "
                    f"Retry {attempt + 1}/{MAX_RETRIES} in {delay:.0f}s.",
                    file=sys.stderr,
                )
                time.sleep(delay)

    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise RuntimeError("The LLM returned an empty reply.")
    return content
