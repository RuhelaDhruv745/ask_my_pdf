# Ask My PDFs
Small multi-agent RAG project for learning.

## Stack
- Python 3.11+, openai client (OpenAI-compatible), python-dotenv
- Later: sentence-transformers, ChromaDB

## Rules
- LLM calls only through llm.py. Provider set by LLM_BASE_URL, LLM_MODEL, LLM_API_KEY in .env.
- Never hard-code or print secrets.
- Small changes. Summarise what changed after each step.