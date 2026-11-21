
import os

from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm() -> ChatGoogleGenerativeAI:
    """Create the shared Gemini chat model."""

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    if not model_name:
        raise ValueError("GEMINI_MODEL is not configured.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_retries=2,
    )
