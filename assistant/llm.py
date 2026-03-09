"""LLM chat completion integration."""

from __future__ import annotations

import time

from openai import OpenAI

from config import (
    MAX_RETRIES,
    MODEL_NAME,
    OPENAI_API_KEY,
    RETRY_DELAY_SECONDS,
    SYSTEM_PROMPT,
)


def generate_reply(
    user_text: str,
    conversation_history: list[dict[str, str]] | None = None,
    client: OpenAI | None = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Send user text to the chat model and return the assistant response."""
    api_client = client or OpenAI(api_key=OPENAI_API_KEY)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    for attempt in range(1, max_retries + 1):
        try:
            result = api_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
            )
            reply = (
                result.choices[0].message.content.strip()
                if result.choices and result.choices[0].message.content
                else ""
            )
            return reply
        except Exception as exc:
            print(f"[LLM] Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY_SECONDS)

    return ""
