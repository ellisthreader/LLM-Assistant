"""LLM chat completion integration."""

from __future__ import annotations

import time
from collections.abc import Iterator

from openai import OpenAI

from config import (
    ENABLE_WEB_SEARCH,
    MAX_RETRIES,
    MODEL_NAME,
    OPENAI_API_KEY,
    RETRY_DELAY_SECONDS,
    SYSTEM_PROMPT,
)

def _extract_text_from_response(response: object) -> str:
    """Best-effort extraction for text from Responses API payloads."""
    output_text = getattr(response, "output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = getattr(response, "output", None)
    if not output_items:
        return ""

    chunks: list[str] = []
    for item in output_items:
        if getattr(item, "type", "") != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                text = getattr(content, "text", "")
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


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
            if ENABLE_WEB_SEARCH:
                try:
                    response = api_client.responses.create(
                        model=MODEL_NAME,
                        input=messages,
                        tools=[{"type": "web_search_preview"}],
                    )
                    reply = _extract_text_from_response(response)
                    if reply:
                        return reply
                except Exception as web_exc:
                    print(f"[LLM] Web search path unavailable, falling back: {web_exc}")

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


def stream_reply_tokens(
    user_text: str,
    conversation_history: list[dict[str, str]] | None = None,
    client: OpenAI | None = None,
    max_retries: int = MAX_RETRIES,
) -> Iterator[str]:
    """Stream assistant response token-by-token for low-latency TTS."""
    api_client = client or OpenAI(api_key=OPENAI_API_KEY)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    for attempt in range(1, max_retries + 1):
        try:
            stream = api_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            return
        except Exception as exc:
            print(f"[LLM-STREAM] Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY_SECONDS)
