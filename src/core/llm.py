"""Utility helpers to instantiate the shared LLM client (OpenRouter-only)."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def _get_openrouter_model() -> str:
    model = (os.getenv("LLM_MODEL") or "").strip()
    if not model:
        raise RuntimeError(
            "LLM_MODEL must be provided (e.g. 'google/gemini-pro' or 'qwen/qwen2-72b-instruct')."
        )
    if model.lower().startswith("openrouter:"):
        return model.split(":", 1)[1].strip()
    return model


def _build_openrouter_client(model_name: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY must be set to call OpenRouter models.")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    headers = {}
    referer = os.getenv("OPENROUTER_REFERER")
    title = os.getenv("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    default_headers = headers or None
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        model=model_name,
        temperature=temperature,
    )


@lru_cache(maxsize=4)
def get_llm_client(*, temperature: float = 0.4):
    """Return the shared OpenRouter-backed chat client."""
    model_name = _get_openrouter_model()
    return _build_openrouter_client(model_name, temperature)


def history_to_text(history) -> str:
    """Utility to turn conversation history into plain text for prompting."""
    lines = []
    for entry in history:
        role = getattr(entry, "type", "") or entry.get("type") if isinstance(entry, dict) else ""
        content = getattr(entry, "content", "") if not isinstance(entry, dict) else entry.get("content", "")
        if not content:
            continue
        if role == "human":
            prefix = "用户"
        elif role == "ai":
            prefix = "SEE"
        else:
            prefix = role or "系统"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines[-12:])

