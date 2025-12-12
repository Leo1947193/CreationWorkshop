"""Shared logging helpers for backend observability."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_FILE = LOG_DIR / "backend.log"


class _SessionFilter(logging.Filter):
    """Ensure every record carries a session_id field."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure console + file logging once."""
    logger = logging.getLogger("workshop")
    if getattr(configure_logging, "_configured", False):
        logger.setLevel(level)
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [session=%(session_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    session_filter = _SessionFilter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.addFilter(session_filter)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.addFilter(session_filter)
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    configure_logging._configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(name: str = "workshop") -> logging.Logger:
    """Return a child logger."""
    return logging.getLogger(name)


def _clip(text: str, limit: int = 800) -> str:
    """Return a safe preview of text."""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head} ...[+{len(text) - limit} chars]... {tail}"


def log_event(logger: logging.Logger, session_id: str, event: str, **kwargs: Any) -> None:
    """Helper for structured-ish event logs."""
    payload = {"event": event, **kwargs}
    try:
        message = json.dumps(payload, ensure_ascii=False)
    except Exception:
        message = f"{event} | {payload}"
    logger.info(message, extra={"session_id": session_id})


def invoke_with_logging(llm: Any, prompt: Any, *, session_id: str, label: str, preview_chars: int = 800):
    """Invoke an LLM client while logging prompt/response summaries."""
    logger = get_logger("workshop.llm")
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown-model"
    prompt_text = str(prompt)
    logger.info(
        f"[LLM:{label}] model={model_name} prompt_len={len(prompt_text)} preview={_clip(prompt_text, preview_chars)}",
        extra={"session_id": session_id},
    )
    try:
        response = llm.invoke(prompt)
    except Exception:
        logger.exception(f"[LLM:{label}] invoke failed", extra={"session_id": session_id})
        raise

    response_text = getattr(response, "content", None)
    response_text = response_text if isinstance(response_text, str) else str(response)
    logger.info(
        f"[LLM:{label}] response_len={len(response_text)} preview={_clip(response_text, preview_chars)}",
        extra={"session_id": session_id},
    )
    return response

