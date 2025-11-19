"""Disk-backed persistence helpers for GlobalState.

Implements the workflow outlined in design_doc.md §6.2.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .schemas import GlobalState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_BASE_PATH = PROJECT_ROOT / "db_storage"
STATE_FILENAME = "state.json"


def _ensure_base_dirs(session_id: str) -> Path:
    STATE_BASE_PATH.mkdir(parents=True, exist_ok=True)
    session_dir = STATE_BASE_PATH / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "internal_kb").mkdir(parents=True, exist_ok=True)
    return session_dir


def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
            serialized.append({"type": msg.type, "content": msg.content})
        elif isinstance(msg, dict):
            serialized.append(msg)
        else:
            serialized.append({"type": "unknown", "content": str(msg)})
    return serialized


def _deserialize_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    restored: List[Any] = []
    for msg in messages:
        msg_type = msg.get("type")
        content = msg.get("content", "")
        if msg_type == "human":
            restored.append(HumanMessage(content=content))
        elif msg_type == "ai":
            restored.append(AIMessage(content=content))
        elif msg_type == "system":
            restored.append(SystemMessage(content=content))
        else:
            restored.append(msg)
    return restored


def _default_state(session_id: str) -> GlobalState:
    session_dir = _ensure_base_dirs(session_id)
    return GlobalState(
        session_id=session_id,
        conversation_history=[],
        current_user_input=None,
        socratic_question_needed=True,
        internal_kb_path=str(session_dir / "internal_kb"),
        current_world_version=0,
        world_spec_snapshot=None,
        defect_reports=[],
        top_defect=None,
        generated_story=None,
        next_module_to_call="IDLE",
        last_error=None,
        analysis_insights=None,
    )


def load_state(session_id: str) -> GlobalState:
    """Load state.json for a session or create a new default state."""
    session_dir = _ensure_base_dirs(session_id)
    state_path = session_dir / STATE_FILENAME
    if not state_path.exists():
        return _default_state(session_id)

    with state_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "conversation_history" in payload:
        payload["conversation_history"] = _deserialize_messages(payload["conversation_history"])
    if "internal_kb_path" not in payload:
        payload["internal_kb_path"] = str(session_dir / "internal_kb")

    return GlobalState(**payload)


def save_state(state: GlobalState) -> None:
    """Persist GlobalState to disk as JSON."""
    session_dir = _ensure_base_dirs(state.session_id)
    state_path = session_dir / STATE_FILENAME

    state_dict = state.model_dump()
    state_dict["conversation_history"] = _serialize_messages(state.conversation_history)

    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state_dict, f, ensure_ascii=False, indent=2)

