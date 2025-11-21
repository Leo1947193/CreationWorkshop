import uuid
from typing import Any, Mapping, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.core.schemas import DefectReport, GlobalState
from src.core.state_manager import load_state, save_state
from src.modules.main_graph import execute_main_graph
from src.modules.see_agent_v2 import SEE_GRAPH_V2


class SessionInitResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Existing session identifier")
    message: str = Field(..., description="Latest user utterance")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    conversation_length: int


class AnalyzeRequest(BaseModel):
    session_id: str


class AnalyzeResponse(BaseModel):
    session_id: str
    story: Optional[str]
    top_defect: Optional[DefectReport]


app = FastAPI(
    title="Creative Workshop V1 Experiment API",
    description="本地实验原型，用于测试 4 模块反馈循环。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}


@app.post("/api/v1/session/init", response_model=SessionInitResponse)
async def session_init() -> SessionInitResponse:
    session_id = str(uuid.uuid4())
    state = load_state(session_id)
    save_state(state)
    return SessionInitResponse(session_id=session_id)


def _coerce_global_state(value: Any) -> GlobalState:
    if isinstance(value, GlobalState):
        return value
    if isinstance(value, Mapping):
        return GlobalState(**dict(value))
    raise HTTPException(status_code=500, detail=f"Unexpected state type: {type(value)!r}")


def _ensure_session_state(session_id: str) -> GlobalState:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    return load_state(session_id)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    state = _ensure_session_state(request.session_id)
    state.current_user_input = request.message
    state.socratic_question_needed = True
    state.next_module_to_call = "SEE"

    updated_state = execute_main_graph(state)
    save_state(updated_state)

    ai_messages = []
    for msg in updated_state.conversation_history:
        if getattr(msg, "type", None) == "ai":
            ai_messages.append(getattr(msg, "content", ""))
        elif isinstance(msg, dict) and msg.get("type") == "ai":
            ai_messages.append(msg.get("content", ""))
    response_text = ai_messages[-1] if ai_messages else "（尚未生成 AI 回复）"

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        conversation_length=len(updated_state.conversation_history),
    )


@app.post("/api/v1/chat_simple", response_model=ChatResponse)
async def chat_simple_endpoint(request: ChatRequest) -> ChatResponse:
    """Alternative SEE variant: LLM prompt only sees history + current input."""
    state = _ensure_session_state(request.session_id)
    state.current_user_input = request.message
    state.socratic_question_needed = True

    updated_state_raw = SEE_GRAPH_V2.invoke(state)
    updated_state = _coerce_global_state(updated_state_raw)
    save_state(updated_state)

    ai_messages = []
    for msg in updated_state.conversation_history:
        if getattr(msg, "type", None) == "ai":
            ai_messages.append(getattr(msg, "content", ""))
        elif isinstance(msg, dict) and msg.get("type") == "ai":
            ai_messages.append(msg.get("content", ""))
    response_text = ai_messages[-1] if ai_messages else "（尚未生成 AI 回复）"

    return ChatResponse(
        session_id=request.session_id,
        response=response_text,
        conversation_length=len(updated_state.conversation_history),
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:
    state = _ensure_session_state(request.session_id)
    state.current_user_input = None
    state.next_module_to_call = "CDA"

    updated_state = execute_main_graph(state)
    save_state(updated_state)

    return AnalyzeResponse(
        session_id=request.session_id,
        story=updated_state.generated_story,
        top_defect=updated_state.top_defect,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
