"""Alternative SEE (Socratic Elaboration Engine) implementation.

This variant intentionally only exposes conversation history and the latest
user input to the LLM prompt, without surfacing WM-KG gap/conflict analysis.
It still ingests extracted axioms/facts into WM-KG so that CDA/CDNG work
unchanged, but the Socratic question itself is purely history-driven.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import END
from langgraph.graph import StateGraph

from src.core.llm import get_llm_client, history_to_text
from src.core.logging_utils import get_logger, invoke_with_logging, log_event
from src.core.schemas import GlobalState
from .wm_kg import WorldModelKnowledgeGraph, analyze_input_for_gaps_and_conflicts, ingest_to_wm_kg

logger = get_logger("workshop.see_v2")


SIMPLE_PROMPT = ChatPromptTemplate.from_template(
    """<role>
你是一个富有洞察力的“苏格拉底式”创意导师。你的目标不是回答问题，而是通过提问来帮助用户完善他们复杂的世界观。
</role>

<context>
历史对话:
{conversation_history}

最新输入:
"{current_user_input}"
</context>

<task>
仅基于以上对话与最新输入，提出一个自然的、简洁的引导性问题，帮助用户进一步澄清或扩展这个世界的设定。
不要给出解释或建议，只输出一个问题句子。
</task>"""
)


def add_user_message_to_state_v2(state: GlobalState) -> GlobalState:
    if state.current_user_input:
        state.conversation_history.append(HumanMessage(content=state.current_user_input))
        log_event(
            logger,
            state.session_id,
            "see_v2_add_user_message",
            content_len=len(state.current_user_input),
            history_len=len(state.conversation_history),
        )
    return state


def call_kg_parser_v2(state: GlobalState) -> GlobalState:
    """Reuse WM-KG analysis purely for ingestion; do not expose to LLM."""
    if not state.current_user_input:
        state.analysis_insights = None
        return state

    wmkg = WorldModelKnowledgeGraph(state.internal_kb_path)
    analysis = analyze_input_for_gaps_and_conflicts(
        state.current_user_input,
        wmkg.collection,
        session_id=state.session_id,
    )
    ingestion_count = ingest_to_wm_kg(
        graph=wmkg,
        analysis_results=analysis,
        version=state.current_world_version + 1,
        source_id=f"user_input_turn_{len(state.conversation_history)+1}",
        session_id=state.session_id,
    )
    if ingestion_count:
        state.current_world_version += 1
    # 保留分析结果以便调试，但不会出现在 LLM 提示词中
    state.analysis_insights = analysis
    log_event(
        logger,
        state.session_id,
        "see_v2_wmkg_analysis",
        gaps=len(analysis.get("gaps", [])),
        conflicts=len(analysis.get("conflicts", [])),
        ingested=ingestion_count,
        world_version=state.current_world_version,
    )
    return state


def _fallback_question(current_input: str | None) -> str:
    base = "这个设定很有意思。你能再多讲讲这个世界中最关键的规则、限制或冲突吗？"
    if not current_input:
        return base
    return f"你提到“{current_input}”，在这个世界里，还有哪些关键规则或机制是支撑这一点的？"


def call_socratic_llm_v2(state: GlobalState) -> GlobalState:
    llm = get_llm_client(temperature=0.65)
    history_text = history_to_text(state.conversation_history)
    formatted = SIMPLE_PROMPT.format(
        conversation_history=history_text or "（暂无历史）",
        current_user_input=state.current_user_input or "",
    )

    response = ""
    try:
        llm_output = invoke_with_logging(
            llm,
            formatted,
            session_id=state.session_id,
            label="SEE_v2_Socratic",
        )
        response = getattr(llm_output, "content", "") or str(llm_output)
    except Exception:
        response = ""

    if not response or not str(response).strip():
        response = _fallback_question(state.current_user_input)

    state.conversation_history.append(AIMessage(content=response))
    log_event(
        logger,
        state.session_id,
        "see_v2_ai_response",
        response_len=len(response),
    )
    return state


def add_ai_response_to_state_v2(state: GlobalState) -> GlobalState:
    state.socratic_question_needed = False
    state.current_user_input = None
    return state


def build_see_graph_v2() -> StateGraph:
    """SEE graph variant that only exposes history + current input to the LLM."""
    workflow = StateGraph(GlobalState)
    workflow.add_node("add_user_message_to_state_v2", add_user_message_to_state_v2)
    workflow.add_node("call_kg_parser_v2", call_kg_parser_v2)
    workflow.add_node("call_socratic_llm_v2", call_socratic_llm_v2)
    workflow.add_node("add_ai_response_to_state_v2", add_ai_response_to_state_v2)

    workflow.set_entry_point("add_user_message_to_state_v2")
    workflow.add_edge("add_user_message_to_state_v2", "call_kg_parser_v2")
    workflow.add_edge("call_kg_parser_v2", "call_socratic_llm_v2")
    workflow.add_edge("call_socratic_llm_v2", "add_ai_response_to_state_v2")
    workflow.add_edge("add_ai_response_to_state_v2", END)

    return workflow.compile()


SEE_GRAPH_V2 = build_see_graph_v2()
