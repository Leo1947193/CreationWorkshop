"""SEE (Socratic Elaboration Engine) LangGraph implementation."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import END
from langgraph.graph import StateGraph

from src.core.llm import get_llm_client, history_to_text
from src.core.logging_utils import get_logger, invoke_with_logging, log_event
from src.core.schemas import GlobalState
from .wm_kg import WorldModelKnowledgeGraph, analyze_input_for_gaps_and_conflicts, ingest_to_wm_kg

logger = get_logger("workshop.see")

GAP_PROMPT = ChatPromptTemplate.from_template(
    """<role>
你是一个富有洞察力的“苏格拉底式”创意导师。你的目标不是回答问题，而是通过提问来帮助用户完善他们复杂的世界观。
</role>

<context>
历史对话:
{conversation_history}

最新输入:
"{current_user_input}"

本体论空白:
<gaps>
{gap_list}
</gaps>
</context>

<task>
生成一个自然的引导性问题，帮助用户填补这些空白。
</task>"""
)

CONFLICT_PROMPT = ChatPromptTemplate.from_template(
    """<role>
你是一个富有洞察力的“苏格拉底式”创意导师。你的目标是帮助用户建立一个逻辑一致的世界。
</role>

<context>
历史对话:
{conversation_history}

最新输入:
"{current_user_input}"

潜在冲突:
<conflicts>
{conflict_list}
</conflicts>
</context>

<task>
生成一个调和性的提问，温和地指出这些规则如何共存。
</task>"""
)


def add_user_message_to_state(state: GlobalState) -> GlobalState:
    if state.current_user_input:
        state.conversation_history.append(HumanMessage(content=state.current_user_input))
        log_event(
            logger,
            state.session_id,
            "see_add_user_message",
            content_len=len(state.current_user_input),
            history_len=len(state.conversation_history),
        )
    return state


def call_kg_parser(state: GlobalState) -> GlobalState:
    if not state.current_user_input:
        state.analysis_insights = {"gaps": [], "conflicts": [], "extracted_data": {}}
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
    state.analysis_insights = analysis
    log_event(
        logger,
        state.session_id,
        "see_wmkg_analysis",
        gaps=len(analysis.get("gaps", [])),
        conflicts=len(analysis.get("conflicts", [])),
        ingested=ingestion_count,
        world_version=state.current_world_version,
    )
    return state


def _render_gap_prompt(gaps: list[str], history_count: int) -> str:
    if not gaps:
        return "你的设定越来越有意思了。有没有哪些关键机制或人物关系是你还没详细描述的？"
    joined = "; ".join(gaps[:3])
    return (
        f"我在你的描述中还找不到这些细节：{joined}。"
        "你愿意分享一下这些部分是如何运作的吗？"
    )


def _render_conflict_prompt(conflicts: list[str]) -> str:
    if not conflicts:
        return ""
    conflict = conflicts[0]
    return (
        "你的最新想法和我们之前记录的一些规则之间似乎存在互动。"
        f"比如：{conflict}。你觉得它们会如何共存？"
    )


def call_socratic_llm(state: GlobalState) -> GlobalState:
    insights = state.analysis_insights or {}
    gaps = insights.get("gaps", [])
    conflicts = insights.get("conflicts", [])

    response = ""
    llm = get_llm_client(temperature=0.65)
    history_text = history_to_text(state.conversation_history)
    if llm:
        prompt = CONFLICT_PROMPT if conflicts else GAP_PROMPT
        formatted = prompt.format(
            conversation_history=history_text or "（暂无历史）",
            current_user_input=state.current_user_input or "",
            gap_list="\n".join(gaps) or "（暂无空白）",
            conflict_list="\n".join(conflicts) or "（暂无冲突）",
        )
        try:
            llm_output = invoke_with_logging(
                llm,
                formatted,
                session_id=state.session_id,
                label="SEE_Socratic",
            )
            response = getattr(llm_output, "content", "") or str(llm_output)
        except Exception:
            response = ""
    if not response:
        response = _render_conflict_prompt(conflicts) if conflicts else _render_gap_prompt(gaps, len(state.conversation_history))

    state.conversation_history.append(AIMessage(content=response))
    log_event(
        logger,
        state.session_id,
        "see_ai_response",
        response_len=len(response),
        conflicts=len(conflicts),
        gaps=len(gaps),
    )
    return state


def add_ai_response_to_state(state: GlobalState) -> GlobalState:
    state.socratic_question_needed = False
    state.current_user_input = None
    return state


def build_see_graph() -> StateGraph:
    workflow = StateGraph(GlobalState)
    workflow.add_node("add_user_message_to_state", add_user_message_to_state)
    workflow.add_node("call_kg_parser", call_kg_parser)
    workflow.add_node("call_socratic_llm", call_socratic_llm)
    workflow.add_node("add_ai_response_to_state", add_ai_response_to_state)

    workflow.set_entry_point("add_user_message_to_state")
    workflow.add_edge("add_user_message_to_state", "call_kg_parser")
    workflow.add_edge("call_kg_parser", "call_socratic_llm")
    workflow.add_edge("call_socratic_llm", "add_ai_response_to_state")
    workflow.add_edge("add_ai_response_to_state", END)

    return workflow.compile()
