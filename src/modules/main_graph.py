"""Main LangGraph orchestrator for the 4-module feedback loop."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.constants import END
from langgraph.graph import StateGraph

from src.core.schemas import GlobalState
from .cda_agent import run_cda_module
from .cdng_chain import run_cdng_module
from .see_agent import build_see_graph


SEE_GRAPH = build_see_graph()


def _coerce_global_state(value: Any) -> GlobalState:
    """Convert LangGraph outputs (e.g., AddableValuesDict) back to GlobalState."""
    if isinstance(value, GlobalState):
        return value
    if isinstance(value, Mapping):
        return GlobalState(**dict(value))
    raise TypeError(f"Unsupported state type: {type(value)!r}")


def run_see_module(state: GlobalState) -> GlobalState:
    updated_state = SEE_GRAPH.invoke(state)
    updated_state = _coerce_global_state(updated_state)
    updated_state.next_module_to_call = "IDLE"
    return updated_state


def run_cda_module_node(state: GlobalState) -> GlobalState:
    return run_cda_module(state)


def run_cdng_module_node(state: GlobalState) -> GlobalState:
    return run_cdng_module(state)


def router_decision(state: GlobalState) -> str:
    return state.next_module_to_call


def build_main_graph() -> StateGraph:
    workflow = StateGraph(GlobalState)
    workflow.add_node("router", lambda s: s)
    workflow.add_node("run_see_module", run_see_module)
    workflow.add_node("run_cda_module", run_cda_module_node)
    workflow.add_node("run_cdng_module", run_cdng_module_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        router_decision,
        {
            "SEE": "run_see_module",
            "CDA": "run_cda_module",
            "CDNG": "run_cdng_module",
            "IDLE": END,
        },
    )
    workflow.add_edge("run_see_module", END)
    workflow.add_edge("run_cda_module", "router")
    workflow.add_edge("run_cdng_module", END)
    return workflow.compile()


MAIN_GRAPH = build_main_graph()


def execute_main_graph(state: GlobalState) -> GlobalState:
    result = MAIN_GRAPH.invoke(state)
    return _coerce_global_state(result)
