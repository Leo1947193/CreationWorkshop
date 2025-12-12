"""LLM-driven CDA implementation with a true RATT workflow."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict

import torch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.core.llm import get_llm_client
from src.core.logging_utils import get_logger, invoke_with_logging, log_event
from src.core.schemas import DefectReport, GlobalState, WorldSpecificationSnapshot
from .dual_rag import DualRARetriever, ValidatedWebRetriever
from .wm_kg import EMBEDDING_MODEL_NAME, WorldModelKnowledgeGraph

logger = get_logger("workshop.cda")


class InternalAxiomRetriever(BaseRetriever):
    """Retriever that uses dense search over axioms."""

    def __init__(self, wmkg: WorldModelKnowledgeGraph, top_k: int = 5):
        super().__init__()
        self._wmkg = wmkg
        self._top_k = top_k
        self._vector_store = None
        self._embeddings = None

    def _ensure_embeddings(self):
        if self._embeddings is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device},
            )
        return self._embeddings

    def _ensure_vector_store(self):
        if self._vector_store is None:
            embeddings = self._ensure_embeddings()
            self._vector_store = Chroma(
                collection_name="internal_kb",
                persist_directory=str(self._wmkg.persist_directory),
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._vector_store

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        sid = getattr(run_manager, "session_id", None) or "-"
        try:
            store = self._ensure_vector_store()
            retriever = store.as_retriever(search_kwargs={"k": self._top_k, "filter": {"type": "AXIOM"}})
            candidates = retriever.invoke(query)
            log_event(
                logger,
                sid,
                "internal_axiom_dense",
                query_len=len(query),
                candidates=len(candidates),
                k=self._top_k,
                similarity="cosine",
            )
        except Exception:
            candidates = []
            log_event(logger, sid, "internal_axiom_dense_failed", query_len=len(query))

        if not candidates:
            # Fallback: return all axioms (truncated) if dense search fails.
            docs = self._wmkg.list_axioms()
            return docs[: self._top_k]

        # Return dense results directly (cosine similarity) capped by top_k
        return candidates[: self._top_k]


class DefectCandidateModel(BaseModel):
    id: str
    title: str
    description: str
    supporting_axioms: List[str] = Field(default_factory=list)
    supporting_external: List[str] = Field(default_factory=list)
    long_term_consequence: str


class SeedThoughtModel(BaseModel):
    id: str
    focus_question: str
    path_summary: str


class SeedThoughtResponse(BaseModel):
    thoughts: List[SeedThoughtModel]


class GeneratedChildThought(BaseModel):
    id: str
    focus_question: str
    path_summary: str
    status: Literal["EXPAND", "DEFECT", "SAFE"]
    defect_candidates: List[DefectCandidateModel] = Field(default_factory=list)


class GeneratedThoughtResponse(BaseModel):
    thoughts: List[GeneratedChildThought]


class JudgedThoughtModel(BaseModel):
    status: Literal["EXPAND", "DEFECT", "SAFE"]
    refined_path_summary: str
    score: float = 0.0
    defect_candidates: List[DefectCandidateModel] = Field(default_factory=list)


class FinalDefectModel(BaseModel):
    id: str
    title: str
    description: str
    long_term_consequence: str
    supporting_axioms: List[str]
    supporting_external: List[str]
    likelihood: int = Field(ge=1, le=5)
    severity: int = Field(ge=1, le=5)


class FinalDefectResponse(BaseModel):
    defects: List[FinalDefectModel]


@dataclass
class DefectCandidate:
    id: str
    title: str
    description: str
    supporting_axioms: List[str]
    supporting_external: List[str]
    long_term_consequence: str


@dataclass
class ThoughtNode:
    id: str
    parent_id: Optional[str]
    depth: int
    path_summary: str
    focus_question: str
    status: Literal["PENDING", "EXPAND", "DEFECT", "SAFE"] = "EXPAND"
    score: float = 0.0
    defect_candidates: List[DefectCandidate] = field(default_factory=list)
    context_internal: List[str] = field(default_factory=list)
    context_external: List[str] = field(default_factory=list)


class RATTGraphState(TypedDict, total=False):
    session_id: str
    world_spec: WorldSpecificationSnapshot
    dual_retriever: DualRARetriever
    axioms: List[str]
    active_nodes: List[ThoughtNode]
    all_nodes: Dict[str, ThoughtNode]
    final_defects: List[DefectCandidate]
    iteration: int
    max_depth: int
    max_nodes: int


def _format_axioms(axioms: List[str], limit: int = 12) -> str:
    if not axioms:
        return "（暂无公理）"
    clipped = axioms[:limit]
    body = "\n".join(f"- {ax}" for ax in clipped)
    if len(axioms) > limit:
        body += "\n- ...(其余公理略)"
    return body


def _convert_candidate(model: DefectCandidateModel) -> DefectCandidate:
    return DefectCandidate(
        id=model.id or str(uuid.uuid4()),
        title=model.title.strip(),
        description=model.description.strip(),
        supporting_axioms=[s.strip() for s in model.supporting_axioms if s.strip()],
        supporting_external=[s.strip() for s in model.supporting_external if s.strip()],
        long_term_consequence=model.long_term_consequence.strip(),
    )


def seed_problem_space(state: RATTGraphState) -> RATTGraphState:
    sid = state.get("session_id", "-")
    llm = get_llm_client(temperature=0.15).with_structured_output(SeedThoughtResponse)
    axioms = state.get("axioms", [])
    prompt = (
        "<role>你是一名系统缺陷分析师，负责理解一个复杂世界的关键风险维度。</role>\n"
        "<context>\n"
        "以下是当前世界的公理（核心规则）：\n"
        f"{_format_axioms(axioms, limit=30)}\n"
        "</context>\n"
        "<task>\n"
        "请输出 3-5 个最值得深入探索的系统性风险方向。每个方向需要包含：\n"
        "1. id（例如 R1, R2...）\n"
        "2. focus_question：需要追问的问题\n"
        "3. path_summary：迄今为止的推理假设\n"
        "</task>"
    )
    log_event(logger, sid, "cda_seed_start", axiom_count=len(axioms))
    try:
        seed_response = invoke_with_logging(llm, prompt, session_id=sid, label="CDA_seed")
        thoughts = seed_response.thoughts
    except Exception:
        # Fallback：如果 LLM 解析失败，创建一个默认节点
        thoughts = [
            SeedThoughtModel(
                id="R1",
                focus_question="在现有公理下，哪类系统性风险最可能发生？",
                path_summary="需要进一步梳理风险维度。",
            )
        ]

    active_nodes: List[ThoughtNode] = []
    all_nodes: Dict[str, ThoughtNode] = state.get("all_nodes", {})
    # 限制初始种子数量，以控制 RATT 宽度
    max_seeds = 3
    for seed in thoughts[:max_seeds]:
        node_id = seed.id or f"R{len(all_nodes)+1}"
        node = ThoughtNode(
            id=node_id,
            parent_id=None,
            depth=0,
            path_summary=seed.path_summary.strip(),
            focus_question=seed.focus_question.strip(),
            status="EXPAND",
        )
        active_nodes.append(node)
        all_nodes[node.id] = node

    state["active_nodes"] = active_nodes
    state["all_nodes"] = all_nodes
    state["iteration"] = 0
    log_event(logger, sid, "cda_seed_result", seeds=len(active_nodes))
    return state


def generate_thoughts(state: RATTGraphState) -> RATTGraphState:
    # 每一轮只扩展少量得分最高或顺序靠前的节点，避免节点爆炸
    expandables = [node for node in state.get("active_nodes", []) if node.status == "EXPAND"]
    max_parents_per_iter = 3
    expandables = expandables[:max_parents_per_iter]
    if not expandables:
        state["active_nodes"] = []
        return state

    sid = state.get("session_id", "-")
    llm = get_llm_client(temperature=0.3).with_structured_output(GeneratedThoughtResponse)
    axioms_text = _format_axioms(state.get("axioms", []), limit=20)
    new_nodes: List[ThoughtNode] = []
    all_nodes = state.get("all_nodes", {})
    log_event(logger, sid, "cda_generate_start", parents=len(expandables))

    for node in expandables:
        prompt = (
            "<role>你是一名系统思维专家，正在扩展一棵“后果推理树”。</role>\n"
            "<context>\n"
            f"当前节点深度: {node.depth}\n"
            f"累积推理: {node.path_summary}\n"
            f"聚焦问题: {node.focus_question}\n"
            "世界公理如下：\n"
            f"{axioms_text}\n"
            "</context>\n"
            "<task>\n"
            "为该节点生成 1-3 个最重要的后续探索方向。请自行判断哪些应该继续展开，哪些已经代表明确缺陷或安全区。\n"
            "每个方向必须包含：id、focus_question、path_summary、status(EXPAND/DEFECT/SAFE)，\n"
            "必要时可附带 defect_candidates。\n"
            "</task>"
        )
        try:
            resp = invoke_with_logging(llm, prompt, session_id=sid, label="CDA_generate")
            children = resp.thoughts
        except Exception:
            children = []

        # 限制每个父节点生成的子节点数量
        max_children_per_parent = 2
        for child in children[:max_children_per_parent]:
            child_id = child.id or str(uuid.uuid4())
            if child_id in all_nodes:
                child_id = f"{child_id}_{uuid.uuid4().hex[:4]}"
            new_node = ThoughtNode(
                id=child_id,
                parent_id=node.id,
                depth=node.depth + 1,
                path_summary=child.path_summary.strip(),
                focus_question=child.focus_question.strip(),
                status=child.status,
            )
            for cand in child.defect_candidates:
                new_node.defect_candidates.append(_convert_candidate(cand))
            all_nodes[new_node.id] = new_node
            new_nodes.append(new_node)

    state["active_nodes"] = new_nodes
    state["all_nodes"] = all_nodes
    log_event(logger, sid, "cda_generate_result", children=len(new_nodes), total_nodes=len(all_nodes))
    return state


def retrieve_context(state: RATTGraphState) -> RATTGraphState:
    retriever = state["dual_retriever"]
    sid = state.get("session_id", "-")
    for node in state.get("active_nodes", []):
        query = node.focus_question or node.path_summary
        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = []
        node.context_internal = []
        node.context_external = []
        for doc in docs:
            text = doc.page_content.strip()
            if not text:
                continue
            if doc.metadata.get("source") == "KB_INTERNAL":
                node.context_internal.append(text)
            else:
                node.context_external.append(text)
        log_event(
            logger,
            sid,
            "cda_retrieve_node",
            node_id=node.id,
            internal=len(node.context_internal),
            external=len(node.context_external),
        )
    return state


def judge_thought(state: RATTGraphState) -> RATTGraphState:
    llm = get_llm_client(temperature=0.15).with_structured_output(JudgedThoughtModel)
    axioms = state.get("axioms", [])
    sid = state.get("session_id", "-")
    # 每一轮最多裁判若干个节点，避免调用次数失控
    max_judged_per_iter = 6
    for node in state.get("active_nodes", [])[:max_judged_per_iter]:
        internal_context = "\n".join(node.context_internal[:5]) or "（暂无）"
        external_context = "\n".join(node.context_external[:5]) or "（暂无）"
        prompt = (
            "<role>你现在是系统缺陷裁判。请基于提供的上下文评估该推理节点。</role>\n"
            "<context>\n"
            f"累积推理: {node.path_summary}\n"
            f"聚焦问题: {node.focus_question}\n"
            "相关公理:\n"
            f"{_format_axioms(axioms, limit=15)}\n"
            "内部检索到的公理片段:\n"
            f"{internal_context}\n"
            "外部检索到的现实常识/原理:\n"
            f"{external_context}\n"
            "</context>\n"
            "<task>\n"
            "1. 决定该节点的状态（EXPAND/DEFECT/SAFE）。\n"
            "2. 更新推理摘要 refined_path_summary。\n"
            "3. 给出 score（0-1 之间的小数，用于决定是否继续探索）。\n"
            "4. 如果发现缺陷，补充 defect_candidates（可为 0 个）。\n"
            "输出需符合结构化格式。\n"
            "</task>"
        )
        try:
            judged = invoke_with_logging(llm, prompt, session_id=sid, label="CDA_judge")
        except Exception:
            judged = JudgedThoughtModel(
                status="SAFE",
                refined_path_summary=node.path_summary,
                score=0.0,
                defect_candidates=[],
            )
        node.status = judged.status
        node.path_summary = judged.refined_path_summary
        node.score = judged.score
        for cand in judged.defect_candidates:
            node.defect_candidates.append(_convert_candidate(cand))
        log_event(
            logger,
            sid,
            "cda_judge_node",
            node_id=node.id,
            status=node.status,
            score=node.score,
            defects=len(node.defect_candidates),
        )
    return state


def evaluate_and_prune(state: RATTGraphState) -> RATTGraphState:
    final_defects = state.get("final_defects", [])
    next_nodes: List[ThoughtNode] = []
    max_depth = state.get("max_depth", 3)
    max_nodes = state.get("max_nodes", 40)
    sid = state.get("session_id", "-")

    for node in state.get("active_nodes", []):
        final_defects.extend(node.defect_candidates)
        if node.status == "EXPAND" and node.depth < max_depth:
            next_nodes.append(node)

    next_nodes.sort(key=lambda n: n.score, reverse=True)
    # Enforce total node budget
    all_nodes = state.get("all_nodes", {})
    remaining_budget = max(0, max_nodes - len(all_nodes))
    if remaining_budget and len(next_nodes) > remaining_budget:
        next_nodes = next_nodes[:remaining_budget]

    state["final_defects"] = final_defects
    state["active_nodes"] = next_nodes
    state["iteration"] = state.get("iteration", 0) + 1
    log_event(
        logger,
        sid,
        "cda_prune",
        kept=len(next_nodes),
        defects=len(final_defects),
        iteration=state["iteration"],
    )
    return state


def generate_final_report(state: RATTGraphState) -> RATTGraphState:
    defects = state.get("final_defects", [])
    if not defects:
        return state

    sid = state.get("session_id", "-")
    llm = get_llm_client(temperature=0.2).with_structured_output(FinalDefectResponse)
    serialized = []
    for defect in defects[:20]:
        serialized.append(
            {
                "id": defect.id,
                "title": defect.title,
                "description": defect.description,
                "long_term_consequence": defect.long_term_consequence,
                "supporting_axioms": defect.supporting_axioms,
                "supporting_external": defect.supporting_external,
            }
        )
    prompt = (
        "<role>你是最终的风险评审官。</role>\n"
        "<task>\n"
        "以下是候选缺陷列表（JSON 格式）。请合并相似缺陷，并为每条缺陷打出 likelihood/severity (1-5)。\n"
        "输出结构必须包含 defects 数组。\n"
        "</task>\n"
        f"<candidates>{json.dumps(serialized, ensure_ascii=False)}</candidates>"
    )
    try:
        final_response = invoke_with_logging(llm, prompt, session_id=sid, label="CDA_final_report")
        merged = final_response.defects
    except Exception:
        merged = []

    if merged:
        state["final_defects"] = [
            DefectCandidate(
                id=item.id,
                title=item.title,
                description=item.description,
                supporting_axioms=item.supporting_axioms,
                supporting_external=item.supporting_external,
                long_term_consequence=item.long_term_consequence,
            )
            for item in merged
        ]
        state["final_scores"] = {item.id: (item.likelihood, item.severity) for item in merged}
    log_event(logger, sid, "cda_final_report", merged=len(state.get("final_defects", [])))
    return state


def _ratt_flow_decision(state: RATTGraphState) -> str:
    if state.get("iteration", 0) >= state.get("max_depth", 3):
        return "generate_final_report"
    if not state.get("active_nodes"):
        return "generate_final_report"
    if len(state.get("all_nodes", {})) >= state.get("max_nodes", 40):
        return "generate_final_report"
    return "generate_thoughts"


def build_cda_graph() -> StateGraph:
    workflow = StateGraph(RATTGraphState)
    workflow.add_node("seed_problem_space", seed_problem_space)
    workflow.add_node("generate_thoughts", generate_thoughts)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("judge_thought", judge_thought)
    workflow.add_node("evaluate_and_prune", evaluate_and_prune)
    workflow.add_node("generate_final_report", generate_final_report)

    workflow.set_entry_point("seed_problem_space")
    workflow.add_edge("seed_problem_space", "generate_thoughts")
    workflow.add_edge("generate_thoughts", "retrieve_context")
    workflow.add_edge("retrieve_context", "judge_thought")
    workflow.add_edge("judge_thought", "evaluate_and_prune")
    workflow.add_conditional_edges(
        "evaluate_and_prune",
        _ratt_flow_decision,
        {
            "generate_thoughts": "generate_thoughts",
            "generate_final_report": "generate_final_report",
        },
    )
    workflow.add_edge("generate_final_report", END)
    return workflow.compile()


def _build_world_snapshot(state: GlobalState, wmkg: WorldModelKnowledgeGraph) -> WorldSpecificationSnapshot:
    docs = wmkg.list_axioms()
    return WorldSpecificationSnapshot(
        snapshot_path=state.internal_kb_path,
        version=state.current_world_version,
        entity_count=max(1, len(docs)),
        axiom_count=len(docs),
    )


def run_cda_module(state: GlobalState) -> GlobalState:
    wmkg = WorldModelKnowledgeGraph(state.internal_kb_path)
    state.world_spec_snapshot = state.world_spec_snapshot or _build_world_snapshot(state, wmkg)
    axiom_docs = wmkg.list_axioms()
    axioms = [doc.page_content for doc in axiom_docs]
    log_event(
        logger,
        state.session_id,
        "cda_start",
        axiom_count=len(axioms),
        world_version=state.current_world_version,
    )

    if not axioms:
        state.defect_reports = []
        state.top_defect = None
        state.generated_story = "目前的世界规则还不够具体，无法开展后果分析。请继续在 SEE 模块完善你的设定。"
        state.next_module_to_call = "IDLE"
        return state

    dual_retriever = DualRARetriever(InternalAxiomRetriever(wmkg), ValidatedWebRetriever())

    ratt_state: RATTGraphState = {
        "session_id": state.session_id,
        "world_spec": state.world_spec_snapshot,
        "dual_retriever": dual_retriever,
        "axioms": axioms,
        "active_nodes": [],
        "all_nodes": {},
        "final_defects": [],
        "iteration": 0,
        "max_depth": 2,
        "max_nodes": 12,
    }

    ratt_graph = build_cda_graph()
    ratt_result = ratt_graph.invoke(ratt_state)

    final_defects = ratt_result.get("final_defects", [])
    final_scores = ratt_result.get("final_scores", {})  # optional dict of id -> (likelihood, severity)

    defect_reports: List[DefectReport] = []
    for defect in final_defects:
        likelihood, severity = final_scores.get(defect.id, (3, 3))
        defect_reports.append(
            DefectReport(
                defect_id=defect.id,
                description=defect.description or defect.title,
                ratt_branch=defect.title,
                likelihood=likelihood,
                severity=severity,
                risk_score=likelihood * severity,
                long_term_consequence=defect.long_term_consequence,
            )
        )

    if not defect_reports:
        state.defect_reports = []
        state.top_defect = None
        state.generated_story = "当前规则下未发现明确缺陷；如需深入，请继续在 SEE 中添加关键机制/约束。"
        state.next_module_to_call = "IDLE"
        log_event(logger, state.session_id, "cda_no_defect")
        return state

    defect_reports.sort(key=lambda d: d.risk_score, reverse=True)
    state.defect_reports = defect_reports
    state.top_defect = defect_reports[0]
    state.next_module_to_call = "CDNG"
    log_event(
        logger,
        state.session_id,
        "cda_result",
        defects=len(defect_reports),
        top_defect=state.top_defect.defect_id,
        top_risk=state.top_defect.risk_score,
    )
    return state
