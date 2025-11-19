"""Pydantic schemas shared across modules.

These mirror the data contracts outlined in design_doc.md §0.4.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class WorldSpecificationSnapshot(BaseModel):
    """Snapshot exported from the world model / knowledge graph (WM-KG)."""

    snapshot_path: str = Field(description="Filesystem path to the persistent ChromaDB snapshot")
    version: int = Field(description="World specification version identifier")
    entity_count: int
    axiom_count: int


class DefectReport(BaseModel):
    """Defect analysis output produced by the CDA module."""

    defect_id: str = Field(description="Unique identifier for the detected defect")
    description: str = Field(description="Concise description of the defect")
    ratt_branch: str = Field(description="RATT tree path that led to this discovery")
    likelihood: int = Field(description="Likelihood score in the 1-5 scale")
    severity: int = Field(description="Severity score in the 1-5 scale")
    risk_score: int = Field(description="Composite risk score (likelihood * severity)")
    long_term_consequence: str = Field(description="Long-term catastrophic outcome of the defect")


class GlobalState(BaseModel):
    """Central mutable state that flows through the main LangGraph."""

    session_id: str = Field(description="Unique session identifier to scope persistence")

    # Module 1 (SEE)
    conversation_history: List[BaseMessage | Dict[str, Any]] = Field(
        default_factory=list, description="Full dialogue history for the SEE module",
    )
    current_user_input: Optional[str] = Field(
        None, description="Latest message from the user captured by FastAPI",
    )
    socratic_question_needed: bool = Field(
        True, description="Flag indicating whether SEE should craft a Socratic prompt",
    )
    analysis_insights: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Scratch pad for SEE/WM-KG derived insights (gaps, conflicts, extracted data)",
    )

    # Module 2 (WM-KG)
    internal_kb_path: str = Field(description="Directory storing this session's ChromaDB artifacts")
    current_world_version: int = Field(
        0, description="Monotonic version number for the internal world specification",
    )

    # Module 3 (CDA)
    world_spec_snapshot: Optional[WorldSpecificationSnapshot] = Field(
        None, description="Snapshot used by CDA for deeper analysis",
    )
    defect_reports: List[DefectReport] = Field(
        default_factory=list, description="List of defects detected in the current CDA run",
    )
    top_defect: Optional[DefectReport] = Field(
        None, description="Highest risk defect surfaced by CDA",
    )

    # Module 4 (CDNG)
    generated_story: Optional[str] = Field(
        None, description="Narrative produced by the CDNG chain",
    )

    # Orchestration metadata
    next_module_to_call: Literal["IDLE", "SEE", "CDA", "CDNG"] = Field(
        "IDLE", description="Router flag instructing the main graph which module to run next",
    )
    last_error: Optional[str] = Field(
        None, description="Diagnostics string for the last error",
    )
