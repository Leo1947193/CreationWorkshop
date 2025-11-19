"""World Model Knowledge Graph utilities (Module 2).

Contains the ontological meta-model, heuristic analyzers, and persistence helpers.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.core.llm import get_llm_client

os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")
if not EMBEDDING_MODEL_NAME:
    raise RuntimeError("EMBEDDING_MODEL must be provided via environment.")

try:
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # type: ignore
except ImportError:  # pragma: no cover
    chromadb = None
    SentenceTransformerEmbeddingFunction = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ONTOLOGICAL_META_MODEL: Dict[str, List[str]] = {
    "Process": ["mechanism", "actor", "criteria"],
    "System": ["inputs", "outputs", "controller"],
    "Entity": ["location", "owner", "purpose"],
    "SocialRule": ["scope", "penalty", "enforcer"],
}


class ExtractedAxiom(BaseModel):
    """Structured representation for axioms extracted from user input."""

    description: str = Field(description="Rule description")
    subjects: List[str] = Field(default_factory=list, description="Entities involved in the rule")
    type: str = Field(description="Category e.g. Physical, Social")


class ExtractedFact(BaseModel):
    """Structured representation for factual statements."""

    description: str
    entity: str
    property: str
    value: str


class ExtractionPayload(BaseModel):
    axioms: List[ExtractedAxiom] = Field(default_factory=list)
    facts: List[ExtractedFact] = Field(default_factory=list)


class _FallbackCollection:
    """Disk-backed JSON store used when chromadb is unavailable."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if self.storage_path.exists():
            with self.storage_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        else:
            payload = {"ids": [], "documents": [], "metadatas": []}
        self.ids: List[str] = payload["ids"]
        self.documents: List[str] = payload["documents"]
        self.metadatas: List[Dict[str, Any]] = payload["metadatas"]

    def _sync(self) -> None:
        with self.storage_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas},
                f,
                ensure_ascii=False,
                indent=2,
            )

    # The methods below emulate the chromadb Collection interface subset used in the project.
    def add(self, *, documents: Sequence[str], metadatas: Sequence[Dict[str, Any]], ids: Sequence[str]) -> None:
        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._sync()

    def get(self, *, where: Optional[Dict[str, Any]] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        include = include or ["metadatas", "documents"]
        if where:
            matched_indexes = [
                idx
                for idx, metadata in enumerate(self.metadatas)
                if all(metadata.get(key) == value for key, value in where.items())
            ]
        else:
            matched_indexes = list(range(len(self.ids)))

        ids = [self.ids[i] for i in matched_indexes]
        docs = [self.documents[i] for i in matched_indexes]
        metas = [self.metadatas[i] for i in matched_indexes]

        payload: Dict[str, Any] = {}
        if "ids" in include:
            payload["ids"] = ids
        if "documents" in include:
            payload["documents"] = docs
        if "metadatas" in include:
            payload["metadatas"] = metas
        return payload

    def query(self, *, query_texts: Sequence[str], n_results: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        documents = []
        metadatas = []
        ids = []
        for idx, (doc, meta) in enumerate(zip(self.documents, self.metadatas)):
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            if any(query.lower() in doc.lower() for query in query_texts):
                documents.append(doc)
                metadatas.append(meta)
                ids.append(self.ids[idx])
        documents = documents[:n_results]
        metadatas = metadatas[:n_results]
        ids = ids[:n_results]
        return {"documents": [documents], "metadatas": [metadatas], "ids": [ids]}


class WorldModelKnowledgeGraph:
    """Convenience wrapper responsible for managing the internal KB store."""

    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None
        self._fallback: Optional[_FallbackCollection] = None
        self._embedding_function = None

    def _build_embedding_function(self):
        if SentenceTransformerEmbeddingFunction is None:
            return None
        if self._embedding_function is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self._embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME,
                device=device,
            )
        return self._embedding_function

    @property
    def collection(self):
        if self._collection is not None:
            return self._collection

        if chromadb is None:
            self._fallback = _FallbackCollection(self.persist_directory / "fallback_collection.json")
            self._collection = self._fallback
            return self._collection

        if self._client is None:
            client_settings = None
            if "CHROMADB_DISABLE_TELEMETRY" in os.environ:
                client_settings = Settings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                    persist_directory=str(self.persist_directory),
                )
            if client_settings:
                self._client = chromadb.PersistentClient(settings=client_settings)
            else:
                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        embedding_fn = self._build_embedding_function()
        if embedding_fn is not None:
            self._collection = self._client.get_or_create_collection(
                "internal_kb",
                embedding_function=embedding_fn,
            )
        else:
            self._collection = self._client.get_or_create_collection("internal_kb")
        return self._collection

    def list_axioms(self) -> List[Document]:
        """Return all AXIOM entries as LangChain Documents for downstream modules."""
        results = self.collection.get(where={"type": "AXIOM"}, include=["documents", "metadatas"])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        docs: List[Document] = []
        for text, metadata in zip(documents, metadatas):
            docs.append(Document(page_content=text, metadata=metadata))
        return docs


def _classify_sentence(sentence: str) -> str:
    lowered = sentence.lower()
    if any(keyword in lowered for keyword in ["process", "workflow", "allocate"]):
        return "Process"
    if any(keyword in lowered for keyword in ["system", "infrastructure", "network"]):
        return "System"
    if any(keyword in lowered for keyword in ["law", "rule", "must", "should", "ban"]):
        return "SocialRule"
    return "Entity"


def _extract_props(sentence: str, entity_type: str) -> Dict[str, str]:
    props = {}
    lowered = sentence.lower()
    for prop in ONTOLOGICAL_META_MODEL.get(entity_type, []):
        if prop in lowered:
            props[prop] = sentence
    return props


def _sentence_subject(sentence: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff ]", "", sentence).strip()
    if not cleaned:
        return "concept"
    tokens = cleaned.split()
    return tokens[0].capitalize()


def _split_sentences(text: str) -> List[str]:
    return [segment.strip() for segment in re.split(r"[。\\.?!]", text) if segment.strip()]


def _llm_extract_entities(user_input: str) -> Optional[ExtractionPayload]:
    llm = get_llm_client()
    if not llm:
        return None
    structured = llm.with_structured_output(ExtractionPayload)
    prompt = (
        "从以下段落中提取所有世界规则 (Axioms) 和具体事实 (Facts)。"
        "规则是描述系统/社会/物理约束的陈述；事实是对当前状态的描述。\n\n"
        f"文本:\n{user_input}"
    )
    try:
        return structured.invoke(prompt)
    except Exception:
        return None


def analyze_input_for_gaps_and_conflicts(user_input: str, internal_collection) -> Dict[str, Any]:
    """Analyze user input to detect ontological gaps and potential conflicts."""

    sentences = _split_sentences(user_input)
    extracted_entities: List[Dict[str, Any]] = []

    llm_payload = _llm_extract_entities(user_input)
    axioms: List[ExtractedAxiom] = llm_payload.axioms if llm_payload else []
    facts: List[ExtractedFact] = llm_payload.facts if llm_payload else []

    if not axioms and not facts:
        for sentence in sentences:
            entity_type = _classify_sentence(sentence)
            props = _extract_props(sentence, entity_type)
            subject = _sentence_subject(sentence)
            extracted_entities.append({"name": subject, "type": entity_type, "props": props})

            if any(keyword in sentence.lower() for keyword in ["must", "should", "need", "required", "always"]):
                axioms.append(
                    ExtractedAxiom(
                        description=sentence,
                        subjects=[subject],
                        type="Social" if entity_type == "SocialRule" else "Physical",
                    )
                )
            else:
                facts.append(
                    ExtractedFact(description=sentence, entity=subject, property="statement", value=sentence)
                )
    else:
        for axiom in axioms:
            extracted_entities.append({"name": (axiom.subjects or ["规则"])[0], "type": "SocialRule", "props": {}})
        for fact in facts:
            extracted_entities.append({"name": fact.entity, "type": "Entity", "props": {"statement": fact.description}})

    gap_list: List[str] = []
    for entity in extracted_entities:
        entity_type = entity["type"]
        required_props = ONTOLOGICAL_META_MODEL.get(entity_type, [])
        missing = [prop for prop in required_props if prop not in entity["props"]]
        if missing:
            gap_list.append(f"{entity_type} '{entity['name']}' 缺少属性: {missing}")

    conflict_list: List[str] = []
    try:
        results = internal_collection.get(include=["documents", "metadatas"])
        existing_docs = results.get("documents", []) or []
        existing_meta = results.get("metadatas", []) or []
    except Exception:  # pragma: no cover - guard against storage failures
        existing_docs, existing_meta = [], []

    for sentence in sentences:
        normalized = sentence.lower()
        for existing_text, metadata in zip(existing_docs, existing_meta):
            existing_norm = (existing_text or "").lower()
            entities_field = metadata.get("entities") or ""
            if isinstance(entities_field, str):
                entities = [token.strip() for token in entities_field.split(",") if token.strip()]
            else:
                entities = []
            same_subject = _sentence_subject(sentence) in entities if entities else False
            contradictory = (" not " in normalized or " never " in normalized) != (
                " not " in existing_norm or " never " in existing_norm
            )
            if same_subject and contradictory:
                conflict_list.append(
                    f"新的描述“{sentence}”可能与先前的规则“{existing_text}”矛盾 (实体: {metadata.get('entities')})"
                )

    return {
        "gaps": gap_list,
        "conflicts": conflict_list,
        "extracted_data": {"axioms": axioms, "facts": facts},
    }


def ingest_to_wm_kg(
    *,
    graph: WorldModelKnowledgeGraph,
    analysis_results: Dict[str, Any],
    version: int,
    source_id: str,
) -> int:
    """Persist extracted axioms/facts to the internal KB."""
    extracted = analysis_results.get("extracted_data") or {}
    axioms: List[ExtractedAxiom] = extracted.get("axioms", [])
    facts: List[ExtractedFact] = extracted.get("facts", [])

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for axiom in axioms:
        entities_str = ", ".join(axiom.subjects) if axiom.subjects else ""
        documents.append(axiom.description)
        metadatas.append(
            {
                "source": source_id,
                "type": "AXIOM",
                "category": axiom.type,
                "entities": entities_str,
                "version": version,
            }
        )
        ids.append(f"axiom-{uuid.uuid4().hex}")

    for fact in facts:
        entity_str = fact.entity
        documents.append(fact.description)
        metadatas.append(
            {
                "source": source_id,
                "type": "FACT",
                "category": fact.property,
                "entities": entity_str,
                "version": version,
            }
        )
        ids.append(f"fact-{uuid.uuid4().hex}")

    if not documents:
        return 0

    graph.collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(documents)


def get_axioms_text(graph: WorldModelKnowledgeGraph) -> str:
    """Aggregate all AXIOM entries into a formatted text block."""
    docs = graph.list_axioms()
    if not docs:
        return "（目前没有已知公理。系统仍在构建中。）"
    bullet_lines = [f"- {doc.page_content}" for doc in docs]
    return "\n".join(bullet_lines)
