"""Dual RAG retriever implementations (Module 3 helpers)."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

try:
    from langchain_tavily import TavilySearchAPIRetriever
except ImportError:  # pragma: no cover
    TavilySearchAPIRetriever = None  # type: ignore


class DualRARetriever(BaseRetriever):
    """Fan-out retriever that merges internal and external knowledge bases."""

    def __init__(self, internal_retriever: BaseRetriever, external_retriever: BaseRetriever):
        super().__init__()
        object.__setattr__(self, "_internal", internal_retriever)
        object.__setattr__(self, "_external", external_retriever)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        internal_docs = self._internal.invoke(query)
        for doc in internal_docs:
            doc.metadata.setdefault("source", "KB_INTERNAL")
            doc.metadata.setdefault("source_type", "Axiom")

        external_docs = self._external.invoke(query)
        for doc in external_docs:
            doc.metadata.setdefault("source", "KB_EXTERNAL")
            doc.metadata.setdefault("source_type", "Principle")

        return internal_docs + external_docs


class ValidatedWebRetriever(BaseRetriever):
    """Lightweight web retriever that prefers breadth and simple deduplication."""

    def __init__(self, max_results: int = 4):
        super().__init__()
        object.__setattr__(self, "_max_results", max_results)
        self._tavily: Optional[BaseRetriever] = None
        if TavilySearchAPIRetriever is not None:
            try:
                self._tavily = TavilySearchAPIRetriever(k=max_results * 3)
            except Exception:
                self._tavily = None

        self._fallback_corpus = [
            Document(
                page_content="海上城市依赖可再生能源时，如果没有储能与备用系统，连续阴影或风暴会导致全面停摆。",
                metadata={"url": "https://example.com/energy"},
            ),
            Document(
                page_content="Goodhart 定律提醒我们：激励指标一旦成为目标，就会被操纵，需要多维度约束。",
                metadata={"url": "https://en.wikipedia.org/wiki/Goodhart%27s_law"},
            ),
            Document(
                page_content="生态系统需要闭环维护，淡水或资源循环中断会在封闭环境里迅速放大风险。",
                metadata={"url": "https://example.com/ecology"},
            ),
        ]

    def _fetch_web(self, query: str) -> List[Document]:
        if not self._tavily:
            return []
        try:
            return self._tavily.invoke(query)
        except Exception:
            return []

    @staticmethod
    def _deduplicate(docs: List[Document]) -> List[Document]:
        seen = set()
        unique_docs: List[Document] = []
        for doc in docs:
            key = doc.metadata.get("url") or doc.page_content[:200]
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs

    def _fallback(self) -> List[Document]:
        return self._fallback_corpus[: self._max_results]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = self._fetch_web(query)
        if not docs:
            return self._fallback()

        docs = self._deduplicate(docs)
        return docs[: self._max_results]
