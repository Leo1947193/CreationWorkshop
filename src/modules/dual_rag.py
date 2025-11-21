"""Dual RAG retriever implementations (Module 3 helpers)."""
from __future__ import annotations

from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# Ensure .env is loaded so Tavily picks up the API key when running via REPL/scripts.
load_dotenv()

try:
    from langchain_tavily._utilities import TavilySearchAPIWrapper  # type: ignore
except ImportError:  # pragma: no cover
    TavilySearchAPIWrapper = None  # type: ignore


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
        self._tavily: Optional[TavilySearchAPIWrapper] = None
        if TavilySearchAPIWrapper is not None:
            try:
                # Use the raw API wrapper from langchain-tavily to avoid interface drift.
                self._tavily = TavilySearchAPIWrapper()
            except Exception:
                self._tavily = None

    def _fetch_web(self, query: str) -> List[Document]:
        if not self._tavily:
            raise RuntimeError("Tavily retriever not available; set TAVILY_API_KEY to enable web search.")
        try:
            payload = self._tavily.raw_results(
                query=query,
                max_results=self._max_results * 3,
                search_depth="advanced",
                include_domains=None,
                exclude_domains=None,
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                include_image_descriptions=False,
                include_favicon=False,
                topic="general",
                time_range=None,
                country=None,
                auto_parameters=True,
                start_date=None,
                end_date=None,
            )
            results = payload.get("results", []) if isinstance(payload, dict) else []
            docs: List[Document] = []
            for item in results:
                content = item.get("content") or item.get("title") or ""
                url = item.get("url")
                if not content:
                    continue
                docs.append(Document(page_content=content, metadata={"url": url}))
            return docs
        except Exception as exc:  # pragma: no cover - surfacing upstream
            raise RuntimeError(f"Tavily retrieval failed: {exc}") from exc

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

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = self._fetch_web(query)
        if not docs:
            raise RuntimeError("Tavily returned no documents for the query.")
        docs = self._deduplicate(docs)
        return docs[: self._max_results]
