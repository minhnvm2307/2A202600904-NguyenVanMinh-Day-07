from typing import Any, Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn
        self.SYSTEM_PROMPT = """You are a retrieval-augmented question answering assistant.
Answer the question using only the provided context chunks.
If the context does not contain the answer, say that the answer was not found in the retrieved context.
Keep the answer concise and preserve names, numbers, dates, and units exactly.

Question:
{query}

Retrieved context:
{context}
"""

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return the top retrieved chunks for a question."""
        results = self.store.search(question, top_k=top_k)
        return [result for result in results if result.get("content")]

    def _format_context(self, results: list[dict[str, Any]]) -> str:
        formatted_chunks = []
        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = metadata.get("target_doc") or metadata.get("source_doc") or metadata.get("source")
            title = metadata.get("title", "Untitled")
            chunk_summary = metadata.get("chunk_summary")
            content = str(result.get("content", "")).strip()
            header = f"[{rank}] source={source} title={title}"
            if chunk_summary:
                header = f"{header}\nsummary={chunk_summary}"
            formatted_chunks.append(f"{header}\ncontent={content}")
        return "\n\n".join(formatted_chunks)

    def build_prompt(self, question: str, results: list[dict[str, Any]]) -> str:
        """Build the final prompt sent to the answer LLM."""
        context = self._format_context(results)
        prompt = self.SYSTEM_PROMPT.format(query=question, context=context)
        return prompt

    def _source_summary(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = []
        for rank, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            sources.append(
                {
                    "rank": rank,
                    "id": result.get("id"),
                    "score": result.get("score"),
                    "target_doc": metadata.get("target_doc") or metadata.get("source_doc"),
                    "title": metadata.get("title"),
                    "chunk_index": metadata.get("chunk_index"),
                    "chunk_title": metadata.get("chunk_title"),
                    "source": metadata.get("source"),
                }
            )
        return sources

    def _fallback_answer(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "Không tìm thấy thông tin phù hợp trong ngữ cảnh được truy xuất."

        best = str(results[0].get("content", "")).strip()
        if len(best) > 900:
            best = best[:900].rsplit(" ", 1)[0] + "..."
        return f"Dựa trên đoạn liên quan nhất được truy xuất: {best}"

    def answer_with_sources(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Answer a question and return the answer with retrieved source metadata."""
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return {
                "question": question,
                "answer": self._fallback_answer(results),
                "sources": [],
                "retrieved": [],
            }

        prompt = self.build_prompt(question, results)
        try:
            answer = self.llm_fn(prompt).strip()
        except Exception:
            answer = self._fallback_answer(results)

        return {
            "question": question,
            "answer": answer or self._fallback_answer(results),
            "sources": self._source_summary(results),
            "retrieved": results,
        }

    def answer(self, question: str, top_k: int = 3) -> str:
        """Return only the generated answer text for backwards compatibility."""
        return self.answer_with_sources(question, top_k=top_k)["answer"]
