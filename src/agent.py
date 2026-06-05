from typing import Callable

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
        self.SYSTEM_PROMPT = """
You are an assistant that answers questions based on the provided context chunks. Use the context to answer the question as accurately as possible.
{query}

Context:
{context}
"""

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        context = "\n\n".join(result["content"] for result in results if result.get("content"))
        prompt = self.SYSTEM_PROMPT.format(query=question, context=context)
        return self.llm_fn(prompt)