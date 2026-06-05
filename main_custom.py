from __future__ import annotations

import os
import sys
import csv
from time import perf_counter

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import PropositionChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.llm import LLM


class MeteredLLM:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def generate(self, prompt: str) -> str:
        result = self.llm.generate(prompt)
        usage = getattr(self.llm, "last_usage", None) or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        return result


def select_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def run_chunk(essay):
    started_at = perf_counter()
    llm = MeteredLLM(LLM())
    atomic_prompt = (
        "You are an expert at extracting atomic propositions from text. Extract clear, concise, and factual propositions "
        "from the given text. Return each proposition on a new line."
    )
    res = llm.generate(f"{atomic_prompt}\n\nText:\n{essay}")
    propositions = [line.strip() for line in res.split("\n") if line.strip()]
    print(f"Extracted {len(propositions)} Propositions:")

    custom_chunker = PropositionChunker(
        propositions=propositions,
        llm=llm,
        load_from_file=False,
        propositions_file="output/chunk_proposition.json",
    )
    custom_chunker.pretty_print_chunks()
    chunks = custom_chunker.get_chunks(get_type="list_of_strings")

    elapsed_seconds = perf_counter() - started_at
    print("\n=== Run Metrics ===")
    print(f"Time elapsed: {elapsed_seconds:.2f} s")
    print(f"Prompt tokens: {llm.prompt_tokens}")
    print(f"Completion tokens: {llm.completion_tokens}")
    print(f"Total tokens: {llm.total_tokens}")

    return chunks


def build_chunk_documents(csv_path: str = "data/test.csv") -> list[Document]:
    documents: list[Document] = []

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            essay = str(row.get("content", "")).strip()
            if not essay:
                continue

            title = row.get("title") or f"row {row_index}"
            chunks = run_chunk(essay)

            for chunk_index, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        id=f"{row_index}_{chunk_index}",
                        content=chunk,
                        metadata={
                            "source_row": int(row_index),
                            "title": str(title),
                            "chunk_index": chunk_index,
                            "source": csv_path,
                        },
                    )
                )

    return documents


def build_chunk_agent(csv_path: str = "data/test.csv") -> tuple[KnowledgeBaseAgent, EmbeddingStore]:
    embedder = select_embedder()
    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    chunk_documents = build_chunk_documents(csv_path)
    store = EmbeddingStore(collection_name="chunk_proposition_store", embedding_fn=embedder)
    store.add_documents(chunk_documents)
    print(f"\nStored {store.get_collection_size()} chunk documents in ChromaDB")

    llm = LLM()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm.generate)
    return agent, store


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip() or "What are the main ideas in the stored chunks?"
    agent, store = build_chunk_agent()

    print("\n=== Retrieval Test ===")
    retrieved = store.search(question, top_k=3)
    for index, result in enumerate(retrieved, start=1):
        print(f"{index}. score={result['score']:.3f} title={result['metadata'].get('title')}")
        print(f"   chunk: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== Agent Answer ===")
    print(f"Question: {question}")
    print(agent.answer(question, top_k=3))

