from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from src.embeddings import LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.llm import LLM
from src.chunking import PropositionChunker

DEFAULT_CSV_PATH = "data/10k_data.csv"
DEFAULT_BENCHMARK_PATH = "data/benchmark.md"
DEFAULT_OUTPUT_PATH = "output/rag_benchmark_answers.md"
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

BENCHMARK_ROW_INDEX_BY_TARGET = {
    "news_1": 4497,
    "news_2": 764,
    "news_3": 5679,
    "news_4": 9015,
    "news_5": 4033,
    "news_6": 3526,
    "news_7": 2465,
    "news_8": 8240,
    "news_9": 2697,
    "news_10": 2136,
}


@dataclass
class BenchmarkItem:
    number: int
    question: str
    target_doc: str
    gold_answer: str


class MeteredLLM:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def generate(self, prompt: str) -> str:
        result = self.llm.generate(prompt)
        print('RESPONSE FROM LLM:\n', result)
        usage = getattr(self.llm, "last_usage", None) or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)
        return result


def select_embedder():
    return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))


def run_chunk(
    essay: str,
    include_metadata: bool = False,
    verbose: bool = False,
) -> list[str] | list[dict[str, Any]]:
    started_at = perf_counter()
    llm = LLM()
    atomic_prompt = (
        "You are an expert at extracting atomic propositions from text. Extract clear, concise, and factual propositions "
        "from the given text. Return each proposition on a new line."
    )
    prompt = f"{atomic_prompt}\n\nText:\n{essay}"

    print("PROMPT LEN:", len(prompt))

    res = llm.generate(prompt)
    print("RES\n:",res)

    propositions = [line.strip() for line in res.split("\n") if line.strip()]
    print("Num propositions:", len(propositions))
    if len(propositions) <= 1:
        raise ValueError("LLM did not return any propositions. Check the LLM response and prompt.")

    if verbose:
        print(f"Extracted {len(propositions)} propositions")

    custom_chunker = PropositionChunker(
        propositions=propositions,
        llm=llm,
        load_from_file=False,
        propositions_file="output/chunk_proposition.json",
        print_logging=verbose,
    )
    chunks_dict = custom_chunker.get_chunks(get_type="dict")

    print("Chunk count:", len(chunks_dict))

    if verbose:
        custom_chunker.pretty_print_chunks()

    chunk_payloads: list[dict[str, Any]] = []
    for chunk_id, chunk in custom_chunker.get_chunks(get_type="dict").items():
        propositions_in_chunk = [
            str(proposition).strip()
            for proposition in chunk.get("propositions", [])
            if str(proposition).strip()
        ]
        chunk_payloads.append(
            {
                "chunk_id": chunk.get("chunk_id", chunk_id),
                "chunk_index": chunk.get("chunk_index", len(chunk_payloads)),
                "chunk_title": chunk.get("title", ""),
                "chunk_summary": chunk.get("summary", ""),
                "content": " ".join(propositions_in_chunk),
                "propositions": propositions_in_chunk,
            }
        )

    elapsed_seconds = perf_counter() - started_at
    if verbose:
        print("\n=== Chunking Metrics ===")
        print(f"Time elapsed: {elapsed_seconds:.2f} s")
        # print(f"Prompt tokens: {llm.prompt_tokens}")
        # print(f"Completion tokens: {llm.completion_tokens}")
        # print(f"Total tokens: {llm.total_tokens}")

    if include_metadata:
        return chunk_payloads
    return [chunk["content"] for chunk in chunk_payloads]


def parse_benchmark_questions(benchmark_path: str = DEFAULT_BENCHMARK_PATH) -> list[BenchmarkItem]:
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    items: list[BenchmarkItem] = []
    current: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        question_match = re.match(r"Câu hỏi\s+(\d+):\s*(.+)", stripped)
        if question_match:
            current = {
                "number": int(question_match.group(1)),
                "question": question_match.group(2).strip(),
            }
            continue

        target_match = re.match(r"Tài liệu đích:\s*(.+)", stripped)
        if target_match and current:
            current["target_doc"] = target_match.group(1).strip()
            continue

        answer_match = re.match(r"Câu trả lời đúng:\s*(.+)", stripped)
        if answer_match and current:
            current["gold_answer"] = answer_match.group(1).strip()
            if {"number", "question", "target_doc", "gold_answer"}.issubset(current):
                items.append(BenchmarkItem(**current))
            current = {}

    return items


def _row_selection_for_targets(
    csv_path: str,
    target_doc_ids: list[str] | None,
    all_rows: bool,
) -> tuple[set[int] | None, dict[int, str]]:
    if all_rows:
        return None, {}

    if target_doc_ids is None:
        if Path(csv_path).name != Path(DEFAULT_CSV_PATH).name:
            return None, {}
        target_doc_ids = list(BENCHMARK_ROW_INDEX_BY_TARGET)

    missing_targets = [target_doc for target_doc in target_doc_ids if target_doc not in BENCHMARK_ROW_INDEX_BY_TARGET]
    if missing_targets:
        print(f"Warning: no CSV row mapping for target docs: {', '.join(missing_targets)}")

    row_to_target_doc = {
        BENCHMARK_ROW_INDEX_BY_TARGET[target_doc]: target_doc
        for target_doc in target_doc_ids
        if target_doc in BENCHMARK_ROW_INDEX_BY_TARGET
    }
    return set(row_to_target_doc), row_to_target_doc


def _chunk_document_content(chunk: dict[str, Any]) -> str:
    parts = []
    if chunk.get("chunk_title"):
        parts.append(f"Chunk title: {chunk['chunk_title']}")
    if chunk.get("chunk_summary"):
        parts.append(f"Chunk summary: {chunk['chunk_summary']}")
    parts.append(f"Chunk propositions: {chunk.get('content', '')}")
    return "\n".join(parts)


def build_chunk_documents(
    csv_path: str = DEFAULT_CSV_PATH,
    target_doc_ids: list[str] | None = None,
    all_rows: bool = False,
    verbose_chunking: bool = False,
) -> list[Document]:
    from src.models import Document

    documents: list[Document] = []
    row_indices, row_to_target_doc = _row_selection_for_targets(csv_path, target_doc_ids, all_rows)

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_indices is not None and row_index not in row_indices:
                continue

            essay = str(row.get("content", "")).strip()
            if not essay:
                continue

            title = str(row.get("title") or f"row {row_index}")
            target_doc = row_to_target_doc.get(row_index, f"row_{row_index}")
            print(f"Chunking {target_doc}: {title}")
            chunks = run_chunk(essay, include_metadata=True, verbose=verbose_chunking)

            for chunk_index, chunk in enumerate(chunks):
                if not isinstance(chunk, dict) or not chunk.get("content"):
                    continue
                documents.append(
                    Document(
                        id=f"{target_doc}_{chunk_index}",
                        content=_chunk_document_content(chunk),
                        metadata={
                            "target_doc": target_doc,
                            "source_doc": target_doc,
                            "source_row": int(row_index),
                            "title": title,
                            "url": str(row.get("url", "")),
                            "category": str(row.get("category", "")),
                            "time": str(row.get("time", "")),
                            "chunk_index": chunk_index,
                            "chunk_id": str(chunk.get("chunk_id", "")),
                            "chunk_title": str(chunk.get("chunk_title", "")),
                            "chunk_summary": str(chunk.get("chunk_summary", "")),
                            "source": csv_path,
                        },
                    )
                )

    return documents


def build_chunk_agent(
    csv_path: str = DEFAULT_CSV_PATH,
    target_doc_ids: list[str] | None = None,
    all_rows: bool = False,
    verbose_chunking: bool = False,
) -> tuple[KnowledgeBaseAgent, EmbeddingStore]:

    embedder = select_embedder()
    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    chunk_documents = build_chunk_documents(
        csv_path=csv_path,
        target_doc_ids=target_doc_ids,
        all_rows=all_rows,
        verbose_chunking=verbose_chunking,
    )
    if not chunk_documents:
        raise RuntimeError("No chunk documents were created. Check the CSV path and benchmark row mapping.")

    store = EmbeddingStore(collection_name="chunk_proposition_store", embedding_fn=embedder)
    store.add_documents(chunk_documents)
    print(f"\nStored {store.get_collection_size()} chunk documents in ChromaDB")

    llm = LLM()
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm.generate)
    return agent, store


def _score_text(score: Any) -> str:
    if isinstance(score, (int, float)):
        return f"{score:.3f}"
    return str(score)


def print_answer_bundle(
    item_number: int,
    question: str,
    bundle: dict[str, Any],
    target_doc: str | None = None,
    gold_answer: str | None = None,
) -> None:
    print("\n" + "=" * 80)
    print(f"Câu hỏi {item_number}: {question}")
    if target_doc:
        print(f"Tài liệu đích: {target_doc}")
    print("\nCâu trả lời RAG:")
    print(bundle["answer"])
    if gold_answer:
        print("\nCâu trả lời đúng:")
        print(gold_answer)
    print("\nNguồn truy xuất:")
    for source in bundle.get("sources", []):
        print(
            f"- rank={source.get('rank')} score={_score_text(source.get('score'))} "
            f"target={source.get('target_doc')} chunk={source.get('chunk_index')} "
            f"title={source.get('title')}"
        )


def write_benchmark_report(
    output_path: str,
    benchmark_results: list[tuple[BenchmarkItem, dict[str, Any]]],
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# RAG Benchmark Answers", ""]
    for item, bundle in benchmark_results:
        lines.extend(
            [
                f"## Câu hỏi {item.number}",
                "",
                f"**Question:** {item.question}",
                "",
                f"**Target document:** {item.target_doc}",
                "",
                f"**RAG answer:** {bundle['answer']}",
                "",
                f"**Gold answer:** {item.gold_answer}",
                "",
                "**Retrieved sources:**",
            ]
        )
        for source in bundle.get("sources", []):
            lines.append(
                f"- rank={source.get('rank')} score={_score_text(source.get('score'))} "
                f"target={source.get('target_doc')} chunk={source.get('chunk_index')} "
                f"title={source.get('title')}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved benchmark answers to {path}")


def run_benchmark(
    csv_path: str = DEFAULT_CSV_PATH,
    benchmark_path: str = DEFAULT_BENCHMARK_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    top_k: int = 5,
    all_rows: bool = False,
    verbose_chunking: bool = False,
) -> int:
    benchmark_items = parse_benchmark_questions(benchmark_path)
    if not benchmark_items:
        print(f"No benchmark questions found in {benchmark_path}")
        return 1

    target_doc_ids = None if all_rows else [item.target_doc for item in benchmark_items]
    agent, _store = build_chunk_agent(
        csv_path=csv_path,
        target_doc_ids=target_doc_ids,
        all_rows=all_rows,
        verbose_chunking=verbose_chunking,
    )

    benchmark_results: list[tuple[BenchmarkItem, dict[str, Any]]] = []
    for item in benchmark_items:
        bundle = agent.answer_with_sources(item.question, top_k=top_k)
        benchmark_results.append((item, bundle))
        print_answer_bundle(
            item_number=item.number,
            question=item.question,
            bundle=bundle,
            target_doc=item.target_doc,
            gold_answer=item.gold_answer,
        )

    if output_path:
        write_benchmark_report(output_path, benchmark_results)
    return 0


def run_single_question(
    question: str,
    csv_path: str = DEFAULT_CSV_PATH,
    benchmark_path: str = DEFAULT_BENCHMARK_PATH,
    top_k: int = 5,
    all_rows: bool = False,
    verbose_chunking: bool = False,
) -> int:
    target_doc_ids = None
    if not all_rows and Path(benchmark_path).exists():
        benchmark_items = parse_benchmark_questions(benchmark_path)
        target_doc_ids = [item.target_doc for item in benchmark_items]

    agent, store = build_chunk_agent(
        csv_path=csv_path,
        target_doc_ids=target_doc_ids,
        all_rows=all_rows,
        verbose_chunking=verbose_chunking,
    )

    print("\n=== Retrieval Test ===")
    retrieved = store.search(question, top_k=top_k)
    for index, result in enumerate(retrieved, start=1):
        metadata = result.get("metadata", {})
        print(
            f"{index}. score={_score_text(result.get('score'))} "
            f"target={metadata.get('target_doc')} title={metadata.get('title')}"
        )
        print(f"   chunk: {result['content'][:160].replace(chr(10), ' ')}...")

    bundle = agent.answer_with_sources(question, top_k=top_k)
    print_answer_bundle(item_number=0, question=question, bundle=bundle)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run proposition-chunk RAG over benchmark questions.")
    parser.add_argument("question_words", nargs="*", help="Optional single question. If omitted, benchmark.md is run.")
    parser.add_argument("--question", help="Optional single question string.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--benchmark-path", default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--all-rows", action="store_true", help="Chunk every row in the CSV instead of benchmark targets.")
    parser.add_argument("--verbose-chunking", action="store_true")
    args = parser.parse_args(argv)

    question = args.question or " ".join(args.question_words).strip()
    if question:
        return run_single_question(
            question=question,
            csv_path=args.csv_path,
            benchmark_path=args.benchmark_path,
            top_k=args.top_k,
            all_rows=args.all_rows,
            verbose_chunking=args.verbose_chunking,
        )

    return run_benchmark(
        csv_path=args.csv_path,
        benchmark_path=args.benchmark_path,
        output_path=args.output_path,
        top_k=args.top_k,
        all_rows=args.all_rows,
        verbose_chunking=args.verbose_chunking,
    )


if __name__ == "__main__":
    raise SystemExit(main())
    # llm = MeteredLLM(LLM())
    # atomic_prompt = (
    #     "You are an expert at extracting atomic propositions from text. Extract clear, concise, and factual propositions "
    #     "from the given text. Return each proposition on a new line."
    # )
    # res = llm.generate(f"{atomic_prompt}\n\nText:\nThis is a test essay. It contains multiple sentences. The goal is to extract atomic propositions.")
    # propositions = [line.strip() for line in res.split("\n") if line.strip()]
    # print("Num propositions:", len(propositions))
    # print(propositions)