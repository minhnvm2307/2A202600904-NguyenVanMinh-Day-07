from __future__ import annotations

import json
import math
import re
import uuid
from pathlib import Path
from typing import Iterable

from .llm import LLM


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        if len(current_text) <= self.chunk_size or not remaining_separators:
            return [current_text.strip()]

        separator = remaining_separators[0]
        parts = current_text.split(separator) if separator else [current_text]
        chunks: list[str] = []
        for part in parts:
            sub_chunks = self._split(part, remaining_separators[1:])
            chunks.extend(sub_chunks)
        return chunks


class PropositionChunker:
    """Group atomic propositions into topic-based chunks using an LLM."""

    def __init__(
        self,
        propositions: Iterable[str] | None = None,
        propositions_file: str = "output/chunk_proposition.json",
        load_from_file: bool = True,
        llm: LLM | None = None,
        model: str = "cx/gpt-5.5",
        provider: str = "openai",
        id_truncate_limit: int = 5,
        generate_new_metadata_ind: bool = True,
        print_logging: bool = True,
    ) -> None:
        self.id_truncate_limit = id_truncate_limit
        self.generate_new_metadata_ind = generate_new_metadata_ind
        self.print_logging = print_logging
        self.propositions_file = Path(propositions_file)
        self.llm = llm or LLM(model=model, provider=provider)
        self.chunks: dict[str, dict] = {}
        self.propositions: list[str] = []

        if load_from_file:
            self._load_from_file()

        if propositions is not None:
            self.add_propositions(propositions)
        elif load_from_file and self.propositions and not self.chunks:
            self.add_propositions(self.propositions)

    def _load_from_file(self) -> None:
        if not self.propositions_file.exists():
            return

        try:
            with self.propositions_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            if self.print_logging:
                print(f"Error loading propositions from file: {exc}")
            return

        if isinstance(payload, dict) and "chunks" in payload:
            raw_chunks = payload.get("chunks", {})
            if isinstance(raw_chunks, dict):
                self.chunks = raw_chunks

            raw_propositions = payload.get("propositions", [])
            if isinstance(raw_propositions, list):
                self.propositions = [str(item) for item in raw_propositions if str(item).strip()]
            return

        if isinstance(payload, list):
            self.propositions = [str(item) for item in payload if str(item).strip()]
            return

        if isinstance(payload, dict):
            raw_propositions = payload.get("propositions", [])
            if isinstance(raw_propositions, list):
                self.propositions = [str(item) for item in raw_propositions if str(item).strip()]

    def _save_state(self) -> None:
        self.propositions_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "propositions": self.propositions,
            "chunks": self.chunks,
        }
        with self.propositions_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def add_propositions(self, propositions: Iterable[str]) -> None:
        for proposition in propositions:
            self.add_proposition(proposition)

    def add_proposition(self, proposition: str) -> None:
        cleaned_proposition = str(proposition).strip()
        if not cleaned_proposition:
            return

        self.propositions.append(cleaned_proposition)

        if self.print_logging:
            print(f"\nAdding: '{cleaned_proposition}'")

        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(cleaned_proposition)
            return

        chunk_id = self._find_relevant_chunk(cleaned_proposition)

        if chunk_id:
            if self.print_logging:
                print(
                    f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}"
                )
            self.add_proposition_to_chunk(chunk_id, cleaned_proposition)
        else:
            if self.print_logging:
                print("No chunks found")
            self._create_new_chunk(cleaned_proposition)

    def add_proposition_to_chunk(self, chunk_id: str, proposition: str) -> None:
        self.chunks[chunk_id]["propositions"].append(proposition)

        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]["summary"] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]["title"] = self._update_chunk_title(self.chunks[chunk_id])

        self._save_state()

    def _update_chunk_summary(self, chunk: dict) -> str:
        prompt = (
            "You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. "
            "A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary "
            "which will inform viewers what a chunk group is about.\n\n"
            "A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.\n\n"
            "You will be given a group of propositions which are in the chunk and the chunk's current summary.\n\n"
            "Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food. "
            "Or month, generalize it to date and times.\n\n"
            "Only respond with the chunk new summary, nothing else.\n\n"
            f"Chunk's propositions:\n{chr(10).join(chunk['propositions'])}\n\n"
            f"Current chunk summary:\n{chunk['summary']}"
        )

        try:
            new_chunk_summary = self.llm.generate(prompt).strip()
        except Exception as exc:
            if self.print_logging:
                print(f"Error updating chunk summary: {exc}")
            new_chunk_summary = chunk["summary"]

        return new_chunk_summary or chunk["summary"]

    def _update_chunk_title(self, chunk: dict) -> str:
        prompt = (
            "You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. "
            "A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title "
            "which will inform viewers what a chunk group is about.\n\n"
            "A good title will say what the chunk is about.\n\n"
            "You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.\n\n"
            "Your title should anticipate generalization. If you get a proposition about apples, generalize it to food. "
            "Or month, generalize it to date and times.\n\n"
            "Only respond with the new chunk title, nothing else.\n\n"
            f"Chunk's propositions:\n{chr(10).join(chunk['propositions'])}\n\n"
            f"Chunk summary:\n{chunk['summary']}\n\n"
            f"Current chunk title:\n{chunk['title']}"
        )

        try:
            updated_chunk_title = self.llm.generate(prompt).strip()
        except Exception as exc:
            if self.print_logging:
                print(f"Error updating chunk title: {exc}")
            updated_chunk_title = chunk["title"]

        return updated_chunk_title or chunk["title"]

    def _get_new_chunk_summary(self, proposition: str) -> str:
        prompt = (
            "You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. "
            "You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.\n\n"
            "A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.\n\n"
            "You will be given a proposition which will go into a new chunk. This new chunk needs a summary.\n\n"
            "Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food. "
            "Or month, generalize it to date and times.\n\n"
            "Only respond with the new chunk summary, nothing else.\n\n"
            f"Determine the summary of the new chunk that this proposition will go into:\n{proposition}"
        )

        try:
            new_chunk_summary = self.llm.generate(prompt).strip()
        except Exception as exc:
            if self.print_logging:
                print(f"Error getting new chunk summary: {exc}")
            new_chunk_summary = "No summary available"

        return new_chunk_summary or "No summary available"

    def _get_new_chunk_title(self, summary: str) -> str:
        prompt = (
            "You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. "
            "You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.\n\n"
            "A good chunk title is brief but encompasses what the chunk is about.\n\n"
            "You will be given a summary of a chunk which needs a title.\n\n"
            "Only respond with the new chunk title, nothing else.\n\n"
            f"Determine the title of the chunk that this summary belongs to:\n{summary}"
        )

        try:
            new_chunk_title = self.llm.generate(prompt).strip()
        except Exception as exc:
            if self.print_logging:
                print(f"Error getting new chunk title: {exc}")
            new_chunk_title = "No title available"

        return new_chunk_title or "No title available"

    def _create_new_chunk(self, proposition: str) -> None:
        new_chunk_id = str(uuid.uuid4())[: self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            "chunk_id": new_chunk_id,
            "propositions": [proposition],
            "title": new_chunk_title,
            "summary": new_chunk_summary,
            "chunk_index": len(self.chunks),
        }

        self._save_state()

        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def get_chunk_outline(self) -> str:
        """Return a text outline of the currently known chunks."""
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = (
                f"Chunk ID: {chunk['chunk_id']}\nChunk Name: {chunk['title']}\nChunk Summary: {chunk['summary']}\n\n"
            )
            chunk_outline += single_chunk_string

        return chunk_outline

    def _find_relevant_chunk(self, proposition: str) -> str | None:
        current_chunk_outline = self.get_chunk_outline()

        prompt = (
            'Determine whether or not the "Proposition" should belong to any of the existing chunks.\n\n'
            "A proposition should belong to a chunk if their meaning, direction, or intention are similar.\n"
            "The goal is to group similar propositions and chunks.\n\n"
            'If you think a proposition should be joined with a chunk, return the chunk id.\n'
            'If you do not think an item should be joined with an existing chunk, just return "No chunks".\n\n'
            "Current Chunks:\n--Start of current chunks--\n"
            f"{current_chunk_outline}"
            "--End of current chunks--\n\n"
            f"Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"
        )

        try:
            chunk_found = self.llm.generate(prompt).strip()
        except Exception as exc:
            if self.print_logging:
                print(f"Error finding relevant chunk: {exc}")
            chunk_found = "No chunks"

        if not chunk_found or chunk_found.lower() == "no chunks":
            return None

        existing_chunk_ids = set(self.chunks)
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "", chunk_found.strip())

        if normalized in existing_chunk_ids:
            return normalized

        for chunk_id in existing_chunk_ids:
            if chunk_id in chunk_found:
                return chunk_id

        match = re.search(r'chunk_id["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]+)', chunk_found, re.IGNORECASE)
        if match and match.group(1) in existing_chunk_ids:
            return match.group(1)

        if len(normalized) == self.id_truncate_limit and normalized in existing_chunk_ids:
            return normalized

        return None

    def get_chunks(self, get_type: str = "dict"):
        """Return chunks as a dictionary or as a list of proposition strings."""
        if get_type == "dict":
            return self.chunks
        if get_type == "list_of_strings":
            chunks: list[str] = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([x for x in chunk["propositions"]]))
            return chunks
        raise ValueError("get_type must be 'dict' or 'list_of_strings'")

    def pretty_print_chunks(self) -> None:
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print("Propositions:")
            for prop in chunk["propositions"]:
                print(f"    -{prop}")
            print("\n\n")

    def pretty_print_chunk_outline(self) -> None:
        print("Chunk Outline\n")
        print(self.get_chunk_outline())

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length")
    try:
        dot_product = _dot(vec_a, vec_b)
        magnitude_a = math.sqrt(_dot(vec_a, vec_a))
        magnitude_b = math.sqrt(_dot(vec_b, vec_b))
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)
    except Exception:
        return 0.0


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        results = {}
        for name, strategy in strategies.items():
            chunks = strategy.chunk(text)
            chunk_lengths = [len(chunk) for chunk in chunks]
            results[name] = {
                "count": len(chunks),
                "chunks": chunks,
                "chunk_lengths": chunk_lengths,
                "min_chunk_length": min(chunk_lengths, default=0),
                "max_chunk_length": max(chunk_lengths, default=0),
                "avg_length": sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
            }
        return results