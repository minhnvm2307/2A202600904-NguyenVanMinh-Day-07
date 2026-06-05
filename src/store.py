from __future__ import annotations

import uuid
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        use_chroma: bool = True,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._use_chroma = use_chroma
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._collection_name = f"{collection_name}_{uuid.uuid4().hex}"
            self._collection = client.get_or_create_collection(name=self._collection_name)
        except Exception:
            self._use_chroma = False
            self._collection = None
            self._collection_name = collection_name

    def _make_record(self, doc: Document) -> dict[str, Any]:
        if not doc.content:
            raise ValueError("Document content is empty")
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "embedding": embedding,
            "doc": doc,
            "metadata": doc.metadata,
        }

    def _record_to_result(self, record: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": record["id"],
            "score": score,
            "content": record["doc"].content,
            "metadata": record["metadata"],
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored_records = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored_records.append(self._record_to_result(record, score))
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = [self._make_record(doc) for doc in docs if doc.content]
        if not records:
            return
        if not self._use_chroma:
            self._store.extend(records)
        else:
            unique_ids = []
            metadatas = []
            for offset, record in enumerate(records):
                unique_ids.append(f"{record['id']}::{self._next_index + offset}")
                metadata = dict(record["metadata"])
                metadata["document_id"] = record["id"]
                metadatas.append(metadata or {"document_id": record["id"]})
            self._collection.add(
                ids=unique_ids,
                metadatas=metadatas,
                documents=[record["doc"].content for record in records],
                embeddings=[record["embedding"] for record in records],
            )
            self._next_index += len(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if not self._use_chroma:
            return self._search_records(query, self._store, top_k)
        else:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
            )
            documents = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            normalized_results: list[dict[str, Any]] = []
            for index, content in enumerate(documents):
                normalized_results.append(
                    {
                        "id": metadatas[index].get("document_id", ids[index] if index < len(ids) else None),
                        "score": -distances[index] if index < len(distances) else 0.0,
                        "content": content,
                        "metadata": metadatas[index] if index < len(metadatas) else {},
                    }
                )
            return normalized_results

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if not self._use_chroma:
            return len(self._store)
        else:
            return self._collection.count()

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not self._use_chroma:
            filtered_records = [
                record
                for record in self._store
                if metadata_filter is None
                or all(record["metadata"].get(k) == v
                       for k, v in metadata_filter.items())
            ]
            return [self._record_to_result(record, 0.0) for record in filtered_records[:top_k]]
        else:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter,
            )
            documents = results.get("documents", [[]])[0]
            ids = results.get("ids", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            normalized_results: list[dict[str, Any]] = []
            for index, content in enumerate(documents):
                normalized_results.append(
                    {
                        "id": metadatas[index].get("document_id", ids[index] if index < len(ids) else None),
                        "score": -distances[index] if index < len(distances) else 0.0,
                        "content": content,
                        "metadata": metadatas[index] if index < len(metadatas) else {},
                    }
                )
            return normalized_results

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        init_len = self.get_collection_size()
        if not self._use_chroma:
            self._store = [record for record in self._store if record["id"] != doc_id]
            return self.get_collection_size() < init_len
        else:
            self._collection.delete(where={"document_id": doc_id})
            return self.get_collection_size() < init_len