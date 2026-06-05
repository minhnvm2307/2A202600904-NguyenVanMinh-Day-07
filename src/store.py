from __future__ import annotations

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
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            client = chromadb.Client()
            self._use_chroma = True
            self._collection = client.get_or_create_collection(name=self._collection_name)
        except Exception:
            self._use_chroma = False
            self._collection = None

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

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored_records = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored_records.append({"score": score, **record})
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = [self._make_record(doc) for doc in docs if doc.content]
        if not self._use_chroma:
            self._store = records
        else:
            self._collection.add(
                ids=[record["id"] for record in records],
                metadatas=[record["metadata"] for record in records],
                documents=[record["doc"].content for record in records],
                embeddings=[record["embedding"] for record in records],
            )

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
            return results

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
            return filtered_records[:top_k]
        else:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                where=metadata_filter,
            )
            return results

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        init_len = len(self._store)
        if not self._use_chroma:
            self._store = [record for record in self._store if record["id"] != doc_id]
            return len(self._store) < init_len
        else:
            self._collection.delete(ids=[doc_id])
            return len(self._collection) < init_len