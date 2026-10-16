import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional


class VectorStoreError(Exception):
    """Custom exception for VectorStore operations."""
    pass


class VectorStoreService:
    def __init__(self, store_dir: str = "data/vector_store", dimension: int = 384):
        self.store_dir = store_dir
        self.dimension = dimension
        self.index_path = os.path.join(store_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(store_dir, "metadata.json")
        self.registry_path = os.path.join(store_dir, "doc_registry.json")
        
        os.makedirs(store_dir, exist_ok=True)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
        self.registry: Dict[str, Dict[str, Any]] = {}
        self._load()

    def create_and_add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """Backward-compatible ingestion method for test suites."""
        if len(embeddings) != len(chunks):
            raise VectorStoreError(
                f"Embeddings count ({len(embeddings)}) does not match chunks count ({len(chunks)})"
            )
        self.index = faiss.IndexFlatIP(self.dimension)
        if len(embeddings) > 0:
            self.index.add(embeddings.astype(np.float32))
        self.metadata = chunks.copy()

    def add_document(
        self,
        document_id: str,
        filename: str,
        page_count: int,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
    ) -> int:
        if embeddings.shape[0] != len(chunks):
            raise VectorStoreError("Embedding row count must match chunk count.")

        if embeddings.shape[0] == 0:
            return len(self.metadata)

        self.index.add(embeddings.astype(np.float32))
        for chunk in chunks:
            chunk_entry = {
                "document_id": document_id,
                "filename": filename,
                "page": chunk.get("page") or chunk.get("page_number", 1),
                "text": chunk["text"],
            }
            self.metadata.append(chunk_entry)

        self.registry[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": len(chunks),
        }
        self._save()
        return len(self.metadata)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        search_k = min(self.index.ntotal, max(top_k * 4, 20) if document_id else top_k)
        scores, indices = self.index.search(query_vector.astype(np.float32), search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx].copy()
            if document_id and item.get("document_id") != document_id:
                continue
            item["score"] = float(score)
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self.registry.values())

    def save(self, path: Optional[str] = None) -> None:
        target_dir = path or self.store_dir
        os.makedirs(target_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(target_dir, "faiss_index.bin"))
        with open(os.path.join(target_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        target_dir = path or self.store_dir
        idx_p = os.path.join(target_dir, "faiss_index.bin")
        meta_p = os.path.join(target_dir, "metadata.json")
        if os.path.exists(idx_p) and os.path.exists(meta_p):
            self.index = faiss.read_index(idx_p)
            with open(meta_p, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _save(self):
        self.save(self.store_dir)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    def _load(self):
        self.load(self.store_dir)
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self.registry = json.load(f)


# Backward compatibility alias
VectorStore = VectorStoreService