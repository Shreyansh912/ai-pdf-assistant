import json
import os
from typing import List, Dict, Any, Optional
import faiss
import numpy as np


class VectorStoreError(Exception):
    """Exception raised for vector store operations failures."""
    pass


class VectorStore:
    """
    Manages a local FAISS index alongside a synchronized JSON metadata store.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # IndexFlatIP performs exact brute-force search using Inner Product (Cosine similarity for unit vectors)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

    def create_and_add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Populates the FAISS index and stores chunk metadata.

        Args:
            embeddings: Float32 numpy array of shape (N, dimension).
            chunks: List of chunk metadata dicts of length N.
        """
        if len(embeddings) != len(chunks):
            raise VectorStoreError(
                f"Mismatch: Got {len(embeddings)} embeddings but {len(chunks)} chunk metadata records."
            )

        if len(embeddings) == 0:
            return

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Reset any existing in-memory state
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

        # Add vectors to FAISS and record metadata
        self.index.add(embeddings)
        self.metadata = list(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the FAISS index for the top_k most similar chunks.

        Args:
            query_embedding: Float32 numpy array of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of dictionaries containing chunk data and a similarity 'score'.
        """
        if self.index.ntotal == 0:
            return []

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Ensure 2D query shape: (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)

        return results

    def save(self, directory: str) -> None:
        """
        Serializes the FAISS index and metadata to a target folder.
        """
        os.makedirs(directory, exist_ok=True)
        index_file = os.path.join(directory, "index.faiss")
        metadata_file = os.path.join(directory, "metadata.json")

        try:
            faiss.write_index(self.index, index_file)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise VectorStoreError(f"Failed to save vector store: {str(e)}")

    def load(self, directory: str) -> None:
        """
        Loads the FAISS index and metadata from disk into memory.
        """
        index_file = os.path.join(directory, "index.faiss")
        metadata_file = os.path.join(directory, "metadata.json")

        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise VectorStoreError(f"Index or metadata not found in directory: {directory}")

        try:
            self.index = faiss.read_index(index_file)
            with open(metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise VectorStoreError(f"Failed to load vector store: {str(e)}")
