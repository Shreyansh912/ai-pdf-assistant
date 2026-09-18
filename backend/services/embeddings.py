from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Manages loading the sentence transformer model and generating
    L2-normalized embeddings suitable for FAISS vector search.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Load weights once during initialization
        self.model = SentenceTransformer(model_name)
        # Vector dimension for all-MiniLM-L6-v2 is 384
        self.dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generates L2-normalized float32 embeddings for a string or list of strings.

        Args:
            texts: A single text query or a list of text chunks.

        Returns:
            np.ndarray of shape (N, dimension) with dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # normalize_embeddings=True divides each vector by its L2 norm
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # FAISS requires contiguous float32 data
        return np.ascontiguousarray(embeddings, dtype=np.float32)
