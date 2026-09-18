import numpy as np
import pytest
from backend.services.embeddings import EmbeddingService


@pytest.fixture(scope="module")
def embedding_service():
    # scope="module" ensures the transformer model is downloaded and loaded only once for all tests
    return EmbeddingService()


def test_embedding_dimension_and_type(embedding_service):
    text = "Gradient descent is an optimization algorithm."
    vector = embedding_service.generate_embeddings(text)

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert vector.shape == (1, 384)


def test_batch_embeddings(embedding_service):
    chunks = [
        "First chunk about loss functions.",
        "Second chunk about learning rates.",
        "Third chunk about backpropagation."
    ]
    vectors = embedding_service.generate_embeddings(chunks)

    assert vectors.shape == (3, 384)


def test_l2_normalization(embedding_service):
    text = "Verify unit length vector."
    vector = embedding_service.generate_embeddings(text)[0]

    # Calculate L2 norm: sqrt(sum(v_i^2))
    norm = np.linalg.norm(vector)
    # Norm should be 1.0 (allowing minor floating-point tolerance)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_semantic_similarity(embedding_service):
    # Sentences with similar meaning should have higher dot product than unrelated ones
    query = "How to adjust weights in a neural network?"
    relevant_doc = "Backpropagation computes gradients to update network weights."
    irrelevant_doc = "The recipe calls for two cups of flour and butter."

    q_vec = embedding_service.generate_embeddings(query)[0]
    rel_vec = embedding_service.generate_embeddings(relevant_doc)[0]
    irrel_vec = embedding_service.generate_embeddings(irrelevant_doc)[0]

    # Since vectors are unit-normalized, dot product == cosine similarity
    similarity_rel = np.dot(q_vec, rel_vec)
    similarity_irrel = np.dot(q_vec, irrel_vec)

    assert similarity_rel > similarity_irrel
