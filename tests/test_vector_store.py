import os
import numpy as np
import pytest
from backend.services.vector_store import VectorStore, VectorStoreError


@pytest.fixture
def sample_data():
    dim = 4
    # Create two orthogonal unit vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = np.vstack([v1, v2])

    chunks = [
        {"chunk_id": "doc1_p1_c0", "filename": "doc1.pdf", "page_number": 1, "text": "Topic A"},
        {"chunk_id": "doc1_p1_c1", "filename": "doc1.pdf", "page_number": 1, "text": "Topic B"},
    ]
    return dim, embeddings, chunks


def test_create_and_search(sample_data):
    dim, embeddings, chunks = sample_data
    store = VectorStore(dimension=dim)
    store.create_and_add(embeddings, chunks)

    # Search for vector identical to v1
    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = store.search(query, top_k=2)

    assert len(results) == 2
    # First match should be Topic A with similarity ~1.0
    assert results[0]["chunk_id"] == "doc1_p1_c0"
    assert np.isclose(results[0]["score"], 1.0, atol=1e-5)
    # Second match is orthogonal (Topic B) with similarity ~0.0
    assert results[1]["chunk_id"] == "doc1_p1_c1"
    assert np.isclose(results[1]["score"], 0.0, atol=1e-5)


def test_save_and_load(tmp_path, sample_data):
    dim, embeddings, chunks = sample_data
    store = VectorStore(dimension=dim)
    store.create_and_add(embeddings, chunks)

    save_dir = os.path.join(tmp_path, "vector_data")
    store.save(save_dir)

    # Load into a new store
    loaded_store = VectorStore(dimension=dim)
    loaded_store.load(save_dir)

    assert loaded_store.index.ntotal == 2
    assert len(loaded_store.metadata) == 2

    # Query loaded store
    query = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    results = loaded_store.search(query, top_k=1)
    assert results[0]["chunk_id"] == "doc1_p1_c1"


def test_mismatched_lengths_raises_error():
    store = VectorStore(dimension=4)
    embeddings = np.zeros((3, 4), dtype=np.float32)
    chunks = [{"text": "Only one"}]

    with pytest.raises(VectorStoreError):
        store.create_and_add(embeddings, chunks)
