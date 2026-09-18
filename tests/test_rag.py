from unittest.mock import MagicMock
import numpy as np
import pytest
from backend.services.rag import RAGService


@pytest.fixture
def mock_embedding_service():
    mock = MagicMock()
    mock.generate_embeddings.return_value = np.zeros((1, 384), dtype=np.float32)
    return mock


@pytest.fixture
def mock_vector_store():
    mock = MagicMock()
    mock.search.return_value = [
        {
            "chunk_id": "doc.pdf_p2_c0",
            "filename": "doc.pdf",
            "page_number": 2,
            "text": "Gradient descent minimizes the loss function.",
            "score": 0.89
        },
        {
            "chunk_id": "doc.pdf_p2_c1",
            "filename": "doc.pdf",
            "page_number": 2,
            "text": "The learning rate controls update magnitude.",
            "score": 0.82
        }
    ]
    return mock


def test_rag_answer_with_sources(mock_embedding_service, mock_vector_store):
    rag = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        api_key="fake_test_key"
    )

    # Mock the GenAI response
    mock_response = MagicMock()
    mock_response.text = "Gradient descent is an optimization method that minimizes loss."
    rag.client = MagicMock()
    rag.client.models.generate_content.return_value = mock_response

    result = rag.answer_question("What is gradient descent?")

    assert "minimizes loss" in result["answer"]
    assert len(result["sources"]) == 1  # Deduplicated from 2 chunks on page 2
    assert result["sources"][0]["filename"] == "doc.pdf"
    assert result["sources"][0]["page"] == 2


def test_rag_fallback_when_info_missing(mock_embedding_service, mock_vector_store):
    rag = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        api_key="fake_test_key"
    )

    mock_response = MagicMock()
    mock_response.text = "I couldn't find this information in the uploaded document."
    rag.client = MagicMock()
    rag.client.models.generate_content.return_value = mock_response

    result = rag.answer_question("What is the recipe for chocolate cake?")

    assert "I couldn't find this information" in result["answer"]
    assert result["sources"] == []
