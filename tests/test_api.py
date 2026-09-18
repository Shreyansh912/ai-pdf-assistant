import io
import fitz
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_without_document_fails(client):
    # Ensure active FAISS index is cleared
    app.state.vector_store.index.reset()
    app.state.vector_store.metadata = []

    response = client.post("/ask", json={"question": "What is AI?"})
    assert response.status_code == 400
    assert "No document has been uploaded" in response.json()["detail"]


def test_upload_non_pdf_fails(client):
    file_content = b"Plain text note"
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", io.BytesIO(file_content), "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_upload_valid_pdf_and_ask(client):
    # 1. Create a dynamic test PDF in memory
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "The gradient descent algorithm uses learning rate alpha.")
    pdf_bytes = doc.write()
    doc.close()

    # 2. Upload PDF
    upload_res = client.post(
        "/upload",
        files={"file": ("test_doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    )
    assert upload_res.status_code == 200
    data = upload_res.json()
    assert data["filename"] == "test_doc.pdf"
    assert data["pages"] == 1
    assert data["chunks"] >= 1

    # 3. Mock RAG service response
    mock_rag = MagicMock()
    mock_rag.answer_question.return_value = {
        "answer": "Learning rate alpha controls the step size in gradient descent.",
        "sources": [{"filename": "test_doc.pdf", "page": 1}]
    }
    app.state.rag_service = mock_rag

    # 4. Ask a question
    ask_res = client.post("/ask", json={"question": "What is alpha?"})
    assert ask_res.status_code == 200
    res_data = ask_res.json()
    assert "Learning rate alpha" in res_data["answer"]
    assert res_data["sources"][0]["filename"] == "test_doc.pdf"
    assert res_data["sources"][0]["page"] == 1
