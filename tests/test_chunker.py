import pytest
from backend.services.chunker import TextChunker


def test_chunker_validation():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=150)


def test_short_text_single_chunk():
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    pages = [{
        "filename": "ml_basics.pdf",
        "page_number": 1,
        "text": "Short page content under chunk size."
    }]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "ml_basics.pdf_p1_c0"
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["text"] == "Short page content under chunk size."


def test_chunking_with_overlap():
    chunker = TextChunker(chunk_size=60, chunk_overlap=15)
    sentence = (
        "Supervised learning trains on labeled data. "
        "Unsupervised learning looks for patterns in unlabeled data. "
        "Reinforcement learning optimizes reward signals in dynamic environments."
    )
    pages = [{
        "filename": "sample.pdf",
        "page_number": 3,
        "text": sentence
    }]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk["filename"] == "sample.pdf"
        assert chunk["page_number"] == 3
        assert chunk["chunk_id"] == f"sample.pdf_p3_c{i}"
        assert len(chunk["text"]) > 0


def test_empty_page_handling():
    chunker = TextChunker(chunk_size=800, chunk_overlap=150)
    pages = [{
        "filename": "sample.pdf",
        "page_number": 1,
        "text": ""
    }]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) == 0
