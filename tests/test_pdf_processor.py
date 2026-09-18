import os
import fitz
import pytest
from backend.services.pdf_processor import PDFProcessor, PDFProcessingError


@pytest.fixture
def processor():
    return PDFProcessor()


@pytest.fixture
def sample_pdf(tmp_path):
    """Creates a temporary valid 2-page PDF."""
    pdf_path = os.path.join(tmp_path, "sample_document.pdf")
    doc = fitz.open()

    p1 = doc.new_page()
    p1.insert_text((50, 72), "Machine Learning is a subset of artificial intelligence.")

    p2 = doc.new_page()
    p2.insert_text((50, 72), "Supervised learning uses labeled training datasets.")

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_text_pdf(tmp_path):
    """Creates a temporary PDF with no text (simulates blank or scanned doc)."""
    pdf_path = os.path.join(tmp_path, "empty.pdf")
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extract_text_success(processor, sample_pdf):
    results = processor.extract_text_by_page(sample_pdf)
    assert len(results) == 2
    assert results[0]["page_number"] == 1
    assert results[0]["filename"] == "sample_document.pdf"
    assert "Machine Learning is a subset" in results[0]["text"]
    assert results[1]["page_number"] == 2
    assert "Supervised learning" in results[1]["text"]


def test_empty_pdf_raises_error(processor, empty_text_pdf):
    with pytest.raises(PDFProcessingError) as exc_info:
        processor.extract_text_by_page(empty_text_pdf)
    assert "No readable text could be extracted" in str(exc_info.value)


def test_invalid_extension_raises_error(processor, tmp_path):
    txt_file = os.path.join(tmp_path, "notes.txt")
    with open(txt_file, "w") as f:
        f.write("Just text")
    with pytest.raises(PDFProcessingError) as exc_info:
        processor.extract_text_by_page(txt_file)
    assert "Unsupported file type" in str(exc_info.value)
