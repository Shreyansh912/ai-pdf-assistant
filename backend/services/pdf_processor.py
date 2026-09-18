import os
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF


class PDFProcessingError(Exception):
    """Custom domain exception raised when PDF extraction or validation fails."""
    pass


class PDFProcessor:
    """
    Handles PDF document loading, per-page text extraction, and text normalization.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted text:
        - Replaces Windows carriage returns (\r\n -> \n)
        - Normalizes irregular whitespace, tabs, and duplicate spaces
        - Collapses 3+ consecutive newlines to 2 to preserve paragraph breaks
        """
        if not text:
            return ""

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_text_by_page(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extracts readable text from each page of a given PDF.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            List of dictionaries, each containing:
            - filename: str
            - page_number: int (1-indexed)
            - text: str (cleaned text)

        Raises:
            PDFProcessingError: If the file is missing, invalid, corrupted,
                               or contains no extractable text.
        """
        if not os.path.exists(file_path):
            raise PDFProcessingError(f"File not found at path: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise PDFProcessingError("Unsupported file type: File must have a .pdf extension.")

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise PDFProcessingError(f"Failed to read or parse PDF file: {str(e)}")

        filename = os.path.basename(file_path)
        pages_data: List[Dict[str, Any]] = []
        total_extracted_characters = 0

        try:
            if doc.page_count == 0:
                raise PDFProcessingError("The PDF contains 0 pages.")

            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                raw_text = page.get_text("text")
                cleaned_text = self.clean_text(raw_text)

                if cleaned_text:
                    total_extracted_characters += len(cleaned_text)

                pages_data.append({
                    "filename": filename,
                    "page_number": page_index + 1,  # 1-indexed for end-user citation
                    "text": cleaned_text
                })

        except Exception as e:
            if not isinstance(e, PDFProcessingError):
                raise PDFProcessingError(f"Error during page text extraction: {str(e)}")
            raise e
        finally:
            doc.close()

        # Reject image-only, scanned, or empty PDFs
        if total_extracted_characters == 0:
            raise PDFProcessingError(
                "No readable text could be extracted. "
                "The PDF may be scanned, image-only, or empty. OCR is not supported."
            )

        return pages_data
