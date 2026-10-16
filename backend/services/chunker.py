from typing import List, Dict, Any


class TextChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, overlap: int = None):
        actual_overlap = overlap if overlap is not None else chunk_overlap

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if actual_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        if actual_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = actual_overlap
        self.overlap = actual_overlap

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []

        for page in pages:
            raw_text = page.get("text", "")
            if not raw_text or not raw_text.strip():
                continue

            clean_text = raw_text.strip()
            filename = page.get("filename", "document.pdf")
            page_no = page.get("page_number") or page.get("page", 1)

            step = self.chunk_size - self.chunk_overlap
            start = 0
            chunk_idx = 0

            while start < len(clean_text):
                end = start + self.chunk_size
                chunk_text = clean_text[start:end]

                chunk_entry = {
                    "chunk_id": f"{filename}_p{page_no}_c{chunk_idx}",
                    "filename": filename,
                    "page": page_no,
                    "page_number": page_no,
                    "text": chunk_text,
                }
                all_chunks.append(chunk_entry)
                chunk_idx += 1

                if end >= len(clean_text):
                    break
                start += step

        return all_chunks

    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.chunk_pages(pages)