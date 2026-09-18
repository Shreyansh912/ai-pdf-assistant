import re
from typing import List, Dict, Any


class TextChunker:
    """
    Splits page-level document text into overlapping chunks
    while preserving page citations and metadata.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """
        Splits a continuous string into chunks of approximately self.chunk_size
        with self.chunk_overlap, prioritizing sentence and paragraph boundaries.
        """
        if not text:
            return []

        text = text.strip()
        text_len = len(text)
        if text_len <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < text_len:
            end = start + self.chunk_size

            # If the remaining window reaches the end of the text
            if end >= text_len:
                remaining = text[start:].strip()
                if remaining:
                    chunks.append(remaining)
                break

            slice_to_search = text[start:end]
            split_idx = -1

            # Priority 1: Paragraph break (\n)
            last_newline = slice_to_search.rfind("\n")
            if last_newline > self.chunk_overlap:
                split_idx = last_newline

            # Priority 2: Sentence terminal (. ! ? followed by space)
            if split_idx == -1:
                sentence_matches = list(re.finditer(r"[\.!\?]\s+", slice_to_search))
                for match in reversed(sentence_matches):
                    # We cut after the punctuation and following whitespace
                    if match.end() > self.chunk_overlap:
                        split_idx = match.end()
                        break

            # Priority 3: Word boundary (space)
            if split_idx == -1:
                last_space = slice_to_search.rfind(" ")
                if last_space > self.chunk_overlap:
                    split_idx = last_space

            # Fallback: Hard cut if no natural punctuation or space is found
            if split_idx == -1:
                split_idx = len(slice_to_search)

            chunk_text = slice_to_search[:split_idx].strip()
            if chunk_text:
                chunks.append(chunk_text)

            # Advance starting pointer, preserving overlap
            next_start = start + split_idx - self.chunk_overlap
            if next_start <= start:
                # Prevent infinite loop on pathological or edge-case strings
                next_start = start + max(split_idx, 1)

            start = next_start

        return chunks

    def chunk_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes extracted page records and produces chunks with metadata.

        Args:
            pages_data: List of dicts with keys 'filename', 'page_number', 'text'

        Returns:
            List of chunk dicts:
            - chunk_id: str
            - filename: str
            - page_number: int
            - text: str
        """
        all_chunks: List[Dict[str, Any]] = []

        for page in pages_data:
            filename = page["filename"]
            page_num = page["page_number"]
            page_text = page.get("text", "")

            if not page_text:
                continue

            page_chunks = self._split_text(page_text)

            for idx, chunk_text in enumerate(page_chunks):
                chunk_id = f"{filename}_p{page_num}_c{idx}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "filename": filename,
                    "page_number": page_num,
                    "text": chunk_text
                })

        return all_chunks
