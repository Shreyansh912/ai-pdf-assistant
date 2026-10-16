from typing import List, Optional
from typing import Optional
from pydantic import BaseModel, Field
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)


class DocumentMetadata(BaseModel):
    document_id: Optional[str] = None
    filename: str
    page_count: int
    chunk_count: int

class DocumentUploadResponse(BaseModel):
    document_id: Optional[str] = None
    filename: str
    pages: int
    chunks: int
    total_indexed_chunks: int
    message: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total_documents: int
    total_chunks: int

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    document_id: Optional[str] = Field(
        None, description="Optional target document ID. If None, queries across all uploaded documents."
    )

class SourceCitation(BaseModel):
    filename: str
    page: int
    document_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
